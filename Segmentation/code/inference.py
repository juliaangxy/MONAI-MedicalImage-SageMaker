# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
# from monai.utils import first, set_determinism
import numpy
from monai.transforms import (
    AsDiscrete,
    AsDiscreted,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    ScaleIntensityRanged,
    Spacingd,
    EnsureTyped,
    EnsureType,
    Invertd,
)
from monai.handlers.utils import from_engine
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.metrics import DiceMetric
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch
from monai.config import print_config
from monai.apps import download_and_extract
import torch
import matplotlib.pyplot as plt
import tempfile
import shutil
import os, sys, glob, argparse, json, subprocess
import logging
from pathlib import Path
import boto3



logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


## load model artifact here
def model_fn(model_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm=Norm.BATCH
    ).to(device) 

    print("model_dir is", model_dir)
    print("inside model_dir is", os.listdir(model_dir))
    with open(os.path.join(model_dir, 'model.pth'), 'rb') as f:
        model.load_state_dict(torch.load(f,map_location=torch.device('cpu') ))
        print("model load with cpu!")
    return model.to(device)   


## define data loader for validation dataset
## Notice: val_files including both original image as well as label
## further work should be done in the situation without labels
def get_val_data_loader(val_files):
    ## define transform for validation 
    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Spacingd(keys=["image", "label"], pixdim=(
                1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            ScaleIntensityRanged(
                keys=["image"], a_min=-57, a_max=164,
                b_min=0.0, b_max=1.0, clip=True,
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            EnsureTyped(keys=["image", "label"]),
        ]
    )
    
    val_ds = CacheDataset( data=val_files, transform=val_transforms, cache_rate=1.0)
    val_loader = DataLoader(val_ds, batch_size=1)

    return val_loader

## input is in json, to indicate the location of testing files in S3
JSON_CONTENT_TYPE= 'application/json'

#output is in numpy, which is transformed from tensor as direct output from model
NUMPY_CONTENT_TYPE = 'application/json'



s3_client = boto3.client('s3')
s3 = boto3.resource('s3') # assumes credentials & configuration are handled outside python in .aws directory or environment variables
## function to download the whole folder
def download_s3_folder(bucket_name, s3_folder, local_dir=None):
    """
    Download the contents of a folder directory
    Args:
        bucket_name: the name of the s3 bucket
        s3_folder: the folder path in the s3 bucket
        local_dir: a relative or absolute directory path in the local file system
    """
    bucket = s3.Bucket(bucket_name)
    for obj in bucket.objects.filter(Prefix=s3_folder):
        target = obj.key if local_dir is None \
            else os.path.join(local_dir, os.path.relpath(obj.key, s3_folder))
        if not os.path.exists(os.path.dirname(target)):
            os.makedirs(os.path.dirname(target))
        if obj.key[-1] == '/':
            continue
        bucket.download_file(obj.key, target)
    return


def input_fn(serialized_input_data, content_type):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Received request of type:{content_type}")
    
    print("serialized_input_data is---", serialized_input_data)
    if content_type == 'application/json':
        
        #data = flask.request.data.decode('utf-8')
        data = json.loads(serialized_input_data)
        
        
        bucket=data['bucket']
        s3_folder=data['key']## prefix with all the image files as well as labelings
        
        ## Download the folder from s3 
        print("bucket:" , bucket, " key is: ",s3_folder)
        
        # download into local folder
        local_dir="tmp"
        download_s3_folder(bucket, s3_folder, local_dir=local_dir)
        
        ## define key for image and labels
        images = sorted(glob.glob(os.path.join(local_dir, "imagesTr", "*.nii.gz")))
        labels = sorted(glob.glob(os.path.join(local_dir, "labelsTr", "*.nii.gz")))
        
        if(len(images)==len(labels)):
            data_dicts = [{"image": image_name, "label": label_name} for image_name, label_name in zip(images, labels)]
        
        
            print('Download finished!')
            print('Start to inference for the files and labels >>> ', data_dicts)


            val_loader = get_val_data_loader(data_dicts)
            print('get_val_data_loader finished!')


            for i, val_data in enumerate(val_loader):
                val_inputs, val_labels = (
                    val_data["image"].to(device),
                    val_data["label"].to(device),
                )
            
            shutil.rmtree(local_dir)
            print('removed the downloaded files after loading them!')
            
            return val_inputs
        else:
            raise Exception('Inputs for Labels and Images are not matched:  ', len(images), "!= ", len(labels))



    else:
        raise Exception('Requested unsupported ContentType in Accept: ' + content_type)
        return


def predict_fn(input_data, model):
    print('Got input Data: {}'.format(input_data))
    print("input_fn in predict:",input_data)
    #print(inputs[0,0,1])## debugging purpose
    model.eval()
    
    roi_size = (160, 160, 160)
    sw_batch_size = 4
    val_outputs = sliding_window_inference(input_data, roi_size, sw_batch_size, model)
    print("response from modeling prediction is", val_outputs.shape)
    return val_outputs


def output_fn(prediction_output, accept=JSON_CONTENT_TYPE):
    
    print("accept is:", accept)
    if accept == JSON_CONTENT_TYPE:
        print("response in output_fn is", prediction_output)
        pred = torch.argmax(prediction_output, dim=1).detach().cpu()[0, :, :, 80].tolist()
        inference_result = { 'pred': pred}
        
        print("inference_result is: ", inference_result)
        return inference_result

    raise Exception('Requested unsupported ContentType in Accept: ' + accept)

