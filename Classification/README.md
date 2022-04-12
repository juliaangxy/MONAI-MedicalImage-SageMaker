# CovidCTClassification

In this example we will demonstrate how to integrate the [MONAI](http://monai.io) framework into Amazon SageMaker, use manifest json file as labeling, and give example code of MONAI pre-processing transforms and neural network (DenseNet) that you can use to train a medical image classification model using DICOM images directly.  Please also visit [Build a medical image analysis pipeline on Amazon SageMaker using the MONAI framework](https://aws.amazon.com/blogs/industries/build-a-medical-image-analysis-pipeline-on-amazon-sagemaker-using-the-monai-framework/) for additional details on how to deploy the MONAI model, pipe input data from S3, and perform batch inferences using SageMaker batch transform.

For more information about the PyTorch in SageMaker, please visit [sagemaker-pytorch-containers](https://github.com/aws/sagemaker-pytorch-containers) and [sagemaker-python-sdk](https://github.com/aws/sagemaker-python-sdk) github repositories.

The dataset is obtained from this [source COVID-CT-MD](https://github.com/ShahinSHH/COVID-CT-MD), with the total dataset contains volumetric chest CT scans (DICOM files) of 169 patients positive for COVID-19 infection, 60 patients with CAP (Community Acquired Pneumonia), and 76 normal patients. For this demo, only 26 images are selected. 

# Content
+ In training_deplying_model.ipynb notebooks, I will introduce a typical ML  workflow in SageMaker which covers data preparation --> model training --> model deployment --> Inference. 
+ In Inference-explanation.ipynb notebook, I will introduce a few options of deployments, and dive deep into what is happening inside the container. 

model.tar.gz/
|- model.pth
|- source/
  |- inference.py
  |- requirements.txt  # only for versions 1.3.1 and higher
