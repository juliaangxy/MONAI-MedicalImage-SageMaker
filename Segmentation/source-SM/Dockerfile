FROM 763104351884.dkr.ecr.ap-southeast-1.amazonaws.com/pytorch-training:1.8.0-gpu-py3


# Update Python to 3.8
RUN apt update
RUN apt install python3.8 -y
RUN rm /usr/bin/python3
RUN ln -s python3.8 /usr/bin/python3

# Install sagemaker-training toolkit that contains the common functionality necessary to create a container compatible with SageMaker and the Python SDK.

RUN pip install monai[all]==0.8.0

ENV PATH="/opt/ml/code:${PATH}"

# /opt/ml and all subdirectories are utilized by SageMaker, we use the /code subdirectory to store our user code.
# COPY /code /opt/ml/code
COPY code/* /opt/ml/code/

# this environment variable is used by the SageMaker PyTorch container to determine our program entry point
# for training and serving.
# For more information: https://github.com/aws/sagemaker-pytorch-container
ENV SAGEMAKER_PROGRAM main.py

# RUN bash install.sh
