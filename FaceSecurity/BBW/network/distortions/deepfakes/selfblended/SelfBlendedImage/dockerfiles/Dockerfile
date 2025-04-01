FROM nvidia/cuda:11.4.1-runtime-ubuntu20.04


RUN apt-get -y update && \
    apt-get -y install python3

RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y libgl1-mesa-dev && \
    apt-get install -y cmake && \
    apt-get -y install python3-pip && \
    apt-get -y update && apt-get install -y libopencv-dev \
    apt-get install -y git


RUN pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html && \
    pip3 install dlib imutils scipy pandas opencv-python tqdm pretrainedmodels imgaug efficientnet_pytorch&&\
    pip3 install -U retinaface_pytorch &&\
    apt-get clean

WORKDIR /app 
