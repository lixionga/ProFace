# About platform
## Web project environment
  Since conflicts may occur among some dependent packages when configuring the environment, it is necessary to install them step by step
    conda create -n env_name python=3.10
    conda activate env_name
    pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1
    pip install -r requirements.txt
    pip install pip==24.0
    pip install omegaconf==2.0.5 fairseq==0.12.2
### Quick start
Run this script: app.py
## About -Face Forgery Detection- Module
When users upload a face image, this module will crop the face area and return the probability value of the face image being forged as well as the heatmap
### Environment
For details, please refer to EG under the model module
### attention
The image formats allowed to upload are: 'png', 'jpg', 'jpeg'.
Some Linux systems will cause Chinese garbled characters when decompressing with the'zip 'command, which only affects the display of some pictures and does not affect the operation of the function.
If this problem occurs, please upload the still picture separately after re-decompression on another platform. Or try another decompression command and specify the encoding as GBK.
Static images directory: static/images
## About -Text Sentiment Analysis- Module
This module is used to detect the emotional components of the input sentence. For details, please refer to nlp_structbert_emotion-classification_chinese-base under the model module
