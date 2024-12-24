# Source Code for FPG

Ruiyang Xia, Dawei Zhou, Decheng Liu, Lin Yuan, Shuodi Wang, Jie Li, Nannan Wang, Xinbo Gao. Advancing Generalized Deepfake Detector with Forgery Perception Guidance. ACM International Conference on Multimedia (MM '24), 6676–6685, 2024. https://doi.org/10.1145/3664647.3680713

# Prepraration

### Dependencies

The project's runtime environment is based on Miniconda. You can use the following command to install the project's runtime environment：

``conda create --name FPG --file requirements.txt``

### FIQA model

First, the pre-trained face image quality assessment model checkpoints is downloaded from the [GoogleDrive](https://drive.google.com/file/d/1AM0iWVfSVWRjCriwZZ3FXiUGbcDzkF25/view) of TFace repository and put in ``src/utils/qnet/model/pth`` .

### Datasets

Our training is conducted on the FaceForensics++ dataset, which can be downloaded from the repository [FaceForensics](https://github.com/ondyari/FaceForensics) and place it in the ``data/``.

# Training
If you have followed the previous steps to prepare, simply use `bash train.sh` to start the training process. If you wish to modify any configurations, you can review the parameter settings in the `src/train.py` file and add them to the `train.sh` script.
