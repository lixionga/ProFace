# Source Code for PRO-Face
Our paper can be downloaded from [ACM website](https://dl.acm.org/doi/10.1145/3503161.3548202). 

# Introduction
- Supports multiple obfuscation methods for flexible privacy protection.
- Enables accurate identity recognition from obfuscated images, balancing privacy and utility.
- Achieves recognition with pre-trained models across various image types.
# Prepraration
### Dependencies
All dependencies we use from this implementation is listed in ``requirements.txt``.

### Face recognition models
First, download the pretrained face recognizer checkpoints from any of the following links.
- [BaiduDisk link](https://pan.baidu.com/s/1QTybIrA3XjiKmwtU0yyXMg) (Password: ``byhx``) 
- [OneDrive link](https://cqupteducn-my.sharepoint.com/:f:/g/personal/yuanlin_cqupt_edu_cn/Eq7QW_8bW6RJtUUcagbzxjYBL514v2zwCvlW9tvQb9dUVg?e=ln5ObN)

Then, place the entire folder ``checkpoints`` under the ``face/``.

### FaceShifter models
To run FaceShifter, you need to download its pretrained models from the following link:

- [BaiduDisk link](https://pan.baidu.com/s/1JBk6TH1chhx0P4XxgQkQyQ) (Password: ``53is``)
- [OneDrive link](https://cqupteducn-my.sharepoint.com/:f:/g/personal/yuanlin_cqupt_edu_cn/En548_xYLfRHjPirNfFbp9kBrMVj-1NJQ3kswd8j_FdsyQ?e=UccY83)

Then, place the file ``model_ir_se50.pth`` under ``FaceShifter/face_modules/`` and file ``G_latest.pth`` under ``FaceShifter/saved_models/``.


### SimSwap models
To run SimSwap, you need to download its pretrained models from the following link:
- [BaiduDisk for SimSwap models](https://pan.baidu.com/s/1gKHj_ca8uvFeGhVyDuHyJA) (Password: ``f2bk``)
- [OneDrive link](https://cqupteducn-my.sharepoint.com/:f:/g/personal/yuanlin_cqupt_edu_cn/Evt4Ks4XOxBMpQJPTHLNx5IBzsXsvPHwIddG13B-pJVGVQ?e=XuQJ2H)

Then, place the file ``arcface_checkpoint.tar`` under ``SwimSwap/arcface_model`` and the three files 
``latest_net_*.pth`` under ``SwimSwap/models/checkpoints/people/``.


### Datasets
Our training was done is CelebA dateset, where all faces images were preprocessed to keep only the facial part. 
We rely on the landmark annotations provided by CelebA to perform image crop, instead of running face detection. 
The script for cropping the CelebA dataset is provided as ``dataset/align_crop_celeba.py``.
Besides, we provide the script to create triplet datasets for CelebA: ``dataset/triplet_dataset.py``.
You may use this script to preprocess any of your dataset.

To assist other researchers, we have made our preprocessed CelebA dataset public. One many obtain the entire datasets (including the train/val/test splits and triplet files) from the following links:
- [BaiduDisk](https://pan.baidu.com/s/1wMf-iRP5kVfeijvvZYOylQ) (Password: dkhd)
- [OneDrive](https://cqupteducn-my.sharepoint.com/:u:/g/personal/yuanlin_cqupt_edu_cn/EckcBzUQ-f1EgobKZGzJKPUB_g_SOxCXv5bF7e6Kx3O8Yw?e=wInwoU)


# Training
Simply run ``train.py`` to start the training process. The script ``train.py`` will load ``config/config.py`` 
for necessary configuration. The config file should specify the following key information: 
- **Dataset:** path to the training/validation dataset (``dataset_dir``) and the target image dataset (``target_img_dir_train``)
- **Face recognizer:** name of the face recognizer, selected from five options, i.e., ``MobileFaceNet, InceptionResNet, IResNet50, IResNet100``, and ``SEResNet50``
- **Obfuscator：** option of the obfuscator, selected from four options, i.e., ``blur_31_2_8, pixelate_7, faceshifter, simswap``. 
The three numbers after blur means the kernel_size, min sigma and max sigma respectively. The number after pixelate means the mean pixelate block size.

Other part of the training script is self-explained. We are open for any questions.

# Testing
Simly run ``evaluate.py`` to start an evaluation process, which generates protected images and runs face recognition 
on the generated images. The script will plot the ROC curves for the recognition results. One can specify the path to 
saved images as preferred.

To promote other research, we provide our trained models corresponding to the 4 obfuscators and 5 recognizers under 
folder ``checkpoints/``. The naming convention of each checkpoint file is ``<obfuscator>_<recognizer>_<epoch>_BEST.pth``.

Note that, our training on SimSwap was done on image with resolution 224x224, while others were on 112x112. 
Therefore, the provided checkpoint for SimSwap is still with resolution 224x224 and you can see an IF condition
for SimSwap in the evaluation script. 
However, the training script we are providing is a updated version that unifies the resolution issue.


# Acknowledgement
Please cite our paper via the following BibTex if you find it useful. Thanks. 

    @inproceedings{10.1145/3503161.3548202,
    author = {Yuan, Lin and Liu, Linguo and Pu, Xiao and Li, Zhao and Li, Hongbo and Gao, Xinbo},
    title = {PRO-Face: A Generic Framework for Privacy-Preserving Recognizable Obfuscation of Face Images},
    year = {2022},
    isbn = {9781450392037},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    url = {https://doi.org/10.1145/3503161.3548202},
    doi = {10.1145/3503161.3548202},
    booktitle = {Proceedings of the 30th ACM International Conference on Multimedia},
    pages = {1661–1669},
    numpages = {9},
    keywords = {face recognition, image fusion, privacy protection, face obfuscation},
    location = {Lisboa, Portugal},
    series = {MM '22}
    }

If you have any question, please don't hesitate to contact us by ``yuanlin@cqupt.edu.cn``.
