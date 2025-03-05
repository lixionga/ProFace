# Source Code for PRO-Face-C

Source code for PRO-Face C: Privacy-preserving Recognition of Obfuscated Face via Feature Compensation (IEEE TIFS 2024: https://ieeexplore.ieee.org/document/10499238)

# Introduction
The PRO-Face-C framework achieves high-precision recognition of obfuscated facial images by employing an end-cloud collaborative feature compensation mechanism. At its core, the system integrates a suite of advanced techniques, including hybrid blurring, block-based feature map channel shuffling, feature map quantization, and pruning. These methodologies are synergistically designed to safeguard the user's original facial characteristics from being overtly revealed, all the while maintaining a high level of recognition accuracy.
Architecturally, PRO-Face-C is structured around a client-server model. The client's role is to acquire the facial image, apply the necessary obfuscation, and transmit both the blurred image and the auxiliary feature map to the server. Upon receipt, the server harnesses a pre-trained facial recognition model, complemented by a feature compensation module, to perform recognition with enhanced precision. This collaborative approach ensures a robust and efficient system for managing privacy concerns while upholding the integrity of facial recognition tasks.

An overview of our network architecture:
![image](https://github.com/user-attachments/assets/cb28c752-fee9-4325-b7c2-316484e571fe)


# Prepraration

### Dependencies

The project's runtime environment is based on Miniconda. You can use the following command to install the project's runtime environment：

``conda create --name PROFaceC --file requirements.txt``

### Server model

First, the pre-trained face recognition model is downloaded from the [AdaFace](https://github.com/mk-minchul/AdaFace) repository as the server-side model and put in ``model/`` .

### Client model

We have put the pre-trained client-side model  ``Backbone.pth`` into ``model/ckpt``.


### Datasets

Our training is done on the CelebA dataset, where all face images are preprocessed to keep only the facial parts with a resolution of 112×112. You can download the dataset from the following link and place it in the `` Data/``.

- [GoogleDrive](https://drive.google.com/drive/folders/1U9GpW-HbYnj1HyTxGdLmL__LB7IEDgWp?usp=drive_link)
- [BaiduDisk](https://pan.baidu.com/s/1IAoE4Q0MecIXGxH0DR-NHQ?pwd=msv2)(Password: msv2)

The validation datasets LFW, CPLFW, CFP-FP, CALFW, AgeDB can be downloaded from [face.evoLVe](https://github.com/ZhaoJ9014/face.evoLVe) and put into the `` Data/validation`` .  Each dataset needs to have `` npy`` ,  `` bin``  files and corresponding binary images.

Note that the file path of the validation dataset can be modified in ``validation\config.py`` and the file path of the training dataset can be modified in ``train.yaml``.


# Training

Simply run ``train.py`` to start the training process. The script ``torchkit\task\base_task.py`` will load ``train.yaml``.
for necessary configuration. The config file should specify the following key information: 

- **Dataset**: the paths for the training dataset are specified by the three options ``DATA_ROOT``, ``INDEX_ROOT``, and ``DATASETS`` in the ``train.yaml`` file. The paths for the validation datasets can be found in the ``VAL_DATA_ROOT`` option of the same file.  If you have followed the previous steps to organize your dataset, you don't need to modify these paths.
- **Server model:** the cloud-side server model utilizes the models provided by the  [AdaFace](https://github.com/mk-minchul/AdaFace) repository: ``adaface_ir101_webface12m.ckpt, adaface_ir50_webface4m.ckpt``.

Other part of the training script is self-explained. We are open for any questions.

# Testing

Simply run ```validation\val_main.py``` to initiate the evaluation process and obtain the accuracy of face recognition on blurred images. If you have trained your own client model, you can modify the `backbone_resume` option in `validation\config.py`.


# Acknowledgement

PRO-Face-C code borrowed some code from [mk-minchul/AdaFace](https://github.com/mk-minchul/AdaFace),  [Tencent/TFace](https://github.com/Tencent/TFace) and [ZhaoJ9014/face.evoLVe](https://github.com/ZhaoJ9014/face.evoLVe) , I would like to express my sincere gratitude for their contributions to this project.

Please cite our paper via the following BibTex if you find it useful. Thanks. 

    @ARTICLE{10499238,
    author={Yuan, Lin and Chen, Wu and Pu, Xiao and Zhang, Yan and Li, Hongbo and Zhang, Yushu and Gao, Xinbo and Ebrahimi, Touradj},
    journal={IEEE Transactions on Information Forensics and Security}, 
    title={PRO-Face C: Privacy-preserving Recognition of Obfuscated Face via Feature Compensation}, 
    year={2024},
    volume={},
    number={},
    pages={1-1},
    keywords={Face recognition;Privacy;Image recognition;Visualization;Data privacy;Servers;Information integrity;Face recognition;Image obfuscation;Privacy protection;Utility},
    doi={10.1109/TIFS.2024.3388976}
    }

If you have any question, please don't hesitate to contact us by ``yuanlin@cqupt.edu.cn``.
