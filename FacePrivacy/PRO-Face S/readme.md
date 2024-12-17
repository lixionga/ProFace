# Source Code for PRO-Face-S

Lin Yuan, Kai Liang, Xiao Pu, Yan Zhang, Jiaxu Leng, Tao Wu, Nannan Wang, Xinbo Gao. Invertible Image Obfuscation for Facial Privacy Protection via Secure Flow. IEEE Transactions on Circuits and Systems for Video Technology. Volume: 34, Issue: 7, July 2024，6077-6091. https://doi.org/10.1109/TCSVT.2023.3344809

# Prepraration

### Dependencies

The project's runtime environment is based on Miniconda. You can use the following command to install the project's runtime environment：

``conda create --name PROFaceS --file requirements.txt``

### Pretrained model
To run **ProFaceS**, you need to download the required pre-trained models from the following link:
- [BaiduDisk link](/https://pan.baidu.com/s/1q-s1G4aqSzcXEofDOEfeHg) (Password:`3cvh`)

After downloading the models, you should place them in the following directory structure:
......
### Datasets



# Training
Simply run `train_tcsvt.py` to start the training process. 

# Testing
Simply run `test_tcsvt.py` to start the testing process. 



# Acknowledgement

Please cite our paper via the following BibTex if you find it useful. Thanks. 

    @ARTICLE{10366303,
    author={Yuan, Lin and Liang, Kai and Pu, Xiao and Zhang, Yan and Leng, Jiaxu and Wu, Tao and Wang, Nannan and Gao, Xinbo},
    journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
    title={Invertible Image Obfuscation for Facial Privacy Protection via Secure Flow}, 
    year={2024},
    volume={34},
    number={7},
    pages={6077-6091},
    keywords={Privacy;Face recognition;Security;Data privacy;Visualization;Information integrity;Information filtering;Privacy protection;face anonymization;image obfuscation;invertible;security},
    doi={10.1109/TCSVT.2023.3344809}
    }


If you have any question, please don't hesitate to contact us by ``yuanlin@cqupt.edu.cn``.