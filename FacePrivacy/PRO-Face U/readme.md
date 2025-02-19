# Source Code for PRO-Face-S

PRO-Face U: Real-World Applications of Utility-Guided Reversible Anonymization of Facial Images is currently in preparation.

# Prepraration
This code proposes a new paradigm for facial privacy protection. By injecting control conditions and keys into reversible neural networks, it can generate three types of facial anonymization images with different properties, including identity-preserving anonymization, visual feature-preserving anonymization, and dual anonymization, thus achieving the diversity of privacy protection. In addition, it can reversibly restore the anonymized images securely through the key, ensuring the security and reversibility of privacy protection.

### Dependencies

The project's runtime environment is based on Miniconda. You can use the following command to install the project's runtime environment：

``conda create --name PROFaceU --file requirements.txt``

### Face classification models
First, download the pretrained face classification checkpoints from any of the following links.
- [BaiduDisk link](https://pan.baidu.com/s/1C7U_VasAV6FuG5_4E9pKbw ) (Password:`grdw`)

Then, place the entire folder checkpoints under the `face`.

### SimSwap models

To run SimSwap, you need to download its pretrained models from the following link:
- [BaiduDisk link](https://pan.baidu.com/s/1q-s1G4aqSzcXEofDOEfeHg) (Password:`3cvh`)

Then, place the file `arcface_checkpoint.tar` under `SwimSwap/arcface_model` and the three files `latest_net_*.pth` under `SwimSwap/models/checkpoints/people/`.

### Datasets
Our training was done is CelebA dateset, where all faces images were preprocessed to keep only the facial part. we have made our preprocessed CelebA dataset public. One many obtain the entire datasets (including the train/val/test splits and triplet files) from the following links:
- [BaiduDisk link](https://pan.baidu.com/share/init?surl=wMf-iRP5kVfeijvvZYOylQ) (Password: `dkhd`)
- [OneDrive](https://cqupteducn-my.sharepoint.com/:u:/g/personal/yuanlin_cqupt_edu_cn/EckcBzUQ-f1EgobKZGzJKPUB_g_SOxCXv5bF7e6Kx3O8Yw?e=wInwoU)

Then, place it in the directory specified by `dataset_dir` in the `config.py` file, or alternatively, modify the `dataset_dir` path to point to the location of the dataset

# Training
Simply run `train_with_utility.py` to start the training process.

# Testing
Simply run `test_with_utility.py` to start the testing process.

# Trained model
You can download our trained model from this [BaiduDisk link](https://pan.baidu.com/s/1q-s1G4aqSzcXEofDOEfeHg) (Password:`3cvh`).
Then, place the file `hybridAll_inv3_recTypeRandom_secretAsNoise_TripMargin1.2_ep12_iter15000.pth` under `experiments/checkpoints_256/`.

