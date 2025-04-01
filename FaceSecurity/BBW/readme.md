### Datasets
Our training was done is CelebA dateset, where all faces images were preprocessed to keep only the facial part. we have made our preprocessed CelebA dataset public. One many obtain the entire datasets (including the train/val/test splits and triplet files) from the following links:
- [BaiduDisk link](https://pan.baidu.com/share/init?surl=wMf-iRP5kVfeijvvZYOylQ) (Password: `dkhd`)
- [OneDrive](https://cqupteducn-my.sharepoint.com/:u:/g/personal/yuanlin_cqupt_edu_cn/EckcBzUQ-f1EgobKZGzJKPUB_g_SOxCXv5bF7e6Kx3O8Yw?e=wInwoU)

Then, place it in the directory specified by `dataset_dir` in the `config.py` file, or alternatively, modify the `dataset_dir` path to point to the location of the dataset

# Testing
Simply run `test.py` to start the testing process.If you have trained your own client model, you can modify the checkpoints option in config\config.py.

# Trained model
You can download our trained model from this [BaiduDisk link](https://pan.baidu.com/s/1q-s1G4aqSzcXEofDOEfeHg) (Password:`3cvh`).
Then, place the file `hybridAll_inv3_recTypeRandom_secretAsNoise_TripMargin1.2_ep12_iter15000.pth` under `experiments/checkpoints_256/`.