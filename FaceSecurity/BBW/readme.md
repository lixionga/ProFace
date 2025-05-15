### Dependencies and Installation
- Python 3 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux)).
- [PyTorch = 1.9.0](https://pytorch.org/) .
- See requirements.yaml for other dependencies.


### Datasets
Our training was done is FFHQ dateset, where all faces images were preprocessed to size 256*256. we have made our preprocessed FFHQ dataset public. One many obtain the datasets (including the train/val/test splits) [here](https://drive.google.com/file/d/14NJtQEhs8jWqtX-WT2WgaLGDByDu5JcQ/view?usp=drive_link).

Then, place it in the directory specified by `TRAIN_PATH`,`VAL_PATH`,`TEST_PATH` in the `cfg.py` file, or alternatively, modify all these paths to point to the location of the dataset.

### Testing
We provide the pre-trained model [here](https://drive.google.com/file/d/1t2cogWi5e421f51BSyH0tEqbSD486kOm/view?usp=drive_link). Simply run `test.py` to start the testing process. If you want to try different common transformations, you can modify  `transformations` values in test.py. Due to license agreements, we are unable to distribute deepfakes codes ourselves.
You can implement [SimSwap](https://github.com/neuralchen/SimSwap), [SelfBlend](https://github.com/mapooon/SelfBlendedImages), [StarGAN2](https://github.com/clovaai/stargan-v2), [FaceShifter](https://github.com/Heonozis/FaceShifter-pytorch) and place them into   `network/distortions/deepfakes`


### Trained model
Then, place the file `` under `results/train/`.