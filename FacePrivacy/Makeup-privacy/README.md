# Requirements
PyTorch 1.9.1
torchvision 0.10.1

# Introduction

# Prepare data
 1. For test phase-Download the test data set to this path：
```
  Makeup-privacy-main/Dataset-test
```
# Pre-trained model weights
 1. Please download four pre-trained face recognition model weights to this path：

```
  Makeup-privacy-main/Pretrained_FR_Models/
```

 2. Please download the pre-trained face parsing model weights to this path：
```
  Makeup-privacy-main/models/networks/face_parsing/
```
 3. Download the trained model weights to this path：
```
  Makeup-privacy-main/checkpoints/1_facenet_multiscale=2/
```

# Train
```
  python train.py
```
# Test
## 1:1 face verification test

 - Using the CelebA-HQ face dataset：

	Set the `--source_dir` path to `"./Dataset-test/CelebA-HQ"`

 - Using the LADN face dataset：

	Set the `--source_dir` path to `"./Dataset-test/LADN/before"`
```
  python test121.py
```
## 1:N face retrieval test

 - Use the LFW face dataset：
 
	 Please set the parameter in `test` to `opt.lfw_data_path`
 
 - Using the CelebA face dataset
	
	 Please set the parameter in `test` to `opt.celeba_data_path`	
```
  python test12n.py
```

