# Source Code for FPG

Ruiyang Xia, Dawei Zhou, Decheng Liu, Lin Yuan, Shuodi Wang, Jie Li, Nannan Wang, Xinbo Gao. Advancing Generalized Deepfake Detector with Forgery Perception Guidance. ACM International Conference on Multimedia (MM '24), 6676–6685, 2024. https://doi.org/10.1145/3664647.3680713

# Introduction

# Prepraration

### Dependencies

The project's runtime environment is based on Miniconda. You can use the following command to install the project's runtime environment：

``conda create --name FPG --file requirements.txt``

### FIQA model

First, the pre-trained face image quality assessment model checkpoints is downloaded from the [GoogleDrive](https://drive.google.com/file/d/1AM0iWVfSVWRjCriwZZ3FXiUGbcDzkF25/view) of TFace repository and put in ``src/utils/qnet/model/pth`` .

### Datasets

Our training is conducted on the FaceForensics++ dataset, which can be downloaded from the repository [FaceForensics](https://github.com/ondyari/FaceForensics) and place it in the ``data/``.

The validation datasets FFIW can be downloaded from [tfzhou.FFIW](https://github.com/tfzhou/FFIW) and put into the `` inference/models`` .

# Training
If you have followed the previous steps to prepare, simply use `bash train.sh` to start the training process. If you wish to modify any configurations, just modify them in the `train.sh` script.The final trained model weights will be saved under the ``output`` directory.

# Testing

The script `test.sh` has a section following the `-w` flag that needs to be replaced with the path to the model. After making this change, you can run `bash test.sh` to start the testing procedure.
# Trained model

You can download our trained model from this [BaiduDisk link](https://pan.baidu.com/s/1QNG3aPI0N3KTsqeHhI9CMA) (Password:`7jem`).

Then, place it anywhere because you can modify the path to load the model in the script file.
# Acknowledgement

Please cite our paper via the following BibTex if you find it useful. Thanks. 

    @inproceedings{10.1145/3664647.3680713,
    author = {Xia, Ruiyang and Zhou, Dawei and Liu, Decheng and Yuan, Lin and Wang, Shuodi and Li, Jie and Wang, Nannan and Gao, Xinbo},
    title = {Advancing Generalized Deepfake Detector with Forgery Perception Guidance},
    year = {2024},
    isbn = {9798400706868},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    url = {https://doi.org/10.1145/3664647.3680713},
    doi = {10.1145/3664647.3680713},
    abstract = {One of the serious impacts brought by artificial intelligence is the abuse of deepfake techniques. Despite the proliferation of deepfake detection methods aimed at safeguarding the authenticity of media across the Internet, they mainly consider the improvement of detector architecture or the synthesis of forgery samples. The forgery perceptions, including the feature responses and prediction scores for forgery samples, have not been well considered. As a result, the generalization across multiple deepfake techniques always comes with complicated detector structures and expensive training costs. In this paper, we shift the focus to real-time perception analysis in the training process and generalize deepfake detectors through an efficient method dubbed Forgery Perception Guidance (FPG). In particular, after investigating the deficiencies of forgery perceptions, FPG adopts a sample refinement strategy to pertinently train the detector, thereby elevating the generalization efficiently. Moreover, FPG introduces more sample information as explicit optimizations, which makes the detector further adapt the sample diversities. Experiments demonstrate that FPG improves the generality of deepfake detectors with small training costs, minor detector modifications, and the acquirement of real data only. In particular, our approach not only outperforms the state-of-the-art on both the cross-dataset and cross-manipulation evaluation but also surpasses the baseline that needs more than 3\texttimes{} training time.},
    booktitle = {Proceedings of the 32nd ACM International Conference on Multimedia},
    pages = {6676–6685},
    numpages = {10},
    keywords = {deepfake detection, forgery perception, training process},
    location = {Melbourne VIC, Australia},
    series = {MM '24}
    }

If you have any question, please don't hesitate to contact us by ``yuanlin@cqupt.edu.cn``.