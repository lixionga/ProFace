# MLEP for AI-generated image detection

## Environment Setup

```sh
conda env create -f environment.yaml
```
Due to the large number of environment packages, the "solving environment" step of creating environment may be time-consuming.

## Getting the Test data

|                | paper  | Url  |
|:--------------:|:------:|:----:|
| Test Dataset1  | [CNNDetection CVPR2020](https://github.com/PeterWang512/CNNDetection)             | [Baidudrive](https://pan.baidu.com/s/1l-rXoVhoc8xJDl20Cdwy4Q?pwd=ft8b)                                    |
| Test Dataset2  | [FreqNet AAAI2024](https://github.com/chuangchuangtan/FreqNet-DeepfakeDetection)  | [googledrive](https://drive.google.com/drive/folders/11E0Knf9J1qlv2UuTnJSOFUjIIi90czSj?usp=sharing)       |
| Test Dataset3  | [DIRE ICCV2023](https://github.com/ZhendongWang6/DIRE)                            | [googledrive](https://drive.google.com/drive/folders/1jZE4hg6SxRvKaPYO_yyMeJN_DOcqGMEf?usp=sharing)       |
| Test Dataset4  | [UniversalFakeDetect CVPR2023](https://github.com/Yuheng-Li/UniversalFakeDetect)  | [googledrive](https://drive.google.com/drive/folders/1nkCXClC7kFM01_fqmLrVNtnOYEFPtWO-?usp=sharing)       |
| Test Dataset5  | [Diffusion1kStep](https://github.com/chuangchuangtan/NPR-DeepfakeDetection)       | [googledrive](https://drive.google.com/drive/folders/14f0vApTLiukiPvIHukHDzLujrvJpDpRq?usp=sharing)       |

You can also create a dataset with few images to test, just follow a similar folder structure below, and modify the dataroot in 'test.py'.

## Directory structure
<details>
<summary> Click to expand the folder tree structure. </summary>

```
datasets
|-- TrainDatasets
|   |-- train
|   `-- val
|   |-- test
`-- TestDatasets
    |-- ForenSynths_test       # Table1
    |   |-- biggan
    |   |-- cyclegan
    |   |-- deepfake
    |   |-- gaugan
    |   |-- progan
    |   |-- stargan
    |   |-- stylegan
    |   `-- stylegan2
    |-- GANGen-Detection     # Table2
    |   |-- AttGAN
    |   |-- BEGAN
    |   |-- CramerGAN
    |   |-- InfoMaxGAN
    |   |-- MMDGAN
    |   |-- RelGAN
    |   |-- S3GAN
    |   |-- SNGAN
    |   `-- STGAN
    |-- DiffusionForensics  # Table3
    |   |-- adm
    |   |-- ddpm
    |   |-- iddpm
    |   |-- ldm
    |   |-- pndm
    |   |-- sdv1_new
    |   |-- sdv2
    |   `-- vqdiffusion
    `-- UniversalFakeDetect # Table4
    |   |-- dalle
    |   |-- glide_100_10
    |   |-- glide_100_27
    |   |-- glide_50_27
    |   |-- guided          # Also known as ADM.
    |   |-- ldm_100
    |   |-- ldm_200
    |   `-- ldm_200_cfg
    |-- Diffusion1kStep     # Table5
        |-- DALLE
        |-- ddpm
        |-- guided-diffusion    # Also known as ADM.
        |-- improved-diffusion  # Also known as IDDPM.
        `-- midjourney

```
</details>

Further down are the '0_real' and '1_fake' folders.

## Testing

Before testing, modify the dataroot in 'test.py', and then run:
```sh
CUDA_VISIBLE_DEVICES=0 python test.py --model_path ./pretrained/model_epoch_best.pth --batch_size 64
```

