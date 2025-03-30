# Source Code for PRO-Face-D

PRO-Face D: Diffusion Model Based Privacy Protection Method of Face Image.

## Prepraration

### Environment

The project's runtime environment is based on Miniconda. You can use the following command to install the project's runtime environment：
```bash
conda create -n PROFaceD python=3.8
pip install -r requirements.txt
```

### Pre-trained model
For identity extraction model ([ArcFace](https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch)), you can download the pre-trained model from [BaiduDisk link](https://pan.baidu.com/s/1CL-l4zWqsI1oDuEEYVhj-g#list/path=%2F) (Password: e8pw) 

- create the directory `./functions/arcface_torch/checkpoints/`
- push the pre-trained model to the directory.

For Diffusion model, we use the pre-trained model from [DiffAE](https://github.com/phizaz/diffae?tab=readme-ov-file).

- create the directory `./checkpoints/`
- push the pre-trained model to the directory.

### Data

We have provided several images as examples in the `datas` folder.

## Testing

Anonymous operation for face images.

You need to set the input path `data_path` and the output path `output_path`, than run `test_anony.py` to test the model:

```bash
conda activate PROFaceD
python test_anony.py
```

