# Source Code for Inspector

Source code for Inspector: Inspector for Face Forgery Detection: Defending Against Adversarial Attacks From Coarse to Fine (IEEE TIP 2024: https://doi.org/10.1109/TIP.2024.3434388)

# Prepraration

### Dependencies

The project's runtime environment is based on Miniconda. You can use the following command to install the project's runtime environment：

``conda create --name Inspector --file requirements.txt``


### Datasets

Our training is conducted on the FaceForensics++ dataset, which can be downloaded from the repository [FaceForensics](https://github.com/ondyari/FaceForensics) and place it in the ``data/``.

# Training

Run the training scripts in the ``train`` folder in the following order: ``train_dec.sh``, ``train_rec.sh``, ``train_aut.sh``, ``train_cor.sh`` .If you wish to modify the relevant configurations, please make the changes within the corresponding script files.
