import os
import shutil
from eval_utils import read_pairs, get_paths


def crop_files(path_list, data_dir_test):
    for f in path_list:
        img_id = f.split('/')[-2]
        img_id_dir = os.path.join(data_dir_test, img_id)
        if not os.path.isdir(img_id_dir):
            os.makedirs(img_id_dir, exist_ok=True)
        shutil.copy(f, img_id_dir)


dataset_name = 'vggface2'
data_dir = '/home/yuanlin/Datasets/VGG-Face2/data/test_align_crop_224_frontal'
data_dir_test = '/home/yuanlin/Datasets/VGG-Face2/data/test_6000pairs_2'
pairs_path = 'pairs_vggface2.txt'
pairs = read_pairs(pairs_path)
path_list, issame_list = get_paths(data_dir, pairs, dataset_name)
if not os.path.isdir(data_dir_test):
    os.makedirs(data_dir_test, exist_ok=True)
crop_files(path_list, data_dir_test)


print("Done")
