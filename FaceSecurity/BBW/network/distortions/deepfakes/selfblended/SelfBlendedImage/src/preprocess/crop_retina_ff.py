from glob import glob
import os
import pandas as pd
import cv2
from tqdm import tqdm
import numpy as np
import shutil
import json
import sys
import argparse
from imutils import face_utils
from retinaface.pre_trained_models import get_model
from retinaface.utils import vis_annotations
import torch


def facecrop(model, org_path, save_path, period=1, num_frames=10):
    cap_org = cv2.VideoCapture(org_path)
    croppedfaces = []
    frame_count_org = int(cap_org.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_idxs = np.linspace(0, frame_count_org - 1, num_frames, endpoint=True, dtype=np.int)

    for cnt_frame in range(frame_count_org):
        ret_org, frame_org = cap_org.read()
        height, width = frame_org.shape[:-1]
        if not ret_org:
            tqdm.write('Frame read {} Error! : {}'.format(cnt_frame, os.path.basename(org_path)))
            continue

        if cnt_frame not in frame_idxs:
            continue

        frame = cv2.cvtColor(frame_org, cv2.COLOR_BGR2RGB)
        faces = model.predict_jsons(frame)
        try:
            if len(faces) == 0:
                print(faces)
                tqdm.write('No faces in {}:{}'.format(cnt_frame, os.path.basename(org_path)))
                continue
            face_s_max = -1
            landmarks = []
            size_list = []
            for face_idx in range(len(faces)):
                x0, y0, x1, y1 = faces[face_idx]['bbox']
                landmark = np.array([[x0, y0], [x1, y1]] + faces[face_idx]['landmarks'])
                face_s = (x1 - x0) * (y1 - y0)
                size_list.append(face_s)
                landmarks.append(landmark)
        except Exception as e:
            print(f'error in {cnt_frame}:{org_path}')
            print(e)
            continue
        landmarks = np.concatenate(landmarks).reshape((len(size_list),) + landmark.shape)
        landmarks = landmarks[np.argsort(np.array(size_list))[::-1]]

        save_path_ = save_path + 'frames/' + os.path.basename(org_path).replace('.mp4', '/')
        os.makedirs(save_path_, exist_ok=True)
        image_path = save_path_ + str(cnt_frame).zfill(3) + '.png'
        land_path = save_path_ + str(cnt_frame).zfill(3)

        land_path = land_path.replace('/frames', '/retina')
        os.makedirs(os.path.dirname(land_path), exist_ok=True)
        np.save(land_path, landmarks)

        if not os.path.isfile(image_path):
            cv2.imwrite(image_path, frame_org)

    cap_org.release()
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', dest='dataset',
                        choices=['DeepFakeDetection_original', 'DeepFakeDetection', 'FaceShifter', 'Face2Face',
                                 'Deepfakes', 'FaceSwap', 'NeuralTextures', 'Original', 'Celeb-real', 'Celeb-synthesis',
                                 'YouTube-real', 'DFDC', 'DFDCP'])
    parser.add_argument('-c', dest='comp', choices=['raw', 'c23', 'c40'], default='raw')
    parser.add_argument('-n', dest='num_frames', type=int, default=32)
    args = parser.parse_args()
    if args.dataset == 'Original':
        dataset_path = 'data/FaceForensics++/original_sequences/youtube/{}/'.format(args.comp)
    elif args.dataset == 'DeepFakeDetection_original':
        dataset_path = 'data/FaceForensics++/original_sequences/actors/{}/'.format(args.comp)
    elif args.dataset in ['DeepFakeDetection', 'FaceShifter', 'Face2Face', 'Deepfakes', 'FaceSwap', 'NeuralTextures']:
        dataset_path = 'data/FaceForensics++/manipulated_sequences/{}/{}/'.format(args.dataset, args.comp)
    elif args.dataset in ['Celeb-real', 'Celeb-synthesis', 'YouTube-real']:
        dataset_path = 'data/Celeb-DF-v2/{}/'.format(args.dataset)
    elif args.dataset in ['DFDC', 'DFDCVal']:
        dataset_path = 'data/{}/'.format(args.dataset)
    else:
        raise NotImplementedError

    device = torch.device('cuda')

    model = get_model("resnet50_2020-07-20", max_size=2048, device=device)
    model.eval()

    movies_path = dataset_path + 'videos/'

    movies_path_list = sorted(glob(movies_path + '*.mp4'))
    print("{} : videos are exist in {}".format(len(movies_path_list), args.dataset))

    n_sample = len(movies_path_list)

    for i in tqdm(range(n_sample)):
        folder_path = movies_path_list[i].replace('videos/', 'frames/').replace('.mp4', '/')
        if len(glob(folder_path.replace('/frames/', '/retina/') + '*.npy')) < args.num_frames:
            facecrop(model, movies_path_list[i], save_path=dataset_path, num_frames=args.num_frames)
