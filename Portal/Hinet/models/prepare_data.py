import os
import cv2
import numpy as np
import glob
import os.path as osp
from insightface.model_zoo import model_zoo
from skimage import transform as trans
import warnings
import onnxruntime as ort

# 设置日志级别为 ERROR
ort.set_default_logger_severity(3)
warnings.filterwarnings('ignore', message='.*Expected shape from model.*')


src1 = np.array([[51.642, 50.115], [57.617, 49.990], [35.740, 69.007],
                 [51.157, 89.050], [57.025, 89.702]],
                dtype=np.float32)
#<--left
src2 = np.array([[45.031, 50.118], [65.568, 50.872], [39.677, 68.111],
                 [45.177, 86.190], [64.246, 86.758]],
                dtype=np.float32)

#---frontal
src3 = np.array([[39.730, 51.138], [72.270, 51.138], [56.000, 68.493],
                 [42.463, 87.010], [69.537, 87.010]],
                dtype=np.float32)

#-->right
src4 = np.array([[46.845, 50.872], [67.382, 50.118], [72.737, 68.111],
                 [48.167, 86.758], [67.236, 86.190]],
                dtype=np.float32)

#-->right profile
src5 = np.array([[54.796, 49.990], [60.771, 50.115], [76.673, 69.007],
                 [55.388, 89.702], [61.257, 89.050]],
                dtype=np.float32)

src = np.array([src1, src2, src3, src4, src5])
src_map = src

ffhq_src = np.array([[192.98138, 239.94708], [318.90277, 240.1936], [256.63416, 314.01935],
                                           [201.26117, 371.41043], [313.08905, 371.15118]])
ffhq_src = np.expand_dims(ffhq_src, axis=0)

# arcface_src = np.array(
#     [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
#      [41.5493, 92.3655], [70.7299, 92.2041]],
#     dtype=np.float32)

# arcface_src = np.expand_dims(arcface_src, axis=0)

# In[66]:


# lmk is prediction; src is template
def estimate_norm(lmk, image_size=112, mode='ffhq'):
    assert lmk.shape == (5, 2)
    tform = trans.SimilarityTransform()
    lmk_tran = np.insert(lmk, 2, values=np.ones(5), axis=1)
    min_M = []
    min_index = []
    min_error = float('inf')
    if mode == 'ffhq':
        # assert image_size == 112
        src = ffhq_src * image_size / 512
    else:
        src = src_map * image_size / 112
    for i in np.arange(src.shape[0]):
        tform.estimate(lmk, src[i])
        M = tform.params[0:2, :]
        results = np.dot(M, lmk_tran.T)
        results = results.T
        error = np.sum(np.sqrt(np.sum((results - src[i])**2, axis=1)))
        #         print(error)
        if error < min_error:
            min_error = error
            min_M = M
            min_index = i
    return min_M, min_index






class LandmarkModel():
    def __init__(self, name, root='/Users/mac/代码/test/Hinet/checkpoints'):
        self.models = {}
        root = os.path.expanduser(root)
        onnx_files = glob.glob(osp.join(root, name, '*.onnx'))
        onnx_files = sorted(onnx_files)
        for onnx_file in onnx_files:
            if onnx_file.find('_selfgen_')>0:
                continue
            model = model_zoo.get_model(onnx_file)
            if model.taskname not in self.models:
                print('find model:', onnx_file, model.taskname)
                self.models[model.taskname] = model
            else:
                print('duplicated model task type, ignore:', onnx_file, model.taskname)
                del model
        assert 'detection' in self.models
        self.det_model = self.models['detection']


    def prepare(self, ctx_id, det_thresh=0.5, det_size=(640, 640), mode ='None'):
        self.det_thresh = det_thresh
        self.mode = mode
        assert det_size is not None
        print('set det-size:', det_size)
        self.det_size = det_size
        for taskname, model in self.models.items():
            if taskname=='detection':
                model.prepare(ctx_id, input_size=det_size)
            else:
                model.prepare(ctx_id)


    def get(self, img, max_num=0):
        bboxes, kpss = self.det_model.detect(img, threshold=self.det_thresh, max_num=max_num, metric='default')
        if bboxes.shape[0] == 0:
            return None
        det_score = bboxes[..., 4]

        # select the face with the hightest detection score
        best_index = np.argmax(det_score)

        kps = None
        if kpss is not None:
            kps = kpss[best_index]
        return kps,bboxes
    
    
    def get_emb(self,img,max_num=0):
        bboxes, kpss = self.det_model.detect(img, threshold=self.det_thresh, max_num=max_num, metric='default')
        if bboxes.shape[0] == 0:
            return None
        det_score = bboxes[..., 4]
        best_index = np.argmax(det_score)
        rec_model = self.models['recognition']
        face = img[bboxes[best_index][1]:bboxes[best_index][3], bboxes[best_index][0]:bboxes[best_index][2]]
        face_emb = rec_model.get(face)
        return  face_emb


    def gets(self, img, max_num=0):
        bboxes, kpss = self.det_model.detect(img, threshold=self.det_thresh, max_num=max_num, metric='default')
        return kpss,bboxes

    # def gets(self, img, max_num=0):
    #     bboxes, kpss = self.det_model.detect(img, threshold=self.det_thresh, max_num=max_num, metric='default')
    #     return kpss