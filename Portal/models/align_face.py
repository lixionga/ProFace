import numpy as np
import cv2


def align_with_five_points(src_points, size=224,expand_ratio=1.2):

    REFERENCE_FACIAL_POINTS = [
        [30.29459953,  51.69630051],
        [65.53179932,  51.50139999],
        [48.02519989,  71.73660278],
        [33.54930115,  92.3655014],
        [62.72990036,  92.20410156]
    ]
    REFERENCE_FACIAL_POINTS = np.array(REFERENCE_FACIAL_POINTS)
    REFERENCE_FACIAL_POINTS[:, 0] += 8
    REFERENCE_FACIAL_POINTS *= (size * expand_ratio) / 112.0


    dst_points = REFERENCE_FACIAL_POINTS
    # align dst to src
    # print("Shape of src_points before calling np.matrix: ", src_points.shape)
    src_pts = np.matrix(src_points.astype(np.float64))
    dst_pts = np.matrix(dst_points.astype(np.float64))

    tfm = np.float32([[1, 0, 0], [0, 1, 0]])
    n_pts = src_pts.shape[0]
    ones = np.ones((n_pts, 1), src_pts.dtype)
    src_pts_ = np.hstack([src_pts, ones])
    dst_pts_ = np.hstack([dst_pts, ones])

    A, res, rank, s = np.linalg.lstsq(src_pts_, dst_pts_, rcond=None)

    if rank == 3:
        tfm = np.float32([
            [A[0, 0], A[1, 0], A[2, 0]],
            [A[0, 1], A[1, 1], A[2, 1]]
        ])
    elif rank == 2:
        tfm = np.float32([
            [A[0, 0], A[1, 0], 0],
            [A[0, 1], A[1, 1], 0]
        ])
    return tfm


def back_matrix(affine_matrix):
    back_matrix = np.zeros((3, 3))
    back_matrix[0:2, :] = affine_matrix
    back_matrix[2, 2] = 1
    back_matrix = np.linalg.pinv(back_matrix)
    back_matrix = back_matrix[0:2, :]
    return back_matrix


def align_img(img, src_lmks, size=224):
    if src_lmks is None:
        return None,None
    else:
        M = align_with_five_points(src_lmks, size)
        # aligned_img = cv2.warpAffine(img, M, (size, size), flags=cv2.INTER_LINEAR)
        aligned_img = cv2.warpAffine(img, M, (size, size), flags=cv2.INTER_CUBIC)

        
        return aligned_img,back_matrix(M[:2])


def align_imgs(img, src_lmks_list, size=224,expand_ratio=1):
    aligned_imgs = []
    back_matrices = []
    if src_lmks_list is None:
        return None,None
    else:
        for src_lmks in src_lmks_list:  # 对每张人脸进行处理
            M = align_with_five_points(src_lmks, size,expand_ratio)
            # out_w = int(size * expand_ratio)
            # out_h = int(size * expand_ratio)
            aligned_img = cv2.warpAffine(img, M, (size, size), flags=cv2.INTER_LINEAR)
            aligned_imgs.append(aligned_img)
            back_matrices.append(back_matrix(M[:2]))
        return aligned_imgs, back_matrices


def dealign(generated, origin, back_affine_matrix,  mask):
    kernel = cv2.getStructuringElement(shape=cv2.MORPH_RECT,ksize=(11,11))

    mask[mask > 0.001] = 1.0
    mask = cv2.dilate(mask, kernel)
    mask = cv2.erode(mask,kernel,iterations=2)

    mask = cv2.blur(mask,(7,7))

    mask_1 = np.zeros_like(mask, dtype=np.float32)
    mask_1[10:-10, 10:-10] = 1.0
    mask_1 = cv2.blur(mask_1, (11, 11))
    mask = mask * mask_1

    target_image = cv2.warpAffine(generated, back_affine_matrix, (origin.shape[1], origin.shape[0]))
    mask = cv2.warpAffine(mask, back_affine_matrix, (origin.shape[1], origin.shape[0]))

    mask = mask[..., np.newaxis]

    dealigned_img = target_image * mask + origin * (1 - mask)

    dealigned_img = dealigned_img.clip(0, 255.0).astype(np.uint8)

    return dealigned_img
