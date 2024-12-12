# Evaluate embeddings by using distance metrics to perform verification on the official LFW test set. The
# functions in the next block are copy pasted from `facenet.src.lfw`. Unfortunately that module has an absolute
# import from `facenet`, so can't be imported from the submodule added functionality to return false positive and
# false negatives
import os
import math
import numpy as np
from sklearn.model_selection import KFold
from scipy import interpolate
from sklearn.metrics import auc


# LFW functions taken from David Sandberg's FaceNet implementation
def distance(embeddings1, embeddings2, distance_metric=0):
    if distance_metric==0:
        # Euclidian distance
        diff = np.subtract(embeddings1, embeddings2)
        dist = np.sum(np.square(diff),1)
    elif distance_metric==1:
        # Distance based on cosine similarity
        dot = np.sum(np.multiply(embeddings1, embeddings2), axis=1)
        norm = np.linalg.norm(embeddings1, axis=1) * np.linalg.norm(embeddings2, axis=1)
        similarity = dot / norm
        dist = np.arccos(similarity) / math.pi
    else:
        raise 'Undefined distance metric %d' % distance_metric

    return dist


def calculate_roc(thresholds, embeddings1, embeddings2, actual_issame, nrof_folds=10, distance_metric=0, subtract_mean=False):
    assert(embeddings1.shape[0] == embeddings2.shape[0])
    assert(embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)

    tprs = np.zeros((nrof_folds,nrof_thresholds))
    fprs = np.zeros((nrof_folds,nrof_thresholds))
    precision = np.zeros((nrof_folds))
    recall = np.zeros((nrof_folds))
    accuracy = np.zeros((nrof_folds))

    # is_false_positive = []
    # is_false_negative = []

    indices = np.arange(nrof_pairs)

    best_thresholds = []
    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        if subtract_mean:
            mean = np.mean(np.concatenate([embeddings1[train_set], embeddings2[train_set]]), axis=0)
        else:
            mean = 0.0
        dist = distance(embeddings1-mean, embeddings2-mean, distance_metric)

        # Find the best threshold for the fold
        acc_train = np.zeros((nrof_thresholds))
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, _, _, acc_train[threshold_idx] = calculate_metrics(
                threshold=threshold, dist=dist[train_set], actual_issame=actual_issame[train_set]
            )
            # _, _, acc_train[threshold_idx], _ ,_ = calculate_accuracy(threshold, dist[train_set], actual_issame[train_set])
        best_threshold_index = np.argmax(acc_train)
        best_thresholds.append(thresholds[best_threshold_index])
        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx,threshold_idx], fprs[fold_idx,threshold_idx], _, _, _ = calculate_metrics(
                threshold=threshold, dist=dist[test_set], actual_issame=actual_issame[test_set]
            )
            # tprs[fold_idx,threshold_idx], fprs[fold_idx,threshold_idx], _, _, _ = calculate_accuracy(threshold, dist[test_set], actual_issame[test_set])
        _, _, precision[fold_idx], recall[fold_idx], accuracy[fold_idx] = calculate_metrics(
            threshold=thresholds[best_threshold_index], dist=dist[test_set], actual_issame=actual_issame[test_set]
        )
        # _, _, accuracy[fold_idx], is_fp, is_fn = calculate_accuracy(thresholds[best_threshold_index], dist[test_set], actual_issame[test_set])

        # is_false_positive.extend(is_fp)
        # is_false_negative.extend(is_fn)
    tpr = np.mean(tprs, 0)
    fpr = np.mean(fprs, 0)

    return tpr, fpr, accuracy, precision, recall, np.array(best_thresholds)


def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    is_fp = np.logical_and(predict_issame, np.logical_not(actual_issame))
    is_fn = np.logical_and(np.logical_not(predict_issame), actual_issame)

    tpr = 0 if (tp+fn==0) else float(tp) / float(tp+fn)
    fpr = 0 if (fp+tn==0) else float(fp) / float(fp+tn)
    acc = float(tp+tn)/dist.size
    return tpr, fpr, acc, is_fp, is_fn


def calculate_metrics(threshold, dist, actual_issame):
    # If distance is less than threshold, then prediction is set to True
    predict_issame = np.less(dist, threshold)

    true_positives = np.sum(np.logical_and(predict_issame, actual_issame))
    false_positives = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    true_negatives = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    false_negatives = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    # For dealing with Divide By Zero exception
    true_positive_rate = 0 if (true_positives + false_negatives == 0) else \
        float(true_positives) / float(true_positives + false_negatives)

    false_positive_rate = 0 if (false_positives + true_negatives == 0) else \
        float(false_positives) / float(false_positives + true_negatives)

    precision = 0 if (true_positives + false_positives) == 0 else\
        float(true_positives) / float(true_positives + false_positives)

    recall = 0 if (true_positives + false_negatives) == 0 else \
        float(true_positives) / float(true_positives + false_negatives)

    accuracy = float(true_positives + true_negatives) / dist.size

    return true_positive_rate, false_positive_rate, precision, recall, accuracy


def calculate_val(thresholds, embeddings1, embeddings2, actual_issame, far_target, nrof_folds=10, distance_metric=0, subtract_mean=False):
    assert(embeddings1.shape[0] == embeddings2.shape[0])
    assert(embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)

    val = np.zeros(nrof_folds)
    far = np.zeros(nrof_folds)

    indices = np.arange(nrof_pairs)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        if subtract_mean:
            mean = np.mean(np.concatenate([embeddings1[train_set], embeddings2[train_set]]), axis=0)
        else:
          mean = 0.0
        dist = distance(embeddings1-mean, embeddings2-mean, distance_metric)

        # Find the threshold that gives FAR = far_target
        far_train = np.zeros(nrof_thresholds)
        for threshold_idx, threshold in enumerate(thresholds):
            _, far_train[threshold_idx] = calculate_val_far(threshold, dist[train_set], actual_issame[train_set])
        if np.max(far_train)>=far_target:
            f = interpolate.interp1d(far_train, thresholds, kind='slinear')
            threshold = f(far_target)
        else:
            threshold = 0.0

        val[fold_idx], far[fold_idx] = calculate_val_far(threshold, dist[test_set], actual_issame[test_set])

    val_mean = np.mean(val)
    far_mean = np.mean(far)
    val_std = np.std(val)
    return val_mean, val_std, far_mean


def calculate_val_far(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    true_accept = np.sum(np.logical_and(predict_issame, actual_issame))
    false_accept = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    n_same = np.sum(actual_issame)
    n_diff = np.sum(np.logical_not(actual_issame))
    val = float(true_accept) / float(n_same)
    far = float(false_accept) / float(n_diff)
    return val, far


def evaluate(embeddings, actual_issame, nrof_folds=10, distance_metric=0, subtract_mean=False):
    # Calculate evaluation metrics
    # thresholds = np.arange(0, 4, 0.01)
    thresholds = np.arange(0, 1, 0.001)
    # List of embeddings of the 1st element in each pair
    embeddings1 = embeddings[0::2]
    # List of embeddings of the 2nd element in each pair
    embeddings2 = embeddings[1::2]
    tpr, fpr, accuracy, precision, recall, best_thresholds  = calculate_roc(thresholds, embeddings1, embeddings2,
        np.asarray(actual_issame), nrof_folds=nrof_folds, distance_metric=distance_metric, subtract_mean=subtract_mean)

    roc_auc = auc(fpr, tpr)
    eer = fpr[np.nanargmin(np.absolute((1 - tpr - fpr)))]

    # thresholds = np.arange(0, 4, 0.001)
    thresholds = np.arange(0, 1, 0.001)
    val_0_001, val_std_0_001, far_0_001 = calculate_val(thresholds, embeddings1, embeddings2,
                                                        np.asarray(actual_issame), 1e-3, nrof_folds=nrof_folds,
                                                        distance_metric=distance_metric, subtract_mean=subtract_mean)
    val_0_01, val_std_0_01, far_0_01 = calculate_val(thresholds, embeddings1, embeddings2,
                                                     np.asarray(actual_issame), 1e-2, nrof_folds=nrof_folds,
                                                     distance_metric=distance_metric, subtract_mean=subtract_mean)
    val_0_1, val_std_0_1, far_0_1 = calculate_val(thresholds, embeddings1, embeddings2,
                                                  np.asarray(actual_issame), 1e-1, nrof_folds=nrof_folds,
                                                  distance_metric=distance_metric, subtract_mean=subtract_mean)
    val = (val_0_001, val_0_01, val_0_1)
    val_std = (val_std_0_001, val_std_0_01, val_std_0_1)
    far = (far_0_001, far_0_01, far_0_1)
    return tpr, fpr, roc_auc, eer, accuracy, precision, recall, val, val_std, far, best_thresholds


def add_extension(path):
    if os.path.exists(path+'.jpg'):
        return path+'.jpg'
    elif os.path.exists(path+'.png'):
        return path+'.png'
    else:
        raise RuntimeError('No file "%s" with extension png or jpg.' % path)


def get_paths(dataset_dir, pairs, dataset='lfw'):
    nrof_skipped_pairs = 0
    path_list = []
    issame_list = []
    for pair in pairs:
        if len(pair) == 3:
            if dataset == 'lfw':
                path0 = add_extension(os.path.join(dataset_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1])))
                path1 = add_extension(os.path.join(dataset_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[2])))
            else:
                path0 = os.path.join(dataset_dir, pair[0], pair[1])
                path1 = os.path.join(dataset_dir, pair[0], pair[2])
            issame = True
        elif len(pair) == 4:
            if dataset == 'lfw':
                path0 = add_extension(os.path.join(dataset_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1])))
                path1 = add_extension(os.path.join(dataset_dir, pair[2], pair[2] + '_' + '%04d' % int(pair[3])))
            else:
                path0 = os.path.join(dataset_dir, pair[0], pair[1])
                path1 = os.path.join(dataset_dir, pair[2], pair[3])
            issame = False
        if os.path.exists(path0) and os.path.exists(path1):    # Only add the pair if both paths exist
            path_list += (path0,path1)
            issame_list.append(issame)
        else:
            nrof_skipped_pairs += 1
    if nrof_skipped_pairs>0:
        print('Skipped %d image pairs' % nrof_skipped_pairs)
    return path_list, issame_list


def read_pairs(pairs_filename):
    pairs = []
    with open(pairs_filename, 'r') as f:
        for line in f.readlines()[1:]:
            pair = line.strip().split()
            pairs.append(pair)
    return np.array(pairs, dtype=object)