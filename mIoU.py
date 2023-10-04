import numpy as np
from typing import List, Union
from joblib import Parallel, delayed


def rle_decode(mask_rle: Union[str, int], shape=(224, 224)) -> np.array:
    '''
    mask_rle: run-length as string formatted (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background
    '''
    if mask_rle == -1:
        return np.zeros(shape)
    
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)

def iou(prediction: np.array, ground_truth: np.array, smooth=1e-7) -> float:
    intersection = np.sum(prediction * ground_truth)
    union = np.sum(prediction) + np.sum(ground_truth) - intersection
    return (intersection + smooth) / (union + smooth)


def calculate_miou(ground_truth_df, prediction_df, img_shape=(224, 224)) -> float:
    prediction_df = prediction_df[prediction_df.iloc[:, 0].isin(ground_truth_df.iloc[:, 0])]
    prediction_df.index = range(prediction_df.shape[0])

    pred_mask_rle = prediction_df.iloc[:, 1]
    gt_mask_rle = ground_truth_df.iloc[:, 1]

    def calculate_iou(pred_rle, gt_rle):
        pred_mask = rle_decode(pred_rle, img_shape)
        gt_mask = rle_decode(gt_rle, img_shape)

        if np.sum(gt_mask) > 0 or np.sum(pred_mask) > 0:
            return iou(pred_mask, gt_mask)
        else:
            return None  # No valid masks found, return None

    iou_scores = Parallel(n_jobs=-1)(
        delayed(calculate_iou)(pred_rle, gt_rle) for pred_rle, gt_rle in zip(pred_mask_rle, gt_mask_rle)
    )

    iou_scores = [score for score in iou_scores if score is not None]  # Exclude None values

    return np.mean(iou_scores)
