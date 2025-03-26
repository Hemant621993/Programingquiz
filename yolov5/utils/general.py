import torch

def non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False, labels=()):
    """Mock implementation of non_max_suppression"""
    # Return an empty detection for each image in the batch
    return [torch.zeros((0, 6)) for _ in range(pred[0].shape[0])]

def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None):
    """Mock implementation of scale_boxes"""
    return boxes