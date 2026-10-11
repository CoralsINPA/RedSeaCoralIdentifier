"""Verbatim copies of the legacy notebook helpers, used only as references in the tests.

Sources: `get_crop` / `get_masked_crop` / `decode_rle` from WC cell 14 and 8, the masked
square crop from CM cell 9, and the filter functions from WC cell 8.
"""

import cv2
import numpy as np
import pycocotools.mask as mask_util
from pycocotools import mask as maskUtils


def decode_rle(rle):
    if isinstance(rle["counts"], str):
        rle = {"size": rle["size"], "counts": rle["counts"].encode("utf-8")}
    return mask_util.decode(rle).astype(bool)


def bbox_intersection(b1, b2):
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    ix = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
    iy = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
    return ix * iy


def aspect_ratio(bbox):
    _, _, w, h = bbox
    return max(w, h) / max(min(w, h), 1)


def filter_annotations(annotations, min_area, containment_thresh, max_aspect_ratio):
    candidates = [a for a in annotations if a["area"] >= min_area]
    candidates = [a for a in candidates if aspect_ratio(a["bbox"]) <= max_aspect_ratio]
    if len(candidates) <= 1:
        return candidates
    to_remove = set()
    for i in range(len(candidates)):
        for j in range(len(candidates)):
            if i == j or candidates[j]["id"] in to_remove:
                continue
            a_small, a_large = candidates[i], candidates[j]
            if a_large["area"] <= a_small["area"]:
                continue
            if bbox_intersection(a_small["bbox"], a_large["bbox"]) == 0:
                continue
            mask_small = decode_rle(a_small["segmentation"])
            mask_large = decode_rle(a_large["segmentation"])
            containment = np.logical_and(mask_small, mask_large).sum() / a_small["area"]
            if containment >= containment_thresh:
                to_remove.add(a_large["id"])
    return [a for a in candidates if a["id"] not in to_remove]


def get_crop(img, bbox, padding_pct=0.15):
    x, y, w, h = map(int, bbox)
    pad_w = int(w * padding_pct)
    pad_h = int(h * padding_pct)
    x1 = max(0, x - pad_w)
    y1 = max(0, y - pad_h)
    x2 = min(img.shape[1], x + w + pad_w)
    y2 = min(img.shape[0], y + h + pad_h)
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return crop


def get_masked_crop(img, bbox, rle, padding_pct=0.05, background="black", min_visible=500):
    mask = decode_rle(rle)
    if mask.shape != img.shape[:2]:
        mask = cv2.resize(
            mask.astype(np.uint8), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST
        ).astype(bool)
    bg_val = {"black": 0, "gray": 128, "white": 255}.get(background, 0)
    masked_img = img.copy()
    masked_img[~mask] = bg_val
    x, y, w, h = map(int, bbox)
    pad_w = int(w * padding_pct)
    pad_h = int(h * padding_pct)
    x1 = max(0, x - pad_w)
    y1 = max(0, y - pad_h)
    x2 = min(img.shape[1], x + w + pad_w)
    y2 = min(img.shape[0], y + h + pad_h)
    crop = masked_img[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    visible = np.any(crop != bg_val, axis=2).sum()
    if visible < min_visible:
        return None
    return crop


def is_sharp_enough(crop, threshold=50.0):
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() >= threshold


def get_clean_masked_square_crop(img, anno, target_h, target_w):
    seg = anno["segmentation"]
    if isinstance(seg, list):
        rles = maskUtils.frPyObjects(seg, target_h, target_w)
        rle = maskUtils.merge(rles)
        mask = maskUtils.decode(rle)
    elif isinstance(seg, dict):
        if isinstance(seg["counts"], list):
            rle = maskUtils.frPyObjects(seg, target_h, target_w)
            mask = maskUtils.decode(rle)
        else:
            mask = maskUtils.decode(seg)

    if len(mask.shape) == 3:
        mask = mask[:, :, 0]

    mask = mask.astype(np.uint8) * 255
    masked_img = cv2.bitwise_and(img, img, mask=mask)

    x, y, w, h = map(int, anno["bbox"])

    x = max(0, x)
    y = max(0, y)
    w = min(w, target_w - x)
    h = min(h, target_h - y)

    crop = masked_img[y : y + h, x : x + w]

    if crop.size == 0:
        return None

    h_c, w_c = crop.shape[:2]
    side = max(h_c, w_c)
    pad_y = (side - h_c) // 2
    pad_x = (side - w_c) // 2

    square_crop = cv2.copyMakeBorder(
        crop,
        pad_y,
        side - h_c - pad_y,
        pad_x,
        side - w_c - pad_x,
        cv2.BORDER_CONSTANT,
        value=[0, 0, 0],
    )
    return square_crop
