import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def preprocess(path: str, enhance: bool = True) -> np.ndarray:
    img = _load(path)
    img = _upscale_if_small(img)
    img = _deskew(img)
    img = _denoise(img)
    if enhance:
        img = _enhance_contrast(img)
        img = _sharpen(img)
    return img


def _load(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {path}")
    return img


def _upscale_if_small(img: np.ndarray, min_height: int = 1000) -> np.ndarray:
    h, w = img.shape[:2]
    if h < min_height:
        scale = min_height / h
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
        logger.debug("Upscaled %dx%d → %dx%d", h, w, *img.shape[:2][::-1])
    return img


def _deskew(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100,
                            minLineLength=100, maxLineGap=10)
    if lines is None:
        return img

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 != x1:
            a = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if -45 < a < 45:
                angles.append(a)

    if not angles:
        return img

    angle = float(np.median(angles))
    if abs(angle) < 0.5:
        return img

    logger.debug("Deskewing by %.2f°", angle)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h),
                          flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


def _denoise(img: np.ndarray) -> np.ndarray:
    return cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)


def _enhance_contrast(img: np.ndarray, clip_limit: float = 2.5) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    return cv2.cvtColor(cv2.merge([clahe.apply(l), a, b]), cv2.COLOR_LAB2BGR)


def _sharpen(img: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(img, (0, 0), 3)
    return cv2.addWeighted(img, 1.5, blurred, -0.5, 0)
