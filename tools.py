import matplotlib.pyplot as plt
import numpy as np
from skimage import transform  # Help us to preprocess the frames
from skimage.color import rgb2gray  # Help us to gray our frames
import cv2
from matplotlib import pyplot as plt


def preprocess_frame(frame):
    # Greyscale frame
    blur = cv2.GaussianBlur(frame, (5, 5), 0)
    frame = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    cv2.waitKey(0)
    gray = rgb2gray(frame)
    gray = gray[500:1500, 0:1000]
    # Resize & normalize
    preprocessed_frame = transform.resize(gray, [60, 60])
    return preprocessed_frame  # 60x60 frame


def state(impath):
    im = plt.imread(impath)
    processed_im = preprocess_frame(im)
    processed_im = processed_im.reshape([60, 60, 1])

    return processed_im

def get_original_frame(impath):
    im = plt.imread(impath)
    blur = cv2.GaussianBlur(im, (5, 5), 0)
    frame = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    cv2.waitKey(0)
    gray = rgb2gray(frame)
    gray = gray[500:1500, 0:1000]
    # Resize & normalize
    preprocessed_frame = transform.resize(gray, [200, 200])
    return preprocessed_frame