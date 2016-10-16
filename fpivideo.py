#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import cv2

"""
Basic real time video processing
"""
__author__ = "Augusto Bennemann"
__license__ = "GPL"
__version__ = "1.0"
__email__ = "abennemann@inf.ufrgs.br"

def apply_none(frame):
    return frame

def apply_gaussian(frame):
    ksize = cv2.getTrackbarPos('Gaussian Kernel Size','FPI Video')
    if(ksize%2 == 0):
        ksize -= 1
    # TODO aceitar só valores impares e a partir de 3. Por enquanto ele ta convertendo os pares para o impar antecessor
    return cv2.GaussianBlur(frame, (ksize, ksize), 0)

def apply_grayscale(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

def apply_canny(frame):
    # TODO ajusar parametros
    frame = cv2.GaussianBlur(frame, (3, 3), 0)
    return cv2.Canny(frame, 25, 200)

def apply_sobel(frame):
    # TODO é o sobel x ou xy?
    return cv2.Sobel(frame, cv2.CV_64F, 1, 0, ksize=5) # Sobel X

def apply_laplacian(frame):
    return cv2.Laplacian(frame, cv2.CV_64F)

def apply_negative(frame):
    return cv2.addWeighted(frame, -1, np.zeros(frame.shape, frame.dtype), 0, 255)

def nothing(x):
    pass

def resize_video(frame):
    return cv2.resize(frame, None, fx=1.0/2.0, fy=1.0/2.0, interpolation = cv2.INTER_CUBIC)

def mirror_video(frame):
    # Mirroring. 1 = horizontal, 0 = vertical, -1 = both
    return cv2.flip(frame, mirroring_mode)

def rotate_video(frame):
    (h, w) = frame.shape[:2]
    center = (w/2, h/2)
    M = cv2.getRotationMatrix2D(center, rotation_mode * 90, 1.0)
    return cv2.warpAffine(frame, M, (w,h))

cv2.namedWindow('FPI Video')
cap = cv2.VideoCapture(0)
apply_effects = apply_none # points to function to be applied
apply_transforms = apply_none  # points to function to be applied
mirroring_mode = -1
rotation_mode = 0

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    frame = apply_transforms(frame)
    frame = apply_effects(frame)

    # Display the resulting frame
    cv2.imshow('FPI Video',frame)

    c = cv2.waitKey(1) & 0xFF
    if c == ord('q'):
        break
    elif c == ord('g'):
        apply_effects = apply_gaussian
        cv2.createTrackbar('Gaussian Kernel Size','FPI Video', 3, 15, nothing)
    elif c == ord('c'):
        apply_effects = apply_canny
    elif c == ord('s'):
        apply_effects = apply_sobel
    elif c == ord('x'):
        apply_effects = apply_grayscale
    elif c == ord('n'):
        apply_effects = apply_negative
    elif c == ord('l'):
        apply_effects = apply_laplacian
    elif c == ord('r'):
        apply_transforms = resize_video
    elif c == ord('m'):
        mirroring_mode = (mirroring_mode%3) - 1
        apply_transforms = mirror_video
    elif c == ord('z'):
        rotation_mode = (rotation_mode + 1) % 4
        apply_transforms = rotate_video

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
