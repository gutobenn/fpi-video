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

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    frame = cv2.GaussianBlur(frame, (3,3), 0)
    frame = cv2.Flip(frame, flipMode=-1)

    #frame = cv2.Canny(frame, 0, 100)

    # Display the resulting frame
    cv2.imshow('FPI Video',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
