import cv2
import numpy as np
import math


def image_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def image_bin(image_gs):
    ret, image_bin = cv2.threshold(image_gs, 100, 255, cv2.THRESH_BINARY)
    return image_bin


def dilate(image):
    kernel = np.ones((3, 3)) # strukturni element 3x3 blok
    return cv2.dilate(image, kernel, iterations=1)


def erode(image):
    kernel = np.ones((3, 3)) # strukturni element 3x3 blok
    return cv2.erode(image, kernel, iterations=1)


def invert(image):
    return 255-image


def resize_region(region):
    return cv2.resize(region, (28, 28), interpolation=cv2.INTER_NEAREST)


def image_bin(image_gs):
    return cv2.threshold(image_gs, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]


def scale_to_range_n_flaten(image):
    return (image/255).flatten()


def resize_region(region):
    '''Transformisati selektovani region na sliku dimenzija 28x28'''
    return cv2.resize(region, (28, 28), interpolation=cv2.INTER_NEAREST)


def distance2points(p1, p2):
    return math.sqrt(((p1[1]-p2[1])**2)+((p1[0]-p2[0])**2))