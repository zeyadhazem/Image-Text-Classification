import cv2
import numpy as np
import imutils

def findCenters (binaryImages, numSymbols):
    """
    A method that calculates the center of the symbols based on the density using K means
    unsupervised learning
    :param binaryImages: A list of 2D numpy matrices representing the binary images
    :param numSymbols: An int that specifies the number of symbols in the image (in this case 3)
    :return: A list of lists of centers of the symbols in each image
    """
    centers = []
    for i in range (0, len(binaryImages)):
        centers.append(findSymbolsCenters(binaryImages[i], 3))

    return centers

def getSymbolImages(binaryImages, centers, width, height):
    """
    Extracts images of the symbols from the original image and using the center of the symbols,
    the desired symbol image's width and image's height
    :param binaryImages: A list of 2D numpy matrices representing the binary images
    :param centers: A list of lists of centers of the symbols in each image
    :param width: The desired width of the images containing the symbol (<= original image size)
    :param height: The desired height of the images containing the symbol (<= original image size)
    :return: A list of list of 2D numpy matrices representing the binary images indexed
    in the following way list[0] returns a list of images (of the symbols) of the original image indexed at 0
    """
    return [[[[]], [[]]]]

def findSymbolsCenters(binaryImage, n_clusters):
    """
    Transform image into 2 features X and Y and their
    :param binaryImage:
    :param n_clusters:
    :return:
    """
    centers = []
    image = np.uint8(binaryImage)

    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]

    # Truncate to the number of clusters by size of area
    cnts = sorted(cnts, key=(lambda x : cv2.contourArea(x)), reverse=True)[0:n_clusters]

    # loop over the contours
    for c in cnts:
        # compute the center of the contour
        M = cv2.moments(c)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        centers.append((cX, cY))

        # draw the contour and center of the shape on the image
        # cv2.circle(image, (cX, cY), 3, (255, 0, 0), -1)

    # cv2.imshow('gray_image',image)
    # k = cv2.waitKey(0)

    return centers
