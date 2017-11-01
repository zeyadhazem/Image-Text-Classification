def findCenter (binaryImages, numSymbols):
    """
    A method that calculates the center of the symbols based on the density using K means
    unsupervised learning
    :param binaryImages: A list of 2D numpy matrices representing the binary images
    :param numSymbols: An int that specifies the number of symbols in the image (in this case 3)
    :return: A list of lists of centers of the symbols in each image
    """
    return [[]]

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
