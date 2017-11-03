def binarize (images):
    processedImages = meanBinary(images)
    return processedImages


def binarizeUsingThreshold(images, threshold):
    processedImages = []

    for i in range (0,len(images)):
        flt = images[i].copy()
        flt[flt < threshold] = 0

        processedImages.append(flt)

    return processedImages

def meanBinary(images):
    processedImages = []

    for i in range (0,len(images)):
        flt = images[i].copy()

        # 3 passes for filtering the data by getting the mean of nonzero elements
        for j in range (0,3):
            mean = flt[flt.nonzero()].mean()
            flt[flt < mean] = 0

        processedImages.append(flt)

    return processedImages
