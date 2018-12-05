import numpy as np

neighbours = [[[-1, 0, 1, 0], [0, 1, 0, -1]], [[-1, -1, 0, 1, 1, 1, 0, -1], [0, 1, 1, 1, 0, -1, -1, -1]]]


def binarization(img, level):
    dest = np.zeros(img.shape, dtype=np.uint8)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i, j] > level:
                dest[i, j] = 255
            else:
                dest[i, j] = 0
    return dest


def grey_level_histo(img):
    histo = np.zeros((256, ), 0)
    for line in img:
        for col in line:
            histo[col] += 1
    return histo


def dilate(img, no_of_neigh=0, is_object_point=lambda x: True if x == 0 else False):
    ngh = neighbours[no_of_neigh]
    dest = np.copy(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if is_object_point(img[i, j]):
                for k in range((no_of_neigh + 1)*4):
                    if 0 <= ngh[0][k] + i < img.shape[0] and 0 <= ngh[1][k] + j < img.shape[1]:
                        dest[ngh[0][k] + i, ngh[1][k] + j] = 0
    return dest


def erode(img, no_of_neigh=0, is_object_point=lambda x: True if x == 0 else False):
    ngh = neighbours[no_of_neigh]
    dest = np.copy(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
                for k in range((no_of_neigh + 1)*4):
                    if not is_object_point(img[ngh[0][k] + i, ngh[1][k] + j]):
                        dest[i, j] = 255
                        break

    return dest


def histogram(img, bins=256):
    if bins > 256:
        bins = 256
    if bins < 0:
        bins = 0
    bin_size = 256 / bins
    _histogram = np.zeros((bins, ), dtype=np.uint32)
    for line in img:
        for col in line:
            _histogram[col / bin_size] += 1
    return _histogram


def display_histogram(_histogram, height=480):
    _max = -1
    for val in _histogram:
        if _max < val:
            _max = val
    img = np.zeros((height, _histogram.shape[0]), dtype=np.uint8)
    for j in range(img.shape[1]):
        for i in range(int((float(_histogram[j]) / float(_max)) * float(height))):
            img[height - i - 1, j] = 255
    return img


def project(img, is_object_point=lambda x: True if x == 0 else False):
    histo_v = np.zeros(img.shape[0], dtype=np.uint32)
    histo_h = np.zeros(img.shape[1], dtype=np.uint32)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if is_object_point(img[i, j]):
                histo_v[i] += 1
                histo_h[j] += 1

    return histo_v, histo_h


def histogram_smoothing(_histogram, kernel_size=5):
    kernel = [i for i in range(-(kernel_size / 2), kernel_size/2 + 1)]
    dest = np.zeros((_histogram.shape[0], ), dtype=np.uint32)
    for i in range(_histogram.shape[0]):
        nos = 0
        val = 0
        for v in kernel:
            if i + v >= 0 and i + v < _histogram.shape[0]:
                val += float(_histogram[i + v])
                nos += 1.
        dest[i] = val / nos
    return dest


def find_loacal_minimas(_histogram, kernel_size=5):
    local_mins = []
    kernel = [i for i in range(-kernel_size/2, kernel_size/2)]
    for i in range(_histogram.shape[0]):

        nos = 0
        val = 0
        for v in kernel:
            if i + v >= 0 and i + v < _histogram.shape[0]:
                val += float(_histogram[i + v])
                nos += 1.
