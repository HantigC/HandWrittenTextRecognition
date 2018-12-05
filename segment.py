import cv2
import numpy as np
from preprocessing import display_histogram, dilate, project, histogram_smoothing, histogram, erode


class WordSegmentator:
    def __init__(self, img, is_object=None):
        self.img = img
        self.is_object = is_object

    def get_word_boundaries(self):
        y_project, _ = project(self.img)

        lines = self.find_lines(y_project)
        image = np.zeros((img.shape[0], img.shape[0]), dtype=np.uint8)

        for ln in lines:
            image[ln[0], ln[1]] = 255
        cv2.imshow("cred", image)
        cv2.waitKey()
        words_rect = []
        for line in lines:
            cols = self.find_words(project(self.img[line[0]:line[1], :])[1])
            for col in cols:
                words_rect.append((line[0], col[0], line[1], col[1]))
        return words_rect

    def find_lines(self, projection):
        """Should return a (start_row, end_row)"""
        raise NotImplementedError("implement this method in subclasses")

    def find_words(self, projection):
        """Should return a (start_col, end_col)"""
        raise NotImplementedError("implement this method in subclasses")


class SpaceWordSegmentator(WordSegmentator):

    def __init__(self, img):
        WordSegmentator.__init__(self, img)

    @staticmethod
    def do_job(projection):
        exited = True
        entered = False
        bounds = []
        enter_coord = 0
        for i in range(projection.shape[0]):
            if projection[i] > 0 and exited:
                exited = False
                entered = True
                enter_coord = i
            elif projection[i] == 0 and entered:
                exited = True
                entered = False
                bounds.append((enter_coord, i))
        if entered is True and exited is False:
            bounds.append((enter_coord, projection.shape[0] - 1))
        return bounds

    def find_lines(self, projection):
        hs = histogram_smoothing(projection, 9)
        return SpaceWordSegmentator.do_job(hs)

    def find_words(self, projection):
        hs = histogram_smoothing(projection, 9)
        return SpaceWordSegmentator.do_job(hs)


class LocalMinimaWordSegmentator(WordSegmentator):

    def __init__(self, img):
        WordSegmentator.__init__(self, img)

    def find_local_maximax(self, sm_histo):
        _from = 0
        while sm_histo[_from] == 0:
            _from += 1

        _to = sm_histo.shape[0] - 1
        avg = 0.0
        while sm_histo[_to] == 0:
            _to -= 1
        for i in range(_from, _to + 1):
            avg += sm_histo[i]
        avg /= float(_to - _from)
        avg *= 0.4
        avg = int(avg)
        for i in range(_from, _to):
            if sm_histo[i] - avg <= 0:
                sm_histo[i] = 0
            else:
                sm_histo[i] -= avg
        return sm_histo

    def find_lines(self, projection):
        smoothed_histog = histogram_smoothing(projection, kernel_size=13)
        _max = smoothed_histog.max(axis=0)
        im = np.zeros((_max + 1, smoothed_histog.shape[0], ), dtype=np.uint8)
        for i in range(smoothed_histog.shape[0]):
            im[_max - smoothed_histog[i], i] = 255

        # smoothed_histog = self.find_local_maximax(smoothed_histog)
        cv2.imshow("y_proj", im)

        cv2.waitKey()

        return SpaceWordSegmentator.do_job(smoothed_histog)

    def find_words(self, projection):
        smoothed_histog = histogram_smoothing(projection, kernel_size=13)
        # cv2.imshow("x_proj", display_histogram(projection))
        # cv2.waitKey()
        return SpaceWordSegmentator.do_job(smoothed_histog)


if __name__ == "__main__":
    img = cv2.imread("test2.jpeg", cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (480, 640))
    thresh = 127
    im_bw = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)[1]
    # im_bw = erode(im_bw)
    word_s = LocalMinimaWordSegmentator(im_bw)
    word_rects = word_s.get_word_boundaries()
    for w_rect in word_rects:
        cv2.rectangle(im_bw, (w_rect[1], w_rect[0]), (w_rect[3], w_rect[2]), 125, 1)

    cv2.imshow("rects", im_bw)
    cv2.waitKey()
