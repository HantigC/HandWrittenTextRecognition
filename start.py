import cv2
from preprocessing import binarization, dilate, erode, histogram, display_histogram, project, histogram_smoothing

if __name__ == "__main__":
    img = cv2.imread("text34.jpeg", cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (640, 480))
    histo = histogram(img)
    dest = display_histogram(histo)
    bin_img = binarization(img, 128)
    histo_v, histo_h = project(bin_img)
    dest_v = display_histogram(histo_v)
    dest_h = display_histogram(histo_h)

    dest_v_s = histogram_smoothing(histo_v, kernel_size=11)
    dest_h_s = histogram_smoothing(histo_h, kernel_size=11)

    dest_h_s = display_histogram(dest_h_s)
    dest_v_s = display_histogram(dest_v_s)

    cv2.imshow("img", img)
    cv2.imshow("bin-img", bin_img)
    cv2.imshow("histo", dest)
    cv2.imshow("histo_v", dest_v)
    cv2.imshow("histo_h", dest_h)

    cv2.imshow("histo_v_s", dest_v_s)
    cv2.imshow("histo_h_s", dest_h_s)
    cv2.waitKey()
