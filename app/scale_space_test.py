
import os
import cv2
import numpy as np


def scale_space_test1():

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    path = cur_dir + '/../data/himawari.jpg'
    img = cv2.imread(path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    k = np.sqrt(2)
    sigma = 1.6

    # for sigma in sigmas:
    for i in range(12):
        sigma_3_radius = 10 * sigma * 2

        # Applying Gaussian filter
        gauss_kernel_1d = cv2.getGaussianKernel(
            ksize=int(sigma_3_radius), sigma=sigma)
        gauss_kernel_2d = np.outer(
            gauss_kernel_1d, gauss_kernel_1d.transpose())
        log_filter = cv2.Laplacian(gauss_kernel_2d, cv2.CV_32FC1, ksize=31)

        laplacian_of_gaussian = cv2.filter2D(
            gray_img, cv2.CV_32FC1, log_filter)

        # Normalize value for visualization.
        min_val, max_val, _, _ = cv2.minMaxLoc(laplacian_of_gaussian)
        abs_max = max(abs(min_val), abs(max_val))
        tmp = (laplacian_of_gaussian * (0.5 / (2.0 * abs_max)) + 0.5) * 255.0
        laplacian_of_gaussian_uchar = np.uint8(tmp)

        title = 'Sigma = ' + str(sigma)
        cv2.namedWindow(title)
        cv2.imshow(title, laplacian_of_gaussian_uchar)
        #cv2.imwrite(title + str('.png'), laplacian_of_gaussian_uchar)
        cv2.waitKey(0)
        cv2.destroyWindow(title)

        sigma = k * sigma


if __name__ == "__main__":

    scale_space_test1()
