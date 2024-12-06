import numpy as np
import cv2
import os
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.misc
from matplotlib.backends.backend_pdf import PdfPages


class FindLaneLinePixels:
    def __init__(self, image, image_name):
        self.image = image
        self.image_name = image_name
        self.lab = cv2.cvtColor(self.image, cv2.COLOR_BGR2LAB)
        self.hls = cv2.cvtColor(self.image, cv2.COLOR_RGB2HLS).astype(np.float)
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
        self.s_thresh = (100, 255)  # S channel threshold
        self.l_thresh = (50, 255)  # L channel threshold
        self.h_thresh = (0, 150)  # H channel threshold
        self.sx_thresh = (10, 255)  # Sobel x threshold
        self.sy_thresh = (0, 255)
        self.mag_thresh = (3, 255)
        #self.dir_thresh = (0.7, 1.3)
        self.sx_thresh = (10, 255)  # Sobel x threshold
        self.sy_thresh = (0, 255)
        self.mag_thresh = (5, 255)
        self.dir_thresh = (0.7, 1.3)
        self.color_binary_image = 0
        self.output_images_path = 'output_images'
        self.binary_image_path = ''.join([self.output_images_path, '\\', 'binary_test_images'])

    def apply_filter(self):
        yellow_binary_rgb = self.yellow_thresh_rgb()
        white_binary_rgb = self.white_thresh_rgb()
        yellow_thresh_hls = self.yellow_thresh_hls()
        yellow_binary_lab = self.yellow_thresh_lab()
        sxbinary = self.sobel_x_thresh(sx_thresh=self.sx_thresh)
        sybinary = self.sobel_y_thresh(sy_thresh=self.sy_thresh)
        dir_binary = self.dir_threshold(angle_thresh=self.dir_thresh)
        mag_binary = self.mag_threshold(mag_thresh=self.mag_thresh)

        # Combine the two binary thresholds
        combine_binary = np.zeros_like(yellow_binary_rgb)
        combine_binary[(((white_binary_rgb == 1) | (yellow_binary_rgb == 1) |
                         (yellow_thresh_hls == 1)) | (yellow_binary_lab == 1))
                          & (dir_binary == 1) & (mag_binary == 1)]= 1

        self.color_binary_image = combine_binary

    def save_filtered_image(self):
        if not os.path.exists(self.binary_image_path):
            os.makedirs(self.binary_image_path)
        pure_file_name = self.image_name[self.image_name.rfind('\\') + 1:]
        plt.imsave(''.join([self.binary_image_path, '\\', pure_file_name]), self.color_binary_image, cmap='gray')

    def yellow_thresh_lab(self):
        yellow_lab_low = np.array([0, 0, 0], dtype="uint8")
        yellow_lab_high = np.array([255, 115, 120], dtype="uint8")
        yellow_mask_lab_1 = cv2.inRange(self.lab, yellow_lab_low, yellow_lab_high)/255.0
        b_channel = self.lab[:, :, 2]
        if np.max(b_channel) > 175:
            b_channel = np.uint8(b_channel * (255 / np.max(b_channel)))
        yellow_mask_lab_2 = np.zeros_like(b_channel)
        yellow_mask_lab_2[(b_channel >= 200) & (b_channel <= 255)] = 1
        yellow_mask_lab = np.zeros_like(yellow_mask_lab_1)
        yellow_mask_lab[(yellow_mask_lab_1 == 1) | (yellow_mask_lab_2 == 1)] = 1
        return yellow_mask_lab

    def yellow_thresh_hls(self):
        yellow_hls_low = np.array([20, 120, 80], dtype="uint8")
        yellow_hls_high = np.array([45, 200, 255], dtype="uint8")
        yellow_mask_hls = cv2.inRange(self.hls, yellow_hls_low, yellow_hls_high)/255.0
        return yellow_mask_hls

    def yellow_thresh_rgb(self):
        #yellow_rgb_low = np.array([255, 180, 0], dtype="uint8")
        #yellow_rgb_high = np.array([255, 255, 210], dtype="uint8")
        yellow_rgb_low = np.array([255, 180, 0], dtype="uint8")
        yellow_rgb_high = np.array([255, 255, 210], dtype="uint8")
        yellow_mask_rgb = cv2.inRange(self.image, yellow_rgb_low, yellow_rgb_high)/255.0
        return yellow_mask_rgb

    def white_thresh_rgb(self):
        white_rgb_low = np.array([100, 100, 200], dtype="uint8")
        white_rgb_high = np.array([255, 255, 255], dtype="uint8")
        white_mask_rgb = cv2.inRange(self.image, white_rgb_low, white_rgb_high)/255.0
        return white_mask_rgb

    def sobel_x_thresh(self, sx_thresh=(0, 255)):
        l_channel = self.hls[:, :, 1]
        # Sobel x
        sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)  # Take the derivative in x
        abs_sobelx = np.absolute(sobelx)  # Absolute x derivative to accentuate lines away from horizontal
        scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

        # Threshold x gradient
        sxbinary = np.zeros_like(scaled_sobel)
        sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
        return sxbinary

    def sobel_y_thresh(self, sy_thresh=(0, 255)):
        l_channel = self.hls[:, :, 1]
        # Sobel x
        sobely = cv2.Sobel(l_channel, cv2.CV_64F, 0, 1)  # Take the derivative in x
        abs_sobely = np.absolute(sobely)  # Absolute x derivative to accentuate lines away from horizontal
        scaled_sobel = np.uint8(255 * abs_sobely / np.max(abs_sobely))

        # Threshold x gradient
        sxbinary = np.zeros_like(scaled_sobel)
        sxbinary[(scaled_sobel >= sy_thresh[0]) & (scaled_sobel <= sy_thresh[1])] = 1
        return sxbinary

    def s_channel_thresh(self, s_thresh=(0, 255)):
        s_channel = self.hls[:, :, 2]
        # Threshold color channel
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
        return s_binary

    def l_channel_thresh(self, l_thresh=(0, 255)):
        # Convert to HLS color space and separate the V channel
        l_channel = self.hls[:, :, 1]
        # Threshold color channel
        l_binary = np.zeros_like(l_channel)
        l_binary[(l_channel >= l_thresh[0]) & (l_channel <= l_thresh[1])] = 1
        return l_binary

    def h_channel_thresh(self, h_thresh=(0, 255)):
        h_channel = self.hls[:, :, 0]
        # Threshold color channel
        h_binary = np.zeros_like(h_channel)
        h_binary[(h_channel >= h_thresh[0]) & (h_channel <= h_thresh[1])] = 1
        return h_binary

    def mag_threshold(self, sobel_kernel=3, mag_thresh=(0, 255)):
        # Apply the following steps to img
        # 1) Convert to gray scale
        gray = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
        # 2) Take the gradient in x and y separately
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        # 3) Calculate the magnitude
        sobel_mag = np.sqrt(sobelx ** 2 + sobely ** 2)
        # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
        scaled_sobel_mag = np.uint8(255 * sobel_mag / np.max(sobel_mag))
        # 5) Create a binary mask where mag thresholds are met
        binary_output = np.zeros_like(scaled_sobel_mag)
        binary_output[(scaled_sobel_mag >= mag_thresh[0]) & (scaled_sobel_mag <= mag_thresh[1])] = 1
        # 6) Return this mask as your binary_output image
        return binary_output

    def dir_threshold(self, sobel_kernel=3, angle_thresh=(0, 255)):
        # 1) Convert to gray scale
        gray = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
        # 2) Take the gradient in x and y separately
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        # 3) Take the absolute value of the x and y gradients
        abs_sobelx = np.absolute(sobelx)
        abs_sobely = np.absolute(sobely)
        # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
        direction = np.arctan2(abs_sobely, abs_sobelx)
        # 5) Create a binary mask where direction thresholds are met
        binary_output = np.zeros_like(direction)
        binary_output[(direction >= angle_thresh[0]) & (direction <= angle_thresh[1])] = 1
        # 6) Return this mask as your binary_output image
        return binary_output

    def monte_carlo_simulation(self, test_number, monte_carlo_pdf, random_threshs):
        self.s_thresh = random_threshs['s_thresh']
        self.sx_thresh = random_threshs['sx_thresh']
        self.sy_thresh = random_threshs['sy_thresh']
        self.mag_thresh = random_threshs['mag_thresh']
        self.dir_thresh = random_threshs['dir_thresh']
        self.apply_filter()
        pure_file_name = self.image_name[self.image_name.rfind('\\') + 1:]
        parameters_string = 's_thresh: {} '.format(self.s_thresh)
        parameters_string = ''.join([parameters_string, 'sx_thresh: {}  '.format(self.sx_thresh)])
        parameters_string = ''.join([parameters_string, 'sy_thresh: {} \n'.format(self.sy_thresh)])
        parameters_string = ''.join([parameters_string, 'mag_thresh: {} '.format(self.mag_thresh)])
        parameters_string = ''.join([parameters_string, 'dir_thresh: {} \n '.format(self.dir_thresh)])
        parameters_string = ''.join([parameters_string, 'test_number: {}'.format(str(test_number))])
        title = ''.join([pure_file_name, '\n', parameters_string])
        fig, ax = plt.subplots(1, 1, figsize=[7, 7])
        ax.imshow(self.color_binary_image)
        ax.set_title(title)
        monte_carlo_pdf.savefig(fig)
        plt.close(fig)


def filter_test_images():
    for warp_image_name in glob.glob('output_images\\undistored_test_images\\*.jpg'):
        image = scipy.misc.imread(warp_image_name, mode="RGBA")[:, :, :3]
        find_test_image_lane_pixels = FindLaneLinePixels(image, warp_image_name)
        find_test_image_lane_pixels.apply_filter()
        find_test_image_lane_pixels.save_filtered_image()


def monte_carlo_simulation():
    monte_carlo_test_path = 'output_images\\monte_carlo_test\\'
    if not os.path.exists(monte_carlo_test_path ):
        os.makedirs(monte_carlo_test_path)
    monte_carlo_pdf = PdfPages(''.join([monte_carlo_test_path, 'monte_carlo_test.pdf']))
    for test_number in range(1, 50):
        print('simulation number: {}'.format(str(test_number)))
        random_threshs = {'s_thresh': (np.random.randint(70, 110, size=1),
                                       255),
                          'sx_thresh': (np.random.randint(10, 30, size=1),
                                        np.random.randint(80, 120, size=1)),
                          'sy_thresh': (0, 255),
                          'mag_thresh': (np.random.randint(10, 50, size=1),
                                         np.random.randint(80, 120, size=1)),
                          'dir_thresh':  (np.random.uniform(0.5, 0.8),
                                          np.random.uniform(1.1, 1.5))}
        for warp_image_name in glob.glob('output_images\\warp_test_images\\*.jpg'):
            image = mpimg.imread(warp_image_name)
            monte_carlo_test = FindLaneLinePixels(image, warp_image_name)
            monte_carlo_test.monte_carlo_simulation(test_number, monte_carlo_pdf, random_threshs)
    monte_carlo_pdf.close()


def main():
    filter_test_images()
    #monte_carlo_simulation()


if __name__ == '__main__':
    main()