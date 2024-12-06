import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
import matplotlib.image as mpimg
import os
import math
#plt.ion()


class FindCurvature:
    def __init__(self, image, image_name):
        self.left_fit = []
        self.right_fit = []
        self.left_lane_inds = []
        self.right_lane_inds = []
        self.left_fit_weight = []
        self.right_fit_weight = []
        self.curvature = 0
        self.out_img = 0
        self.nonzeroy = 0
        self.nonzerox = 0
        self.leftx = 0
        self.lefty = 0
        self.rightx = 0
        self.righty = 0
        self.left_pixel_curvature = 0
        self.right_pixel_curvature = 0
        self.left_curvature = 0
        self.right_curvature = 0
        self.line_base_pos = 0
        self.binary_warped = image
        self.image_name = image_name
        self.output_images_path = 'output_images'
        self.curve_fit_image_path = ''.join([self.output_images_path, '\\', 'curve_fit_images'])

    def find_curvature_fit(self):
        # Assuming you have created a warped binary image called "self.binary_warped"
        # Take a histogram of the bottom half of the image
        histogram = np.sum(self.binary_warped[self.binary_warped.shape[0]//2:,:], axis=0)
        # Create an output image to draw on and  visualize the result
        self.out_img = np.dstack((self.binary_warped, self.binary_warped, self.binary_warped))*255
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]//2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Choose the number of sliding windows
        nwindows = 13
        # Set height of windows
        window_height = np.int(self.binary_warped.shape[0]//nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = self.binary_warped.nonzero()
        self.nonzeroy = np.array(nonzero[0])
        self.nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        # Set the width of the windows +/- margin
        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 50

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = self.binary_warped.shape[0] - (window+1)*window_height
            win_y_high = self.binary_warped.shape[0] - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            # Draw the windows on the visualization image
            cv2.rectangle(self.out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
            cv2.rectangle(self.out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((self.nonzeroy >= win_y_low) & (self.nonzeroy < win_y_high) &
            (self.nonzerox >= win_xleft_low) &  (self.nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((self.nonzeroy >= win_y_low) & (self.nonzeroy < win_y_high) &
            (self.nonzerox >= win_xright_low) &  (self.nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            self.left_lane_inds.append(good_left_inds)
            self.right_lane_inds.append(good_right_inds)
            self.left_fit_weight.append(np.ones(len(good_left_inds))/float(window + 1))
            self.right_fit_weight.append(np.ones(len(good_right_inds)) / float(window + 1))
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(self.nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(self.nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        self.left_lane_inds = np.concatenate(self.left_lane_inds)
        self.right_lane_inds = np.concatenate(self.right_lane_inds)
        self.left_fit_weight = np.concatenate(self.left_fit_weight)
        self.right_fit_weight = np.concatenate(self.right_fit_weight)
        # Extract left and right line pixel positions
        leftx = self.nonzerox[self.left_lane_inds]
        lefty = self.nonzeroy[self.left_lane_inds]
        rightx = self.nonzerox[self.right_lane_inds]
        righty = self.nonzeroy[self.right_lane_inds]
        self.leftx = leftx
        self.lefty = lefty
        self.rightx = rightx
        self.righty = righty
        if len(self.leftx) and len(self.leftx) > 0:
            # Fit a second order polynomial to each
            self.left_fit = np.polyfit(lefty, leftx, 2, w=self.left_fit_weight)
        else:
            self.left_fit = np.array([0, 0, 0])
        if len(self.righty) and len(self.rightx) > 0:
            self.right_fit = np.polyfit(righty, rightx, 2, w=self.right_fit_weight)
        else:
            self.right_fit = np.array([0, 0, 0])

    def plot_curve_fit(self):
        if not os.path.exists(self.curve_fit_image_path):
            os.makedirs(self.curve_fit_image_path)

        # Generate x and y values for plotting
        ploty = np.linspace(0, self.binary_warped.shape[0] - 1, self.binary_warped.shape[0])
        left_fitx = self.left_fit[0] * ploty ** 2 + self.left_fit[1] * ploty + self.left_fit[2]
        right_fitx = self.right_fit[0] * ploty ** 2 + self.right_fit[1] * ploty + self.right_fit[2]
        self.out_img[self.nonzeroy[self.left_lane_inds], self.nonzerox[self.left_lane_inds]] = [255, 0, 0]
        self.out_img[self.nonzeroy[self.right_lane_inds], self.nonzerox[self.right_lane_inds]] = [0, 0, 255]
        plt.imshow(self.out_img)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
        pure_file_name = self.image_name[self.image_name.rfind('\\') + 1:]
        #plt.savefig(''.join([self.curve_fit_image_path, '\\', pure_file_name]))
        plt.clf()

    def get_curvature(self):
        y_eval = self.binary_warped.shape[0] - 1

        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30/720. # meters per pixel in y dimension
        xm_per_pix = 3.7/700. # meters per pixel in x dimension


        mx = xm_per_pix
        my = ym_per_pix
        left_fit_cr = [mx / (my ** 2) * self.left_fit[0], (mx/my) * self.left_fit[1], self.left_fit[2]]
        right_fit_cr = [mx / (my ** 2) * self.right_fit[0], (mx/my) * self.right_fit[1], self.right_fit[2]]
        # Calculate the new radii of curvature
        left_curverad_meter  = ((1 + (2 * left_fit_cr[0] * y_eval + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit_cr[0])
        right_curverad_meter  = ((1 + (2 * right_fit_cr[0] * y_eval + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit_cr[0])
        #print(left_curverad_meter, 'm', right_curverad_meter, 'm')
        self.left_curvature = left_curverad_meter
        self.right_curvature = right_curverad_meter
        self.curvature = (self.left_curvature + self.right_curvature)/2

        # find end vehicle position
        left_x = self.left_fit[0] * (y_eval** 2) + self.left_fit[1] * y_eval + self.left_fit[2]
        right_x = self.right_fit[0] * (y_eval** 2) + self.right_fit[1] * y_eval + self.right_fit[2]

        self.line_base_pos = (self.binary_warped.shape[1]/2 - (left_x + right_x)/2) * mx
        #print(self.binary_warped.shape[1])
        #print(right_x)
        #print(self.image_name)
        #print(self.line_base_pos, 'm')


def image_curve_fit():
    for warped_binary_image_name in glob.glob('output_images\\warp_test_images\\*.jpg'):
        warped_binary_image = cv2.imread(warped_binary_image_name, 0)
        #print(np.max(warped_binary_image))
        test_image_fit_curve = FindCurvature(warped_binary_image, warped_binary_image_name)
        test_image_fit_curve.find_curvature_fit()
        test_image_fit_curve.plot_curve_fit()
        test_image_fit_curve.get_curvature()


def main():
    image_curve_fit()


if __name__ == '__main__':
    main()