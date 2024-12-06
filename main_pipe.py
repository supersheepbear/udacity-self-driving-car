import cv2
import perspective_transform as pers_trans
import binary_filter
import calculate_curvature as curve_fit
import matplotlib.pyplot as plt
import numpy as np
#plt.ion()


class MainPipeline:
    def __init__(self, raw_image, camera_parameters, image_name):
        self.raw_image = raw_image
        self.image_name = image_name
        self.camera_parameters = camera_parameters
        self.dist = 0
        self.mtx = 0
        self.undistorted_image = 0
        self.gray_image = 0
        self.binary_image = 0
        self.warped_image = 0
        self.left_fit = []
        self.right_fit = []
        self.left_curvature = 0
        self.right_curvature = 0
        self.left_detected = False
        self.right_detected = False
        self.line_base_pos = 0
        self.minv = 0
        self.curvature = 0
        self.line_base_pos = 0

    def pipeline_main(self):
        """Main method for the class"""
        self.undistort_image()
        self.binary_filter_image()
        self.perspective_transform()
        self.find_curvature()

    def undistort_image(self):
        """Undistort a single image"""
        self.mtx = self.camera_parameters['mtx']
        self.dist = self.camera_parameters['dist']
        self.undistorted_image = cv2.undistort(self.raw_image, self.mtx, self.dist, None, self.mtx)

    def apply_gray_scale(self):
        self.gray_image = cv2.cvtColor(self.undistorted_image, cv2.COLOR_BGR2GRAY)

    def perspective_transform(self):
        color_binary_image = np.dstack((self.binary_image, self.binary_image, self.binary_image))
        image_pers_trans = pers_trans.PerspectiveTransform(color_binary_image, self.image_name)
        image_pers_trans.warp_image()
        self.warped_image = image_pers_trans.warped_image
        self.minv = image_pers_trans.minv

    def binary_filter_image(self):
        get_binary_image = binary_filter.FindLaneLinePixels(self.undistorted_image, self.image_name)
        get_binary_image.apply_filter()
        self.binary_image = get_binary_image.color_binary_image

    def find_curvature(self):
        gray_warped_image = (self.warped_image[:, :, 0])*255
        get_curvature = curve_fit.FindCurvature(gray_warped_image, self.image_name)
        get_curvature.find_curvature_fit()
        get_curvature.get_curvature()
        get_curvature.plot_curve_fit()
        self.left_fit = get_curvature.left_fit
        self.right_fit = get_curvature.right_fit
        self.left_curvature = get_curvature.left_curvature
        self.right_curvature = get_curvature.right_curvature
        self.curvature = get_curvature.curvature
        self.line_base_pos = get_curvature.line_base_pos



