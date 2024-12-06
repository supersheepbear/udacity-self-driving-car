import cv2
import os
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


class CameraCalibration:
    """Class for camera calibration
    """
    def __init__(self):
        self.images = []  # array of all calibration images' names
        self.objpoints = []  # array to store all images' object points
        self.imgpoints = []  # array to store all images' found corners
        self.corners_nx = 9
        self.corners_ny = 6
        self.mtx = 0
        self.dist = 0

    def camera_calibration_main(self):
        self.open_images()
        self.find_chessboard_corners()
        self.get_undistort_cali_images()

    def open_images(self):
        """Read in all calibration images
        """
        self.images = glob.glob('camera_cal/calibration*.jpg')
        print('reading images done.')

    def find_chessboard_corners(self):
        """Find chessboard corners from each image
        """
        'Start finding chessboard corners'
        # Prepare object points
        objpoints = np.zeros((6 * 9, 3), np.float32)
        objpoints[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
        for filename in self.images:
            # Read image and convert to gray scale
            img = mpimg.imread(filename)
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Use openCV to find chess board corners
            ret, corners = cv2.findChessboardCorners(gray_img, (9, 6), None)

            # If corners found, add object points, image points to lists
            if ret:
                self.imgpoints.append(corners)
                self.objpoints.append(objpoints)
        print('Finding chessboard corners down')

    def get_undistort_cali_images(self):
        """Undistort calibration images if corners found
        """
        # Create a folder to save images
        undistort_directory = 'output_images\\undist_calibration_images\\'
        if not os.path.exists(undistort_directory):
            os.makedirs(undistort_directory)

        # Undistort each calibration image
        for filename in self.images:
            img = mpimg.imread(filename)
            ret, self.mtx, self.dist, rvecs, tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints,
                                                               img.shape[1::-1], None, None)
            undistorted_img = cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
            # save undistored_img to undistorted_images folder
            pure_filename = filename[filename.rfind('\\')+1:]
            undistort_filename = ''.join([undistort_directory, pure_filename])
            plt.imsave(undistort_filename, undistorted_img)

    def save_camera_parameters(self):
        # Save calibration matrix and distortion coefficients to local dictionary
        camera_cali_paras = {'mtx': self.mtx, 'dist': self.dist}
        with open('camera_cali_parameters.pickle', 'wb') as handle:
            pickle.dump(camera_cali_paras, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def perspective_transform(self):
        self.open_images()
        print('Perform perspective transform')
        with open('camera_cali_parameters.pickle', 'rb') as camera_parameters_file:
            camera_parameters = pickle.load(camera_parameters_file)
        pers_trans_directory = 'output_images\\perspective_transform_cali_images\\'
        if not os.path.exists(pers_trans_directory):
            os.makedirs(pers_trans_directory)
        for filename in self.images:
            print('Doing perspective transform for image: {}'.format(filename))
            # Search for corners in the gray scaled image
            img = mpimg.imread(filename)
            undistorted_img = cv2.undistort(img, camera_parameters['mtx'], camera_parameters['dist'],
                                            None, camera_parameters['mtx'])
            gray = cv2.cvtColor(undistorted_img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (self.corners_nx, self.corners_ny), None)
            if ret:
                # If we found corners, draw them! (just for fun)
                cv2.drawChessboardCorners(undistorted_img, (self.corners_nx, self.corners_ny), corners, ret)
                # Choose offset from image corners to plot detected corners
                offset = 50  # offset for dst points
                # Grab the image shape
                img_size = (gray.shape[1], gray.shape[0])
                # For source points I'm grabbing the outer four detected corners
                src = np.float32([corners[0], corners[self.corners_nx - 1], corners[-1], corners[-self.corners_nx]])
                dst = np.float32([[offset, offset], [img_size[0] - offset, offset],
                                  [img_size[0] - offset, img_size[1] - offset],
                                  [offset, img_size[1] - offset]])
                # Given src and dst points, calculate the perspective transform matrix
                pers_trans_matrix = cv2.getPerspectiveTransform(src, dst)
                # Warp the image using OpenCV warpPerspective()
                pers_trans_image = cv2.warpPerspective(undistorted_img, pers_trans_matrix, img_size)
                pure_filename = filename[filename.rfind('\\') + 1:]
                pers_trans_filename = ''.join([pers_trans_directory, pure_filename])
                plt.imsave(pers_trans_filename, pers_trans_image)


def main():

    camera_cali = CameraCalibration()
    camera_cali.camera_calibration_main()
    camera_cali.perspective_transform()


if __name__ == '__main__':
    main()