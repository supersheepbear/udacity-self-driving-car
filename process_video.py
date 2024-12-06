import imageio
from moviepy.editor import VideoFileClip
import main_pipe
import undistort_images
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#plt.ion()


class ProcessVideo:

    def __init__(self, video_name):
        self.video = video_name

    def video_pipeline(self):
        clip1 = VideoFileClip(self.video)
        white_clip = clip1.fl_image(process_video_image)
        white_clip.write_videofile(''.join(['processed_', self.video]), audio=False)


class FrameInfo():
    def __init__(self):
        self.curvature = 0
        self.cycle = 0
        self.line_base_pos = 0
        # was the line detected in the last iteration?
        self.left_detected = False
        self.right_detected = False
        # x values of the last n fits of the line
        self.left_recent_xfitted = []
        self.right_recent_xfitted = []
        # average x values of the fitted line over the last n iterations
        self.left_bestx = None
        self.right_bestx = None
        # polynomial coefficients averaged over the last n iterations
        self.left_best_fit = None
        self.right_best_fit = None
        # polynomial coefficients for the most recent fit
        self.left_fit = [np.array([False])]
        self.right_fit = [np.array([False])]
        self.left_recent_fit = [np.array([False])]
        self.right_recent_fit = [np.array([False])]
        # radius of curvature of the line in some units
        self.left_radius_of_curvature = None
        self.right_radius_of_curvature = None
        # distance in meters of vehicle center from the line
        self.line_base_pos = None
        # difference in fit coefficients between last and new fits
        self.left_diffs = np.array([0, 0, 0], dtype='float')
        self.right_diffs = np.array([0, 0, 0], dtype='float')
        # x values for detected line pixels
        self.allx = None
        # y values for detected line pixels
        self.ally = None
        self.minv = 0
        self.warp_image = None
        self.undistort_image = None

    def get_curvature(self):
        y_eval = self.warp_image.shape[0] - 1
        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30/720. # meters per pixel in y dimension
        xm_per_pix = 3.7/700. # meters per pixel in x dimension


        mx = xm_per_pix
        my = ym_per_pix
        left_fit_cr = [mx / (my ** 2) * self.left_best_fit[0], (mx/my) * self.left_best_fit[1], self.left_best_fit[2]]
        right_fit_cr = [mx / (my ** 2) * self.right_best_fit[0], (mx/my) * self.right_best_fit[1], self.right_best_fit[2]]
        # Calculate the new radii of curvature
        left_curverad_meter  = ((1 + (2 * left_fit_cr[0] * y_eval + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit_cr[0])
        right_curverad_meter  = ((1 + (2 * right_fit_cr[0] * y_eval + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit_cr[0])
        #print(left_curverad_meter, 'm', right_curverad_meter, 'm')
        left_curvature = left_curverad_meter
        right_curvature = right_curverad_meter
        self.curvature = (left_curvature + right_curvature)/2

        # find end vehicle position
        left_x = self.left_best_fit[0] * (y_eval** 2) + self.left_best_fit[1] * y_eval + self.left_best_fit[2]
        right_x = self.right_best_fit[0] * (y_eval** 2) + self.right_best_fit[1] * y_eval + self.right_best_fit[2]
        self.line_base_pos = (self.warp_image.shape[1]/2 - (left_x + right_x)/2) * mx

    def cycle_update(self):
        self.cycle += 1

        if len(self.left_recent_xfitted) >= 5:
            self.left_best_fit = np.sum(self.left_recent_xfitted, axis=0)/5 * 0.5 + self.left_fit * 0.5
            self.right_best_fit = np.sum(self.right_recent_xfitted, axis=0)/5 * 0.5 + self.right_fit * 0.5
            self.left_recent_xfitted.pop(0)
            self.right_recent_xfitted.pop(0)

        else:
            self.left_best_fit = self.left_fit
            self.right_best_fit = self.right_fit
        self.left_recent_xfitted.append(np.array(self.left_fit))
        self.right_recent_xfitted.append(np.array(self.right_fit))

    def draw_line_area(self):
        self.cycle_update()
        self.get_curvature()
        # Create an image to draw the lines on
        warped = self.warp_image[:, :, 0]
        warp_zero = np.zeros_like(warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
        #ploty = np.linspace(0, warped.shape[0] - 1, num=warped.shape[0])
        ploty = np.linspace(0, warped.shape[0] - 1, (warped.shape[0]))
        if len(self.left_best_fit) != 0 and len(self.right_best_fit) != 0:
            left_fitx = self.left_best_fit[0] * ploty ** 2 + self.left_best_fit[1] * ploty + self.left_best_fit[2]
            right_fitx = self.right_best_fit[0] * ploty ** 2 + self.right_best_fit[1] * ploty + self.right_best_fit[2]
            # Recast the x and y points into usable format for cv2.fillPoly()
            pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
            pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
            pts = np.hstack((pts_left, pts_right))
            #print(self.undistort_image.shape)
            #plt.imshow(self.undistort_image)
            #print(color_warp.shape)
            # Draw the lane onto the warped blank image
            cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

            # Warp the blank back to original image space using inverse perspective matrix (Minv)
            newwarp = cv2.warpPerspective(color_warp, self.minv, (color_warp.shape[1], color_warp.shape[0]))
            #plt.imshow(newwarp)
            # Combine the result with the original image
            #print(newwarp.shape)
            result = cv2.addWeighted(self.undistort_image, 1, newwarp, 0.3, 0)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(result, 'curvature: {:.1f} m'.format(self.curvature), (230, 50), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(result, 'vehicle position: {:.4f} m'.format(self.line_base_pos), (230, 80), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            #plt.imshow(result)

            #plt.imsave('new_warp.jpg', result)
            return result
        else:
            return self.undistort_image


get_undistort_parameters = undistort_images.UndistortTestImages()
get_undistort_parameters.get_camera_cali_parameters()
camera_parameters = get_undistort_parameters.camera_parameters
current_frame_info = FrameInfo()


def process_video_image(image):
    image_main_pipe = main_pipe.MainPipeline(image, camera_parameters, 'video_image.jpg')
    image_main_pipe.pipeline_main()
    current_frame_info.warp_image = image_main_pipe.warped_image
    current_frame_info.undistort_image = image_main_pipe.undistorted_image
    current_frame_info.left_fit = image_main_pipe.left_fit
    current_frame_info.right_fit = image_main_pipe.right_fit
    current_frame_info.minv = image_main_pipe.minv
    current_frame_info.curvature = image_main_pipe.curvature
    current_frame_info.line_base_pos = image_main_pipe.line_base_pos
    labeled_image = current_frame_info.draw_line_area()

    return labeled_image


def video_process_main():
    video_process = ProcessVideo('project_video.mp4')
    #video_process = ProcessVideo('challenge_video.mp4')
    #video_process = ProcessVideo('harder_challenge_video.mp4')
    video_process.video_pipeline()


if __name__ == '__main__':
    video_process_main()