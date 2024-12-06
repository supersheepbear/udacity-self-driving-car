
**Advanced Lane Finding Project**
This is the 4th project of self driving nano degree
The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./camera_cal/calibration1.jpg "raw_calibration1"
[image2]: ./camera_cal/calibration2.jpg "raw_calibration2"
[image3]: ./output_images/undist_calibration_images/calibration1.jpg "undist_calibration1"
[image4]: ./output_images/undist_calibration_images/calibration2.jpg "undist_calibration2"
[image5]: ./output_images/perspective_transform_cali_images/calibration3.jpg "warped_calibration1"
[image6]: ./output_images/perspective_transform_cali_images/calibration17.jpg "warped_calibration17"
[image7]: ./test_images/test2.jpg "raw_test_2"
[image8]: ./output_images/undistored_test_images/test2.jpg "undist_test_2"
[image9]: ./output_images/binary_test_images/test2.jpg "binary_test_2"
[image10]: ./output_images/warp_test_images/test2.jpg "warped_test_2"
[image11]: ./output_images/curve_fit_images/test2.jpg "undist_test_2"
[image12]: ./output_images/final_image/test2.jpg "final_image"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The camera calibration is done in **camera_cali.py**<br>
First of all, iterate through the image folder to read all raw calibration images.<br>
The examples of raw images are shown below:<br>
raw calibration image 1:<br>
![raw calibration image 1][image1]<br>
raw calibration image 2:<br>
![raw calibration image 2][image2]<br>
To get the camera undistortion parameters, the first step is to convert images to gray scale using open cv function gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).<br>
As is shown above in the original calibration images, there are 9*6 conerers inside each of them. As a result, use the following code to generate object points numpy array, in order to be used in later calibration function:<br>
```python
objpoints = np.zeros((6 * 9, 3), np.float32)
objpoints[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
```
Then use cv2.findChessboardCorners to find out corresponding 9*6 corners in gray scale image. If the corners are found, append the corneres into a list<br>
```
ret, corners = cv2.findChessboardCorners(gray_img, (9, 6), None)
```
Till this point all the data are prepared. It is able to do camera calibrations according to these object points and corners, using the function below:<br>
```
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[1::-1], None, None)
```
with the parameters, now the raw calibration images will be able to be undistored. examples of undistorted calibration images are shown below:<br>
undistored calibration image 1:<br>
![undistored calibration image 1][image3]<br>
undistored calinration image 2:<br>
![undistored calibration image 2][image4]<br>
Other undistored calibration images can be found in folder **output_images\undist_calibration_images**<br>
Because the camera undistortion parameters mtx and dist should be unchanged, save these parameters in a local pickle file called: 'camera_cali_parameters.pickle'. When running the final image pipeline, I just have to load this pickle file and get the matrix for image undistortion.<br>

Also, do a perspective transform experiment with the undistored calibration images. The source and destination points are seleted automatically by the algorithm, and the perspective transform is done according to code below.<br>
``` python
# Given src and dst points, calculate the perspective transform matrix
pers_trans_matrix = cv2.getPerspectiveTransform(src, dst)
# Warp the image using OpenCV warpPerspective()
pers_trans_image = cv2.warpPerspective(undistorted_img, pers_trans_matrix, img_size)
```
Here are the examples of the perspective transformed calibration images.<br>
warped calibration image 1:<br>
![warped calibration image 3][image5]<br>
warped calibration image 2:<br>
![warped calibration image 4][image6]<br>
Other warped calibration images can be found in folder **output_images\perspective_transform_cali_images**<br>

### Pipeline (single images)

#### 1. example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:<br>
![raw_test_2][image7]<br>
All the image distortion is done in module undistort_image.py<br>
First step is to load the camera parameter matrixs **mtx** and **dist** from **camera_cali_parameters.pickle**. Please note that these parameters are calculated from the camera calibraion step.<br>
Use the function below, we can get the undistored image. <br>
```
self.undistorted_image = cv2.undistort(self.raw_image, self.mtx, self.dist, None, self.mtx)
```
The undistored example image is shown below:<br>
![undist_test_2][image8]<br>
Other undistored test images can be found in folder **output_images\undistored_test_images**<br>

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

My binary filtering is done in module **binary_filter.py**<br>
The class FindLaneLinePixels is for appling binary filter to undistorted image.<br>
There are many methods in the class, but only part of them is used for final submission.<br>
I did not use the original way of color filtering in lectures. Instead, I am trying to identify yellow and white colors in RGB and HLS.<br>
cv2.inRange is used to calculate the binary mask.<br>
The following mask is an exmple of how I apply the threshold to the mask<br>.
```
yellow_mask_hls = cv2.inRange(self.hls, yellow_hls_low, yellow_hls_high)/255.0
```
Also, I apply the sobelx ,sobely, absolute sobel and sobel direction threshold, to make sure that the final filtered pixels fulfill the derivative criteria of lane lines, and have good filter quality on test images.<br>
Finally, I apply a combine_binary as below.<br>
```
combine_binary[(((white_binary_rgb == 1) | (yellow_binary_rgb == 1) |
                         (yellow_thresh_hls == 1)) | (yellow_binary_lab == 1))
                          & (dir_binary == 1) & (mag_binary == 1)]= 1
```
To sum up, the idea behind my logic is to find white and yellow colors, but only with desired sobel threshold. <br>
The way I tuned the thresholds is to do iterative tests on provided test images, to get the best quality on all of them. Ideally, these thresholds shall be tuned in more data, to make sure the binary filter is more generic to different kinds of data.<br>
The binary filter example
![binary_test_2][image9]<br>
Other binary test images can be found in folder **output_images\binary_test_images**<br>

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.
Perspective transform is to apply birds-eye view to the image.
The module **perspective_transform.py** is used to do the perspective transfrom in the pipeline.<br>
I have a class PerspectiveTransform which is to handle a single image perspective transform.
The following code is to perform perspective transform for a single image. self.minv is the reverse perspective transform matrix which will be used lated.<br>

```python
pers_trans_matrix = cv2.getPerspectiveTransform(self.source_points, self.desired_points)
        self.warped_image = cv2.warpPerspective(self.image, pers_trans_matrix, image_size, flags=cv2.INTER_LINEAR)
        self.minv = cv2.getPerspectiveTransform(self.desired_points, self.source_points)
```
The current source and desired points are hard coded.
```python
self.source_points = np.float32([[self.image.shape[1] - 580, self.image.shape[0]/2+100],
                                  [self.image.shape[1] - 200, self.image.shape[0]],
                                  [200, self.image.shape[0]],
                                  [580, self.image.shape[0]/2+100]])
self.desired_points = np.float32([[self.image.shape[1] - 380, 0],
                                  [self.image.shape[1] - 380, self.image.shape[0]],
                                  [380, self.image.shape[0]],
                                  [380, 0]])
```
This is equivalent to:
```
self.source_points = [[ 700.  460.]
                      [1080.  720.]
                      [ 200.  720.]
                      [ 580.  460.]]
self.desired_points = [[900.   0.]
                       [900. 720.]
                       [380. 720.]
                       [380.   0.]]
```
I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.
The warped exmple image is show below.<br>
![warped_test_2][image10]<br>
Other warped images can be found in folder **output_images\warp_test_images**<br>

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

I used the sliding window search method to find lane line pixels.<br>
All the find lane pixels and curve fit class and functions are in module **calculate_curvature.py**
The ideal behind sliding window search, is to find peaks in pixels histogram, for a certain vertical range of the image.<br>
I use 26 windows for a single image, 13 for left lane line, and 13 for right lane line.<br>
The 'good pixels' which are found by algorithm are saved to numpy arrays. These arrays are used later for curve fitting.<br>
Because I found that pixel values of lane lines are less accurate on the top of the image,  I apply weight polynomial fit to all the lane line points. I applied more weights to the pixels on the bottom side, while apply less weights for the above pixels. This makes the fit look better for test images. Please note, in the video frame, I use historical information to do a low pass filter, to stablize the lane line fit.<br>
The example code for weighter curve is as shown below:<br>
```python
self.left_fit = np.polyfit(lefty, leftx, 2, w=self.left_fit_weight)
```
Then according to the curve fit parameters, we can get an example of curve fit image as shown below:
![curve_fit_image][image11]
Other curve fit images can be found in folder **output_images\curve_fit_images**<br>

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.
In order to determine the lane lines curvature, and the vehicle position, the pixel values need to be transfered to meters. The following parameters are used for the transformation.
```python
# Define conversions in x and y from pixels space to meters
ym_per_pix = 30/720. # meters per pixel in y dimension
xm_per_pix = 3.7/700. # meters per pixel in x dimension
```
By these two parameters, the polynomial is adjusted to match the real world meters.<br>
Then the curvature of the lane can be calculated by:<br>

```python
left_curverad_meter  = ((1 + (2 * left_fit_cr[0] * y_eval + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit_cr[0])
right_curverad_meter  = ((1 + (2 * right_fit_cr[0] * y_eval + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit_cr[0])
curverad_meter = (left_curverad_meter + right_curverad_meter)/2
```
also, the vehicle position is able to be calculated, by substracting the middle of the image point from the center of the lane point.<br>
For example image, the curvature is 2616 meter, while the vehicle position is -0.37 meter.<br>
These values are reasonable according to lectures.<br>
In the video, I also print the curvature and vehicle position on the top of each frame
#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.
Finally, in process_video.py, I plot back the lane line into each video frame, if the lane line is detected.<br>
The drawing function is:
```
# Draw the lane onto the warped blank image
cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
```
color_warp is the original undistored image, and np.int_([pts] are four corner positions for the lane drawing areas.<br>
Below is an example of one of the video frame marked with lane line area:

![final_image][image12]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./processed_project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?
* First of all, I haven't implement anything that include historical infomation other that the fitting parameters. What this means is that I only finish the rubric points for now, but this is too simple, and will not work well on challenge videos.If any of the frame, the lane lines are really not there, or the lane finding results is bad, the algorithm won't work well. I have to do more things after submitting the project, including dealing with challenging videos, and make sure to use other historical information.
* Of course, in real world situations, it will become more challenging. Any noise will fail the lane dectections from my pipeline. e.g: In fog, in rain, in heavy sunlight, in snow. 
* The binary image filter have guessing values for the thresholds, which can be be improved. Ideally, there can be more logic there, and the threshold shall be tuned according to real world situations. More samples shall be included for the filter. Also, other color space maybe tried to get better results. Shadows, road surface color, brightness will affect the binary result, and fail to identify lane lines. Therefore, the filter shall have more robust logic for each of the situation.
* The perspective transform can be more precise to get better warped image, so that the curvature calulcation is better.
* The window search method shall use historical information to filter out bad areas.
* There can be a better way of stablize the lane lines instead of low pass filter.


