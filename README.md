
**Vehicle Detection Project**
This is the 5th project of self driving car nano degree.

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Normalize features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run  pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/raw_images.jpg
[image2]: ./output_images/hog_image_sample.jpg
[image3]: ./output_images/bounding_boxes.JPG
[image4]: ./output_images/sliding_window.JPG
[image5]: ./output_images/heat_map.JPG
[image6]: ./output_images/labeled.JPG
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the first code cell of the IPython notebook **vehicle.detection.ipynb**  <br>

I started by reading in all the `vehicle` and `non-vehicle` images. images from the GTI vehicle image database GTI, and the KITTI vision benchmark suite. All images are 64x64 pixels. Here is an example of one of each of the `vehicle` and `non-vehicle` classes:<br>

![raw_images][image1]<br>

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.<br>

Here is an example using the `RGB` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:<br>

![hog_image_sample][image2]<br>
in the notebook, for each training images and test images, the HOG features are extracted through function: **extract_features** <br>


#### 2. Explain how you settled on your final choice of HOG parameters.

In order to achive better performance, I tried different HOG parameters as shown below. I iterate through all the color space, and see the accuracy and how it performs on the test images i grab from project video and test set.<br>

| Parameters        | Tune sets   								  | 
|:-----------------:|:-------------------------------------------:| 
| color_spaces      | 'RGB','HSV','LUV','HLS','YUV','YCrCb'       | 
| spatial_sizes     | 16,32      				            	  |
| spatial_feature   | True       					        	  |
| hist_feats        | True      					              |
| hog_feat          | True        								  |
| hist_bins      	| 16,32            							  |
| orients      		| 9           								  |
| pix_per_cells     | 8            								  |
| cell_per_block    | 2        								      |
| hog_channel       | 'ALL'        								  |

Through out all the color space, I found it very good with both 'YUV' and 'YCrCb'. My final choise is using YUV color space, which I found it to be faster in training speed than YCrCb.<br> The final parameters I chose is as follows:

| Parameters        | Tune sets   								  | 
|:-----------------:|:-------------------------------------------:| 
| color_spaces      | 'YUV'    | 
| spatial_sizes     | 32        				            	  |
| spatial_feature   | True       					        	  |
| hist_feats        | True      					              |
| hog_feat          | True        								  |
| hist_bins      	| 32              							  |
| orients      		| 9           								  |
| pix_per_cells     | 8            								  |
| cell_per_block    | 2        								      |
| hog_channel       | 'ALL'        								  |

I found that with more bins and nire spatial sizes, I got better bounding box result. So that's why I chose these parameters.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM Using LineanSVC from Sklearn package. I used default parameters for training, with totally 8460 features, combining the HOG features, historam features and spatial features.<br>
Note that I Normalize the features before feeding into SVC function. Also, the data are shuffled to make sure that test set, and training set have similar distrubution patterns.<br>
Below is the example of code that applies the normalizaiton and shuffling:
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=rand_state)
    
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X_train)
# Apply the scaler to X
X_train = X_scaler.transform(X_train)
X_test = X_scaler.transform(X_test)
```
The SVC is using 9 orientations, 8 pixels per cell, and 2 cells per block. The total training time is 35.59 sec, with a test accuracy of 0.9862.
### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I implemented Hog Sub-sampling Window Search, in order to be more efficient and more robust for finding the bounding box of the cars. I used the cells per step as 2, which means 75% overlap all the time between two search windows.<br>
I chose x start at 300 all the time to avoid false positives on the left side of the image.
I alse chose multiple window scale factor, with different combinations of ystart and ystop as follows:<br>

| scale factor       | ystart,ystop  | 
|:-----------------:|:-------------------:| 
|1                  | 390 ,464|
|1         | 415,484 |
|1.5   | 415,567|
| 1.8        | 410,624|
| 3          | 464,540|


These parameters are seleted based on fine tuning the bounding box behaviors in the test video. While this is not the correct way of tuning the parameter in real world, I think it's reasonable to tune this way in this project since the training set is relatively small for a very robust result. Also the algorithm is too simple to be generalized for real world scenarios.<br>
Below is an example of the image with sliding window covering areas.<br> 
![sliding_window][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

I searched on two scales using YUV 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, and apply sliding window search for each of the video frame. The examples of sliding window search for test images are shown below:<br> 
![bounding_boxes.JPG][image4]
Note that the through this point there are still false positives. Therefore, heapmap is used to combining the bounding boxes, as well as eliminating some false positives. 



### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./output_project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  
Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:
To optimized the algorithm, I used historical infomation to help filter out false positives. I learned this method from another repo which is working very well from frame to frame. Basically, I saved the previous 4 cycles' bounding boxes infromation to a list, and maintain the length of that list to be 5. When the heapmap is calcualted, I added these bounding boxes also to the calculation. through this, the randomness of generating some false positives are able to be eliminated.<br>

### Here are six frames and their corresponding heatmaps:

![heat_map.JPG][image5]

### Here the resulting bounding boxes drawn onto the original test images. For video file, each frame is drawn the same way.
![labeled][image6]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

* One of the issue is that the model I trained is with limited amount of the data, which means this model is not a generalized model, and too simple to cover real world scenarios. In real world, there are far more cars and non-cars patterns than in the traning and test sets.
* Parameters are tuned according to the test video, which are not a correct way if I want to a real working model.
* Features and algorithms are both too simple, which means complecated road conditions can easily fail the pipeline.
* The light conditions, shape of targets can easily fail the pipeline.
* To improve the accuracy of the algorithm:
1. better classifiers need to be used, such as YOLO, which is using neural network.
2. Larger amount of data are needed for a generalized model.
3. tracking algorithm is needed for smooth object tracking.

