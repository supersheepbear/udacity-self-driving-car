# **Finding Lane Lines on the Road** 

## Yang Xiong

### This project is the first project of self driving nano degreee


---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect  work in this report

---

### Reflection
### The example original image we wanna process is:
![](https://github.com/supersheepbear/CarND-LaneLines-P1-Yang/raw/master/readme_images/input.jpg)  

### My pipeline consisted of the following steps: 
### 1.converted the image to grayscale image , and apply Gaussian blur
Converted the image to grayscale, and apply Gaussian blur to the image.<br>
Through this step, we can remove some of the noise in the image.
```python
gray_image = grayscale(original_image)
gaussian_image = gaussian_blur(gray_image, kernel_size)
```
Here is an output image of the grey scale image after Gaussian blur:
![](https://github.com/supersheepbear/CarND-LaneLines-P1-Yang/raw/master/readme_images/grey.jpg)  

### 2.apply canny edge detection on image
Through canny edge detection, 
The parameter I choose for canny edge detection is:
```python
# Define the Canny edge detection parameters
low_threshold = 100
high_threshold = 180
```
The function to apply canny edge is:
```python
canny_image = canny(gaussian_image, low_threshold, high_threshold)
```
Here is an output image of the canny edge detection:
![](https://github.com/supersheepbear/CarND-LaneLines-P1-Yang/raw/master/readme_images/canny.jpg)  


### 3.apply mask on image
In the canny edge detection output, there are more edges than what we interested, as shown in the figure above.<br>
In order to filter out only the laneline, we apply a masking vertices which select the lane line area.<br>
Below is the masking vertices we choose:
```python
# Define the masking vertices
imshape = original_image.shape
vertices = np.array([[(40,imshape[0]),(imshape[1]/2.1, imshape[0]/1.65), (imshape[1]/1.9, imshape[0]/1.65), (imshape[1],imshape[0])]], dtype=np.int32)
```
The function for masking is:
```python
masked_image = region_of_interest(canny_image, vertices)
```
After masking, the canny edge image becomes:
![](https://github.com/supersheepbear/CarND-LaneLines-P1-Yang/raw/master/readme_images/masked.jpg)  

### 4.apply hough line detection on image:
Now use HoughLinesP with some parameters to filter out some edges which we don't want.
The parmaters chosen for this is as follows. Note that these parameters are acatully chosen according to my preference. They can be other comnations which are totally different.
```python
# Define the Hough transform parameters
rho = 1 # distance resolution in pixels of the Hough grid
theta = np.pi/180 # angular resolution in radians of the Hough grid
threshold = 4    # minimum number of votes (intersections in Hough grid cell)
min_line_len = 6 #minimum number of pixels making up a line
max_line_gap = 1  # maximum gap in pixels between connectable line segments
```
The function for applying hough line detection is:
```python
# Run Hough on edge detected image
hough_image = hough_lines(masked_image, rho, theta, threshold, min_line_len, max_line_gap,imshape)
```
#### In order to draw a single line on the left and right lanes, I modified the draw_lines() function by the following steps:
First of all, iterate through all the lines, and identify if the slope of the line is negative or positive. <br>
Through this step, we split the lines into left and right groups.<br>
For each of the group, stores the lines positions x1,x2,y1,y2 to  defined x and y lists in a specific order.<br>
After split the lines, apply linear regression to x and y lists. Through this we will get the slope and intercept of the left and right lines.<br>

The above steps give us the slope，intercept of the left and right line for current frame. However, for some frame in video, the line detection is bad because of some reasons,  which makes the drawn lines very unstable from frame to frame.<br>
In order to resolve this, I introduce the weighted average method from frame to frame as follows:<br>
```
        l_w_ave = l_w_ave*0.8+l_w*0.2
        l_b_ave = l_b_ave*0.8+l_b*0.2
        r_w_ave = r_w_ave*0.8+r_w*0.2
        r_b_ave = r_b_ave*0.8+r_b*0.2
```
This means we trust 80% of the past value and 20% of current value.<br>
This will mitigate the unstable performance.<br>

After this, we get the weighted slopes and intercepts. Then we can draw lines from bottom of the image to a defined height according to slopes and intercepts.<br>
The source code is as follows:

```python
def draw_lines(img, lines,imshape, color=[255, 0, 0], thickness=8):
    # define emyty lists to store left and right line points
    left_x = []
    right_x = []
    left_y = []
    right_y = []
    global image_count
    global l_w_ave
    global l_b_ave
    global r_w_ave
    global r_b_ave
    
    #iterate through all lines detected in current image
    for line in lines:
        for x1,y1,x2,y2 in line:
            # split the lines into two group: left and right, based on slope
            # should not consider a line with slope equals or close to infinity!
            if x2-x1!=0:
                slope = (float(y2-y1)/float(x2-x1)) 
                # store line end points into lists
                if slope<-0.5 and slope>-100 and x1<float(imshape[1])/2.:
                    left_x.append(x1)
                    left_x.append(x2)
                    left_y.append(y1)
                    left_y.append(y2)                

                elif slope>0.5 and slope<100 and x1>float(imshape[1])/2.:
                    right_x.append(x1)
                    right_x.append(x2)
                    right_y.append(y1)
                    right_y.append(y2)    
    # Use linear regression to get left line and right line slope and intercept
    l_w,l_b = np.linalg.lstsq(np.vstack([np.array(left_x), np.ones(len(left_x))]).T,np.array(left_y))[0]
    r_w,r_b = np.linalg.lstsq(np.vstack([np.array(right_x), np.ones(len(right_x))]).T,np.array(right_y))[0]
    
    # Apply weighted average to slope and intercept in each frame to avoid sudden changes
    # for the first frame, initialize the first weighted average slope and intercept
    if image_count==0:
        l_w_ave = l_w
        l_b_ave = l_b
        r_w_ave = r_w
        r_b_ave = r_b   
    # Apply weighted average to all frames except the first frame
    else:
        l_w_ave = l_w_ave*0.8+l_w*0.2
        l_b_ave = l_b_ave*0.8+l_b*0.2
        r_w_ave = r_w_ave*0.8+r_w*0.2
        r_b_ave = r_b_ave*0.8+r_b*0.2
    # increment the frame counter
    image_count+=1
    # use a fixed maximum y points for y1
    left_top_y = imshape[0]/1.65
    right_top_y = imshape[0]/1.65
    
    # calculate final left line points to draw on image frame
    l_y1 = int(round(left_top_y))
    l_x1 = int(round((float(l_y1-l_b_ave)/l_w_ave)))
    l_y2 = imshape[0]
    l_x2 = int(round(float(l_y2-l_b_ave)/l_w_ave))

    # calculate final right line points to draw on image frame 
    r_y1 = int(round(right_top_y))
    r_x1 = int(round((float(r_y1-r_b_ave)/r_w_ave)))
    r_y2 = imshape[0]
    r_x2 = int(round(float(r_y2-r_b_ave)/r_w_ave))
    # draw left line
    cv2.line(img, (l_x1, l_y1), (l_x2, l_y2), color, thickness)
    # draw right line
    cv2.line(img, (r_x1, r_y1), (r_x2, r_y2), color, thickness) 
```
The output drawn lines image will be like:
![](https://github.com/supersheepbear/CarND-LaneLines-P1-Yang/raw/master/readme_images/lines.jpg)  
### 5.Draw on the image
Combine lines image and original image to get the desired output:
```python
# Draw the lines on the original image
weighted_image = weighted_img(hough_image, original_image, α=0.8, β=1., λ=0.)
```
The final output image will be:

![](https://github.com/supersheepbear/CarND-LaneLines-P1-Yang/raw/master/readme_images/output.jpg)  

### shortcomings with current pipeline

* The masking area is chosen according to current video. It can be different ones. Further improvement are needed.
* The current drawn lines are not curved, which fits bad for the curved road.
* It doesn't work on the challenge video. I think main reason is because it detect some useless edges, and current strategy is not able to filter them out.
* It's not stable enough from frame to frame
* parameters are magic numbers. These parameters need further investigation.
* It's not able to draw lines if lane lines are missing in some frames 
* The algorithm may not work on night, snow, rain or other situations.


### 3. Suggest possible improvements to pipeline

* Apply smarter area masking strategies, which varies from video to video. I assume machine learning might be needed. 
* Draw curved lines on images. This may require multiorder regression instead of linear regression.
* Apply something like color mask to filter out only white and yellow lines. This may make the challenge video works.
* Tune parameters for different situations. This may require training for different videos.
* Apply traking algorithms instead of just weighted average method, to make drawn lines more stable, and to make it work if lines are missing in some of the frames. Kalman filter is a good choice.
