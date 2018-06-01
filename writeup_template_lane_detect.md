## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.


## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the function "calib_camera()" of the IPython notebook located in "advanced_lane_detect.ipynb"  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

[image7](output_images/camera_undistort.png "Undistorted camera image")


### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

Here is an example of a distorted correct image.
[image8](output_images/img_undistort.png "Undistorted road image")

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image.I converted the image from RGB to HLS color space and extracted the s channel within a threshold. Edge Detection(gradient along x) was performed on the original image(grayscale) and then combined with the s channel in the later step.

[image9](output_images/color_gradient.png "Color and gradient combined")

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform(birds eye view) includes a function called corners_warp . I chose to hardcode the source and destination points. The 'M'  matrix is calculated between src and dst points and then the image was warped into a birds eye view.

This resulted in the following source and destination points:
 src = np.float32([[210, 720], [1150,720], [580,450],[750,450]])
 dst = np.float32([[320, 720], [1000,720],   [120,2], [1000,2]])

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear almost parallel in the binary warped image.

[image10](output_images/warped_binary.png "Binary warped image")

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?
The functions "find_lanes_first_im" and "find_lines_remaining" contain code to fit polynomials for both the lanes.A histogram for the lower half of the image is calculated.The image was divided into 2 halves and the index of the max value was considered.A window was fitted around the point and all non zero pixels were collected.Selected indices would be the ones within the limits of each window.All these indices corresponding to non zero pizels would be averaged to calculate the mean index.As we go up the image ,we get a set of points which define a polynomial.

[image11](output_images/plot_curves.png "Fitted Polynomials")

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The function "radius_curv" calculated the radius and the position of the vehicle w.r.t center.The radius for both left and right lanes were calculated accordingly and averaged.The center was calculated by taking the average of left and right lanes and subtracting it from the original center of the image.(vehicle's center).
Here is a pic for the formula used for the above:
[image12](output_images/radius_curv.jpg "Formula")

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

[image13](output_images/lane_detect.png "Image with lane area filled")

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

[Video](project_video_output.mp4)


### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I had issues in selecting the source and destination points for the perspective transform but finally managed to get appropriate values. Snowy roads might cause the problem to fail if the lanes are not visible for the first image.Thresholding L channel and using Gaussian filters can make the detection more robust to nouse.
