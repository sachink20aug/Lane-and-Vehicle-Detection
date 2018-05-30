## Writeup Template
### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the function "get_hog_features" in the file "car_vd.py"

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

[image8]: output_images/car_image.png
[image9]: output_images/not_car_image.png

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example with HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

[image10]: output_images/hog_example.png

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and finalized my list based on how the project video was able to detect cars.
HLS,HSV and RGB did not yield robust detections on test images.

Here are the parameters finalized:
color_space='YcrCb'
Orientations=9
pixels_per_cell=8
cells_per_block=2
bins=32
spatial_size=32x32


#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I used StandardScaler function from sklearn to format the data so that the mean is 0 and SD is 1.I then split the data into training and test. A Linear SVM was used to classify and calculate the accuracy.(line 124-185 in the code).This same classifier was used to detect bounding boxes for the image at the later stages.Spatial,Histogram and HOG features were combined for the feature extraction.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?
The function "find_cars" in the code has the steps entailing SLIDING WINDOW search.Sliding images had a 64x64 resolution which is same as the given training images.Following are the steps implemented:
For example, a 256 pixel image and a 64x64 sliding window would result in a rate of 4 cells per step with 8x8 pix per cell.

    window=64, pix_per_cell=8,cell_per_block=2
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_window=(window // pix_per_cell) 
    cells_per_step = 2  
    step_per_win=cells_per_window//cells_per_step
    nxsteps = (x_windows-1)*step_per_win
    nysteps = (y_windows-1)*step_per_win

To SCALE, I used 4 different sets of scales namely 0.75,1,1.25 and 1.5.It is because the size of the cars in the video can vary based on how far or close they are to our video screen. Some of the values were given in the lecture slides and others were selected through trial and error.


#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on 4 scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result. To optimize,I used SVM classifier and only stored the bounding boxes which detect a car with a probability of 1 which discards other detections.  Here are some example images:

[image11]: output_images/test_output1.png
[image12]: output_images/test_output2.png
[image13]: output_images/test_output3.png


### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
The video for the output result is "project_video_output.mp4"
[video2](project_video_output.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions. A heat of 1 was added for each detection. I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  I also averaged upto a maximum of 15 heatmaps to remove false positives and overlapping boxes for a robust detection. "run_test" function contains code. 




### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Boxes were flashing inspite of averaging the heatmaps. A few false positives were found which requires a better choice of parameters. Deep Learning can be an efficient method which needs further research. The pipeline is likely to fail in situations of snow since a darkly colored car might result in an incorrect detection.

