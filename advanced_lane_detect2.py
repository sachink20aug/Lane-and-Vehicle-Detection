import glob
import numpy as np
import cv2
import os
from itertools import chain
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from collections import deque
c1=glob.glob("vehicles/GTI_Far/*.png" )
c2=glob.glob("vehicles/GTI_Left/*.png" )
c3=glob.glob("vehicles/GTI_Right/*.png" )
c4=glob.glob("vehicles/GTI_MiddleClose/*.png" )
c5=glob.glob("vehicles/KITTI_extracted/*.png" )
cars=list(chain(c1,c2,c3,c4,c5))


nc1=glob.glob("non-vehicles/GTI/*.png" )
nc2=glob.glob("non-vehicles/Extras/*.png" )
not_cars=list(chain(nc1,nc2))

import numpy as np
import cv2
from skimage.feature import hog

def convert_color(img, conv='RGB2YCrCb'):
    if conv == 'RGB2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if conv == 'BGR2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    if conv == 'RGB2LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)

def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        
        features, hog_image = hog(img, orientations=orient, 
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block),
                                  
                                  transform_sqrt=False, 
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient, 
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block),
                       
                       transform_sqrt=False, 
                       visualise=vis, feature_vector=feature_vec)
        return features

def bin_spatial(img, size=(32, 32)):
    color1 = cv2.resize(img[:,:,0], size).ravel()
    color2 = cv2.resize(img[:,:,1], size).ravel()
    color3 = cv2.resize(img[:,:,2], size).ravel()
    return np.hstack((color1, color2, color3))

def color_hist(img, nbins=32):    #bins_range=(0, 256)
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

def extract_features(imgs, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        file_features = []
        # Read in each one by one
        image = mpimg.imread(file)
        # apply color conversion if other than 'RGB'
        if color_space != 'RGB':
            if color_space == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif color_space == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif color_space == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif color_space == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif color_space == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else: feature_image = np.copy(image)      

        if spatial_feat == True:
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            file_features.append(spatial_features)
        if hist_feat == True:
            # Apply color_hist()
            hist_features = color_hist(feature_image, nbins=32)
            file_features.append(hist_features)
        if hog_feat == True:
        # Call get_hog_features() with vis=False, feature_vec=True
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(get_hog_features(feature_image[:,:,channel], 
                                        orient, pix_per_cell, cell_per_block, 
                                        vis=False, feature_vec=True))
                hog_features = np.ravel(hog_features)        
            else:
                hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            # Append the new feature vector to the features list
            file_features.append(hog_features)
        features.append(np.concatenate(file_features))
    # Return list of feature vectors
    return features


#Training and Testing the whole dataset
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.calibration import CalibratedClassifierCV
color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
spatial_size = (32, 32) # Spatial binning dimensions
hist_bins = 32    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
y_start_stop = [None, None] # Min and max in y to search in slide_window()

car_features = extract_features(cars, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)
notcar_features = extract_features(not_cars, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)

# Create an array stack of feature vectors
X = np.vstack((car_features, notcar_features)).astype(np.float64)

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))
X_scaler = StandardScaler().fit(X)
X_scaled = X_scaler.transform(X)

# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 10000)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=rand_state)
print('Training data size: {}'.format(X_train.shape[0]))
print('Test data size: {}'.format(X_test.shape[0]))

X_train,y_train=shuffle(X_train,y_train,random_state=2222)

print('Using:',orient,'orientations',pix_per_cell,
    'pixels per cell and', cell_per_block,'cells per block')
print('Feature vector length:', len(X_train[0]))

svc = LinearSVC(C=1.0)

# Check the training time for the SVC
t=time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC...')
# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
# Check the prediction time for a single sample
t=time.time()

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg



# Here is your draw_boxes function from the previous exercise
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy
    

def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap


def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img
    

def find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins):
    
    draw_img = np.copy(img)
    img = img.astype(np.float32)/255
    car_box_list=[]
    img_tosearch = img[ystart:ystop,:,:]
    ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
        

    # Define blocks and steps as above
    nxblocks = (ctrans_tosearch.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ctrans_tosearch.shape[0] // pix_per_cell) - cell_per_block + 1 
    
    
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_window=(window // pix_per_cell) 
    x_windows = ctrans_tosearch.shape[1]//window
    y_windows = ctrans_tosearch.shape[0]//window
    ch1=ctrans_tosearch[:,:,0]
    ch2=ctrans_tosearch[:,:,1]
    ch3=ctrans_tosearch[:,:,2]
    
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    step_per_win=cells_per_window//cells_per_step
    nxsteps = (x_windows-1)*step_per_win
    nysteps = (y_windows-1)*step_per_win
    
    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
    
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
          
            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))    
            #test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))    
            test_prediction = svc.predict(test_features)
            
            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                #cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),(0,0,255),6) 
                car_box_list.append(((xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart)))
    return car_box_list
    
    

from moviepy.editor import VideoFileClip
from IPython.display import HTML
from scipy.ndimage.measurements import label

    

heatmap_list=[]

import numpy as np
import cv2
import glob
import matplotlib
import matplotlib.pyplot as plt

import pickle
def sobel_func(undist):
        gray=cv2.cvtColor(undist,cv2.COLOR_RGB2GRAY)
        
        sobelx=cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=3)
        abs_sobelx=np.absolute(sobelx)
        scaled_sobel=np.uint8(255*abs_sobelx/np.max(abs_sobelx))
        thresh_min=20
        thresh_max=235
        scaled_binary=np.zeros_like(scaled_sobel)
        scaled_binary[(scaled_sobel>=thresh_min)&(scaled_sobel<=thresh_max)]=1
        return scaled_binary
    
def select_yellow_mask(hls):
    lower = np.array([20,55,95])
    upper = np.array([25,210,255])
    return cv2.inRange(hls, lower, upper) //255
    
def select_white_mask(hls):
    lower = np.array([0,  180, 0])
    upper = np.array([255,255, 255])
    return cv2.inRange(hls, lower, upper) //255
    
    
def comb_binary(img):
    
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    yellow_mask = select_yellow_mask(hls)
    white_mask = select_white_mask(hls)
    
    yw_mask = cv2.bitwise_or(yellow_mask, white_mask)

    
    sobel_mask = sobel_func(img)

    
    final_mask = cv2.bitwise_or(sobel_mask, yw_mask)    
    
    return final_mask


def color_gradient(img,mtx,dist):
        undist = cv2.undistort(img, mtx, dist, None, mtx)
        
        
        
   
        
        
        
       
        final_mask=comb_binary(undist)
        
        
        
        
        
        
        
        return final_mask,undist
        
def corners_warp(img, nx, ny, mtx, dist):
    # Use the OpenCV undistort() function to remove distortion
        offset=100
        img_size = (img.shape[1], img.shape[0])
        # For source points I'm grabbing the outer four detected corners
        #src = np.float32([[210, 720], [1150,720], [580,450],[750,450]])
        # For destination points, I'm arbitrarily choosing some points to be
        # a nice fit for displaying our warped result 
        # again, not exact, but close enough for our purposes
        #dst = np.float32([[320, 720], [1000,720],   [320,2], [1000,2]])
        # Given src and dst points, calculate the perspective transform matrix
        # For source points I'm grabbing the outer four detected corners
        src = np.float32([[545, 460],
                    [735, 460],
                    [1280, 700],
                    [0, 700]])

        dst = np.float32([[0, 0],
                     [1280, 0],
                     [1280, 720],
                     [0, 720]])
# For destination points, I'm arbitrarily choosing some points to be
# a nice fit for displaying our warped result
        
        M = cv2.getPerspectiveTransform(src, dst)
        Inv_M = cv2.getPerspectiveTransform(dst, src)
        # Warp the image using OpenCV warpPerspective()
        warped = cv2.warpPerspective(img, M, img_size,cv2.INTER_LINEAR)
      
    # Return the resulting image and matrix
        return warped, M,Inv_M
    
def find_lanes_first_im(warped):
    histogram=np.sum(warped[warped.shape[0]//2:,:],axis=0)
    out_img = np.dstack((warped, warped, warped))*255
# Find the peak of the left and right halves of the histogram
# These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[0:midpoint])
    rightx_base = np.argmax(histogram[midpoint:1280]) +midpoint
    nwindows = 9
# Set height of windows
    window_height = np.int(warped.shape[0]//nwindows)
# Identify the x and y positions of all nonzero pixels in the image
    nonzero = warped.nonzero()    
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
     # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
# Set the width of the windows +/- margin
    margin = 80
# Set minimum number of pixels found to recenter window
    minpix = 60
# Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    for window in range(nwindows):
        win_y_low = warped.shape[0] - (window+1)*window_height
        win_y_high = warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
    # Draw the windows on the visualization image
        cv2.rectangle(warped,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),
        (0,255,0), 2) 
        cv2.rectangle(warped,(win_xright_low,win_y_low),(win_xright_high,win_y_high),
        (0,255,0), 2)     
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        
    # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
    # Append these indices to the lists
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
        
# Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 

# Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    
    ploty = np.linspace(0, warped.shape[0]-1,warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    
    return left_fit,right_fit
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)

def find_lanes_remaining(binary_warped,left_fit,right_fit):
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 50
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + 
    left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + 
    left_fit[1]*nonzeroy + left_fit[2] + margin))) 

    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + 
    right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + 
    right_fit[1]*nonzeroy + right_fit[2] + margin)))  

# Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
# Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    window_img = np.zeros_like(out_img)
# Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    
# Generate a polygon to illustrate the search window area
# And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, 
                              ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, 
                              ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

# Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    
    return left_fit,right_fit
    
    
def radius_curv(img,left_fit,right_fit,warped):
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    # Define y-value where we want radius of curvature
# I'll choose the maximum y-value, corresponding to the bottom of the image
    ploty=np.linspace(0,warped.shape[0]-1,warped.shape[0])
    y_eval = np.max(ploty)
   
    #Equation of lines:
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
# Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)
    
# Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad =((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    avg_radius=0.5*(left_curverad +right_curverad) 
    y_eval_new=y_eval*ym_per_pix
    left_measure = left_fit[0]*y_eval**2 + left_fit[1]*y_eval + left_fit[2]
    right_measure = right_fit[0]*y_eval**2 + right_fit[1]*y_eval + right_fit[2]
    center_pos=((left_measure+right_measure)/2-0.5*img.shape[1])*xm_per_pix
    # Now our radius of curvature is in meters
    print(left_curverad, 'm', right_curverad, 'm',center_pos)
    return left_curverad,right_curverad,avg_radius,center_pos
    
def draw_lanes_color(left_fit,right_fit,warped,Inv_M,undist,avg_radius,center_pos):
    warped_empty = np.zeros_like(warped).astype(np.uint8)
    warped_color = np.dstack((warped_empty, warped_empty, warped_empty))
    ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0] )
    #Equation of lines in world space
    left_fitx = left_fit[0]*ploty**2+left_fit[1]*ploty+left_fit[2]
    right_fitx = right_fit[0]*ploty**2+right_fit[1]*ploty+right_fit[2] 
  
    left_line_window = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    right_line_window = np.array([np.flipud(np.transpose(np.vstack([right_fitx, 
                              ploty])))])
    
    all_pts = np.hstack((left_line_window, right_line_window))
    cv2.fillPoly(warped_color, np.int_([all_pts]), (0,255, 0))
    warp_orig = cv2.warpPerspective(warped_color, Inv_M, (warped_color.shape[1], warped_color.shape[0])) 
    result = cv2.addWeighted(undist, 1, warp_orig, 0.3, 0)
    
    cv2.putText(result,'Radius of curvature=%f m' %avg_radius,(65,40),cv2.FONT_HERSHEY_SIMPLEX,0.75,(255,255,255),2)
    cv2.putText(result,'Vehicle pos w.r.t center=%f m' %center_pos,(720,40),cv2.FONT_HERSHEY_SIMPLEX,0.75,(255,255,255),2)
    
    return result

def calib_camera():
    global n_cols
    global n_rows
    n_cols=9
    n_rows=6
    objp = np.zeros((n_rows*n_cols,3), np.float32)
    objp[:,:2] = np.mgrid[0:n_cols,0:n_rows].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

# Make a list of calibration images
    calib_images = glob.glob('camera_cal/calibration*.jpg')

# Step through the list and search for chessboard corners
    for fname in calib_images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (n_cols,n_rows),None)
    
    # If found, add object points, image points
        if ret == True:
        
            objpoints.append(objp)
            imgpoints.append(corners)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (9,6), corners, ret)
    
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    
    
    return mtx,dist

global mtx,dist
mtx,dist=calib_camera()

def run_test(image):
	global left_fit
	global right_fit  
	global cnt
    
	scale_list = [0.75,1,1.25,1.5]
	ystart = 400
	ystop = 650
	car_bb = []
	blank_img = np.zeros_like(image[:,:,0]).astype(np.float)
    
	
	for scale in scale_list:
		car_bb.extend(find_cars(image, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins =32 ))
	
	heat = add_heat(blank_img, car_bb)

	heat = apply_threshold(heat,1)
	
	heatmap_list.append(heat)
	
	heatmap = np.zeros_like(heat)
    
	if(len(heatmap_list)>=15):
		heatmap_list.pop(0)
        
	for map in heatmap_list:
		heatmap += np.array(map)
		
    

	labels = label(heat)

    
    
	combined_binary,undist=color_gradient(image,mtx,dist)
   
    
	warped,_,Inv_M=corners_warp(combined_binary,n_cols,n_rows,mtx,dist)
	if(cnt==0):
		left_fit,right_fit=find_lanes_first_im(warped)
		_,_,avg_radius,center_pos=radius_curv(image,left_fit,right_fit,warped)
		result=draw_lanes_color(left_fit,right_fit,warped,Inv_M,undist,avg_radius,center_pos)
		draw_img=draw_labeled_bboxes(result,labels)
	else:
		left_fit,right_fit=find_lanes_remaining(warped,left_fit,right_fit)
		_,_,avg_radius,center_pos=radius_curv(image,left_fit,right_fit,warped)
		result=draw_lanes_color(left_fit,right_fit,warped,Inv_M,undist,avg_radius,center_pos)
		draw_img=draw_labeled_bboxes(result,labels)
	cnt=cnt+1
	return draw_img
    


#Run a video to check performance:
from moviepy.editor import VideoFileClip
from IPython.display import HTML


output = 'project_video_output.mp4'
cnt=0
clip1 = VideoFileClip("project_video.mp4")
result = clip1.fl_image(run_test)
result.write_videofile(output, audio=False)

    
        
        
        
        