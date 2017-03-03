# ytpub@yahoo.com
# Finding Lanes project
# 3/2/2017

import math

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML

def region_of_interest(img, vertices):
    
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def process_image(image):
	# Read in and grayscale the image
	gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)

	# Define a kernel size and apply Gaussian smoothing
	kernel_size = 5
	blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)

	# Define our parameters for Canny and apply
	low_threshold = 50
	high_threshold = 150
	edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

	# Next we'll create a masked edges image using cv2.fillPoly()
	mask = np.zeros_like(edges)   
	ignore_mask_color =  255 
 
	# This time we are defining a four sided polygon to mask
	imshape = image.shape
	x_bottomLeft = 135
	y_bottomLeft = imshape[0]
	x_topLeft = 430
	y_topLeft = 330
	x_topRight = imshape[1]-415
	y_topRight = 330
	x_bottomRight = imshape[1]-75
	y_bottomRight = imshape[0]
	
	vertices = np.array([[(x_bottomLeft,y_bottomLeft),(x_topLeft, y_topLeft), (x_topRight, y_topRight), (x_bottomRight, y_bottomRight)]], dtype=np.int32)
	masked_edges = region_of_interest(edges, vertices)
	
	# Define the Hough transform parameters
	# Make a blank the same size as our image to draw on
	rho = 1 # distance resolution in pixels of the Hough grid
	theta = np.pi/180 # angular resolution in radians of the Hough grid
	threshold = 20     # minimum number of votes (intersections in Hough grid cell)
	min_line_length = 5 #minimum number of pixels making up a line
	max_line_gap = 15    # maximum gap in pixels between connectable line segments
	line_image = np.copy(image)*0 # creating a blank to draw lines on

	# Run Hough on edge detected image
	# Output "lines" is an array containing endpoints of detected line segments
	lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]),
				    min_line_length, max_line_gap)

	# Iterate over the output "lines" and draw lines on a blank image
	x_leftLane = []
	y_leftLane = []
	x_rightLane = []
	y_rightLane = []

	slope_left = 0
	slope_right = 0
	i_right = 0
	i_left = 0

	for line in lines:
		for x1,y1,x2,y2 in line:
			if x1 != x2:
				slope = (y2 - y1)/(x2 - x1)
				if slope < 0:
					slope_left += slope
					i_left += 1
				elif slope > 0:
					slope_right += slope
					i_right += 1
	slope_left = float(slope_left/i_left)
	slope_right = float(slope_right/i_right)
	for line in lines:
		for x1,y1,x2,y2 in line:
			cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)
			if (x1 != x2):		
				slope = (y2 - y1)/(x2 - x1)
				if abs(slope - slope_left)/abs(slope_left) < 0.2:
					x_leftLane.append(x1)
					y_leftLane.append(y1)
					x_leftLane.append(x2)
					y_leftLane.append(y2)
				elif abs(slope - slope_right)/abs(slope_right) < 0.2:
					x_rightLane.append(x1)
					y_rightLane.append(y1)
					x_rightLane.append(x2)
					y_rightLane.append(y2)
	# left lane
	coeffs_leftLane = np.polyfit(x_leftLane, y_leftLane, 1)
	a = coeffs_leftLane[0]
	b = coeffs_leftLane[1]
	poly_leftLane = np.poly1d(coeffs_leftLane)
	x_leftLaneBegin = int((y_bottomLeft - b)/a)
	x_leftLaneEnd = int((y_topLeft - b)/a)
	# right lane
	coeffs_rightLane = np.polyfit(x_rightLane, y_rightLane, 1)
	poly_rightLane = np.poly1d(coeffs_rightLane)
	a = coeffs_rightLane[0]
	b = coeffs_rightLane[1]
	x_rightLaneBegin = int((y_topRight - b)/a)
	x_rightLaneEnd = int((y_bottomRight - b)/a)
	for x in range(x_leftLaneBegin, x_leftLaneEnd,1):
		cv2.line(line_image , (x, int(poly_leftLane(x))), (x+1, int(poly_leftLane(x+1))), [255,0,0], 12)
	for x in range(x_rightLaneBegin, x_rightLaneEnd,1):
		cv2.line(line_image , (x, int(poly_rightLane(x))), (x+1, int(poly_rightLane(x+1))), [255,0,0], 12)


	# Create a "color" binary image to combine with line image
	color_edges = np.dstack((edges, edges, edges)) 

	# Draw the lines on the edge image
	lines_edges = cv2.addWeighted(color_edges, 0.8, line_image, 1, 0) 

	# ytpub@yahoo.com
	# Draw the lines on the input image
	lines_image = cv2.addWeighted(image, 0.8, line_image, 1, 0) 
	return lines_image

white_output = 'solidWhiteRight-result.mp4'
clip1 = VideoFileClip("solidWhiteRight.mp4")
white_clip = clip1.fl_image(process_image)
white_clip.write_videofile(white_output, audio=False)

yellow_output = 'solidYellowLeft-result.mp4'
clip2 = VideoFileClip('solidYellowLeft.mp4')
yellow_clip = clip2.fl_image(process_image)
yellow_clip.write_videofile(yellow_output, audio=False)
