#**Finding Lane Lines on the Road** 

##Writeup

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: grayscale.jpg "Grayscale"

---

### Reflection

###1. My pipeline consisted of 5 steps. First, I converted the images to grayscale, then I applied Canny edge detection, then a polygonal mask to delimit theregion of interest, then the Hough trasnform, which returns a bunch of lines.

In order to draw a single line on the left and right lanes, I collected the slopes and classified the slopes into left lane or right lane according to the direction of the slope, then I averaged each of the right and left slopes and then used these slopes to classify the lines as left or right. Then I extended the left and right lines each into one solid left line and one solid right line, and colored it red.

Here are the static photo results:

![Solid White Curve][solidWhiteCurve-result.jpg]
![Solid White Right][solidWhiteRight-result.jpg]
![Solid Yellow Curve][solidYellowCurve-result.jpg]
![Solid WYellow Curve 2][solidYellowCurve2-result.jpg]
![Solid Yellow Left][solidYellowLeft-result.jpg]
![White Car Lane Switch][whiteCarLaneSwitch-result.jpg]

###2. Potential shortcomings with your current pipeline:


The lane finding I have works but is not stable. It crashes in the extra challenge video.  

Another shortcoming is my code is too long.


###3. Possible improvements to the pipeline

A possible improvement would be to discard lines which are neither left nor right (i.e. parasites) from the list of lines.

Another potential improvement could be to modularize the code.
