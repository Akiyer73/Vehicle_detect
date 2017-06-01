##Writeup Template
###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

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
[Car Image]: ./output_images/car.jpg
[Not-Car Image]: ./output_images/notcar.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the first code cell of the IPython notebook (or in lines # through # of the file called `vehicle_detect_29_may.ipynb`).  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:
There are five folders of Vehicle images and these were gathered in a matrice named cars. The code is in the fourth cell of the ipynb.
There are two folders of non-Vehicle images and these were gathered in a matrix named non-cars using the glob function. The code is in the fifth cell of the ipynb notebook.
There are 8792 cars and 8968 non car images.

[Car Image]:[./output_images/car.jpg]
[Not-Car Image]:[./output_images/notcar.jpg]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and chose the ones that gave correct bboxes.
The final parameters chosen were as below

orient = 9
pix_per_cell = 8
cell_per_block = 8
hog_channel = 0 

I used spatial, hist and hog features to identify the cars.
features.append(np.concatenate((spatial_features, hist_features, hog_features)))

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using the cars and notcars images listed earlier using the glob functions.
I split the cars and notcars to train and test sets.
Then using svc.fit, we trained a svc model.


###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I chose an image area of ystart = 400 ystop = 656, to search for cars. Thereafter using the find cars function, where we search in windows of 64 to find the cars.

[bbox1 for test1 image]:[./output_images/bbox0.jpg]
[bbox1 for test2 image]:[./output_images/bbox1.jpg]
[bbox1 for test3 image]:[./output_images/bbox2.jpg]
[bbox1 for test4 image]:[./output_images/bbox3.jpg]
[bbox1 for test5 image]:[./output_images/bbox4.jpg]
[bbox1 for test6 image]:[./output_images/bbox5.jpg]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result. Applying heatmap to the images gave us the final boxes.  Here are some example images:

[for test1 image]:[./output_images/heatmap0.jpg]
[for test2 image]:[./output_images/heatmap1.jpg]
[for test3 image]:[./output_images/heatmap2.jpg]
[for test4 image]:[./output_images/heatmap3.jpg]
[for test5 image]:[./output_images/heatmap4.jpg]
[for test6 image]:[./output_images/heatmap5.jpg]

[for test1 image]:[./output_images/carbox0.jpg]
[for test2 image]:[./output_images/carbox1.jpg]
[for test3 image]:[./output_images/carbox2.jpg]
[for test4 image]:[./output_images/carbox3.jpg]
[for test5 image]:[./output_images/carbox4.jpg]
[for test6 image]:[./output_images/carbox5.jpg]

---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./white.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions. 
Thereafter I used deque command to chck that the window appeared in other frames too.
I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![Raw heatmap][./output_images/V_heatmap724.jpg]
![Raw heatmap][./output_images/V_heatmap725.jpg]
![Raw heatmap][./output_images/V_heatmap726.jpg]
![Raw heatmap][./output_images/V_heatmap727.jpg]
![Raw heatmap][./output_images/V_heatmap728.jpg]
![Raw heatmap][./output_images/V_heatmap729.jpg]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![avg heatmap][./output_images/V_avg_heatmap724.jpg]
![avg heatmap][./output_images/V_avg_heatmap725.jpg]
![avg heatmap][./output_images/V_avg_heatmap726.jpg]
![avg heatmap][./output_images/V_avg_heatmap727.jpg]
![avg heatmap][./output_images/V_avg_heatmap728.jpg]
![avg heatmap][./output_images/V_avg_heatmap729.jpg]


### Here the resulting bounding boxes are drawn onto the last frame in the series
![avg bbox][./output_images/V_carbox724.jpg]
![avg bbox][./output_images/V_carbox725.jpg]
![avg bbox][./output_images/V_carbox726.jpg]
![avg bbox][./output_images/V_carbox727.jpg]
![avg bbox][./output_images/V_carbox728.jpg]
![avg bbox][./output_images/V_carbox729.jpg]

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?
As we increase the number of samples the number of false detections increases. We need to probably increase the non vehicles images.

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

The technique used has SVM as a classifier. Tuning of classifier, increasing non vehicles databse will probably improve the performance in the future.