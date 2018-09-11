# Vehicle Tracking

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a Linear SVM classifier
  * Other methods, such features, such as histograms of color may also be used to supplement the HOG feature vector.
* Implement a sliding-window technique and use the trained classifier to search for vehicles in images.
* Run your pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for detected vehicles.


I will be touching on the important aspects of the [rubric](https://review.udacity.com/#!/rubrics/513/view) in this writeup.


## Setup
Starting materials and instructions can be found [here](https://github.com/udacity/CarND-Vehicle-Detection).


## 1. Feature Extraction
As with other projects in this program that involve machine learning, data processing and classifier training proved to be the critical portion of the exercise. I began by extracting HOG features alone, without making use of the color histogram and spatial binning techniques discussed in the lecture. The section below summarizes the parameters of my HOG features extraction, with some rationale:

* Colorspace - YUV
  * Given my experience in previous projects throughout the course, where shadows and changes in the color of the road (asphalt vs concrete) reduced the effectiveness of pipelines built solely on the RGB colorspace, I assumed that RGB just wasn't going to cut it. Instead, I'd need to use a colorspace that separates out the brightness or saturation components of images from the color components (see work done with HSV colorspace in [project 1](https://github.com/jjw2/SDC_Term1_P1) and [project 4](https://github.com/jjw2/SDC_Term1_P4)). I originally used the HSV colorspace, but found that the YUV colorspace seemed to perform better in the dark regions of the video, where shadows cast by trees along the side of the road


[//]: # (Image References)

[image1]: ./md_imgs/chessboards.png "chessboards"
[image2]: ./md_imgs/hist.png "hist"
[image3]: ./md_imgs/hist_img.png "hist_img"
[image4]: ./md_imgs/lane_curv_ofst.png "lane_curv_ofst"
[image5]: ./md_imgs/projected_lane.png "projected_lane"
[image6]: ./md_imgs/sliding_window.png "sliding_window"
[image7]: ./md_imgs/srch_around_poly.png "srch_around_poly"
[image8]: ./md_imgs/thresholding.png "thresholding"
[image9]: ./md_imgs/thresholding1.png "thresholding1"
[image10]: ./md_imgs/undist_chessboard.png "undist_chessboard"
[image11]: ./md_imgs/undist_road.png "undist_road"
[image12]: ./md_imgs/warp_dest.png "warp_dest"
[image13]: ./md_imgs/warp_imgs.png "warp_imgs"
[image14]: ./md_imgs/warp_src.png "warp_src"
