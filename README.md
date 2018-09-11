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
Code for this part of the README can be found in the section of SDC_Term1_P4.py titled "HOG FEATURES."

As with other projects in this program that involve machine learning, data processing and classifier training proved to be the critical portion of the exercise. I began by extracting HOG features alone, without making use of the color histogram and spatial binning techniques discussed in the lecture. The section below summarizes the final parameters of my HOG features extraction, with some rationale.

* Colorspace - YUV
  * Given my experience in previous projects throughout the course, where shadows and changes in the color of the road (asphalt vs concrete) reduced the effectiveness of pipelines built solely on the RGB colorspace, I assumed that RGB just wasn't going to cut it. Instead, I'd need to use a colorspace that separates out the brightness or saturation components of images from the color components (see work done with HSV colorspace in [project 1](https://github.com/jjw2/SDC_Term1_P1) and [project 4](https://github.com/jjw2/SDC_Term1_P4)). I originally used the HSV colorspace, but found that the YUV colorspace seemed to perform better in the dark regions of the video, where tress cast shadows onto the road.

* HOG Pixels per Cell - 16 - and Number of Cells - 2
  * Since images were 64 by 64 pixels, there were only a small number of viable options for pixels per cell and number of cells without reshaping all of the images, which is something I didn't want to experiment with (ex: 8 pixels per cell with 2, 4, or 8 cells per block; 16 pixels per cell with 2 or 4 cells per block; 32 pixels per cell with 2 cells per block). In general, having a smaller number of pixels per cell, such as 8, seemed to result in better levels of classification, but were far too slow when processing the actual video (one of the rubric requirements was that the pipeline could process "several frames per second"). Using 32 pixels per cell allowed for very fast processing in the video, but didn't seem to allow for enough cells to properly classify images. Therefore, I settled on 16 pixels per cell. As far as number of cells, I experimented with different options but ultimately stuck with what was used in the coursework - 2 cells per block.

* HOG Orientations
  * It was fairly simple to test out various HOG orientations, so given that the lecture material stated that 8-12 orientations was standard, I tested that range. It wasn't entirely that simple, since I was testing variations of orientation, pixels per cell, cells per block, and colorspaces at the same time, but across the board, 11 orientations seemed to perform the best.

At one point, I attempted to extract histogram of color features as well as HOG features, thinking that this would improve classification accuracy, but it actually ended up creating more false failures, which made later heatmap filtering much more challenging. I attempted to use RGB, HSV, and YUV colorspaces - including variations of select channels of the HSV and YUV colorspaces - but neither seemed to improve performance over the use of just HOG features. Additionally, adding a second set of features increased pipeline runtime. Ultimately, it just didn't make sense to continue to use histogram of color features, and
I decided to drop this from the pipeline.

One important thing to note: especially when I was experimenting with colorspaces, test accuracy measured after training the classifier didn't seem to be the best indicator for performance on the actual video, at least across some of the variations of classifiers I tested. In many cases, I'd achieve a high level of test accuracy, but would have a huge number of false failures on the actual video. This seems to point towards better preproccessing of the training data, perhaps. The tips for the project recommended filtering the data by hand, or using Udacity's supplemental labeled data, and the use of a neural network was another option, but ultimately, I was able to achieve the level of performance I wanted without having to take these steps.

Finally, I provided the ability to save the extracted feature vectors so that feature extraction wouldn't have to occur each time the code was run.

The figures below show examples of the HOG features extraction for vehicle and non-vehicle images.

![alt text][image1]

![alt text][image2]

## 2. Classifier Training
Code for this part of the README can be found in the section of SDC_Term1_P4.py titled "CLASSIFIER TRAINING."

Once HOG features were extracted, I trained a LinearSVC classifier from the sklearn module. With the parameters I've noted above, training took under a minute, and an accuracy if >98% was achieved. Obviously, the accuracy varies each time the classifier is run, since I used the train test split function from sklearn seeded with a random state. I provided the ability to save off models and reload them if desired.

Interestingly, I found that the use of a scaler function from the sklearn module (preprocessing/StandardScaler) reduced the accuracy of the classifier if only HOG features were extracted. I understand that a scaler may be necessary when combining different features (such as HOG and color histogram features) because otherwise one set of features could hold more "weight" during training, but leaving out the scaler when using only HOG features definitely reduced false positives when running the video.


## 3. Sliding Window Search
Code for this part of the README can be found in the section of SDC_Term1_P4.py titled "FINDING CARS IN A SINGLE FRAME."

I based my sliding window search on the function provided in the Udacity classroom sessions, but made several tweaks. I provided the ability to pass the desired step size as an argument, as well as a margin to exclude a given number of pixels on the left and right side of the image from a given sliding window search (more on this below).

I chose the scale and spacing of my sliding windows to accommodate vehicles at different distances away in the video, generally through trial and error on various images captured from the video. In general, the closer the vehicle, the larger the window and the greater the spacing. At the same time, I had to balance processing speed, as the more windows were searched, the slower the pipeline would operate. Ultimately, I settled on 4 sets of windows of various scales. The figures below illustrate these different scales on some sample images.

Below are two sample images on which the sliding window search will be demonstrated:

![alt text][image3]

A scale of 1.1, shown above, was used to pick out vehicles furthest away. Obviously, vehicles get smaller in the image the further away they are, thus requiring smaller boxes and also smaller spacing. Additionally, while for other spacing I used only 2 rows of windows, here I used 3, as it was difficult to place 2 rows appropriately to consistently pick out cars in the distance. I also used the margin feature to search only a portion of the width of the image, since the road doesn't span the full width of the image at the distances beings searched. This method prevented a large number of false positives from occurring at the edges of the image. In cases were road curvature was much tighter, this method may not be ideal, but in this case, it was effective.

![alt text][image4]
![alt text][image5]

Larger scales were used to classify vehicles closer to the camera, as shown below:

![alt text][image6]
![alt text][image7]

![alt text][image8]
![alt text][image9]

![alt text][image10]
![alt text][image11]

Finally, the images below show all of the sliding windows together, and all of the resulting "hits" for the sample images.

![alt text][image12]
![alt text][image13]


In general, the most challenging aspect of developing the sliding window pipeline was determining an appropriate balance between: 1) the number of windows searched and their placement, 2) the likelihood of false positives, and 3) the speed of the pipeline. Given that the classifier is not 100% accurate, the larger the number of windows searched, the larger the potential for false positives, and the slower the pipeline will run. If the classifer were 100% accurate and processing time wasn't an issue, obviously, the easiest thing to do would be to blanket the entire image with windows of various sizes. In this case, it took a fair amount of time to find the right balance, and it was a process that also involved filtering out false positives over time, which will be discussed in the vehicle tracking section below.

## 4. Heatmapping, Thresholding, and Labelling
Code for this part of the README can be found in the section of SDC_Term1_P4.py titled "HEATMAPPING AND THRESHOLDING."

The output of the sliding window function (find_cars) is a set of bounding boxes that were determined to contain vehicles or sections of vehicles (but which may also include false positives). The next step in the process was to consolidate all of the detections across the range of sliding windows into one bounding box. This was done by using a heatmapping, thresholding, and labelling approach, as suggested in the Udacity classroom sessions.

To generate a heatmap for a single image, the value of any area of a blank image within a single bounding box was incremented; therefore, any areas that are contained within multiple bounding boxes receive higher values. This initial heatmap should then contain larger values in areas where cars reside, and ideally 0 in areas where cars don't reside. Of course, there may be false positives in the image, so a "threshold" is applied, whereby any value of the heatmap that is below a given threshold is set to 0. The goal here is to filter out false positives, under the assumption that there will be less "heat" in the areas of false positives as compared to areas that truly contain cars. This isn't always the case unfortunately, as will be discussed in the Vehicle Tracking section.

An example of a heatmap is shown below (technically, this is a grayscale image, but it looks better in color):
![alt text][image14]

Once such a heatmap is generated, the SciPy "label" function is used in order to isolate separate heatmapped areas. Once individual heatmapped areas are isolated, bounding boxes can be drawn around them. An example is shown below.

![alt text][image15]

## 5. Vehicle Tracking
Code for this part of the README can be found in the section of SDC_Term1_P4.py titled "VEHICLE TRACKING."

Finding images in a single frame was, of course, only one aspect of solving this problem. In order to track vehicles over time as well as filter out false positives I created the VehicleTracker class. This class effectively applies heatmapping and thresholding across frames of the video. It does so by storing a history of the bounding boxes for the last N frames of the video, generating a heat map that spans those N frames, and then applying a threshold. This effectively filters out false failures that may exist in some frames of the video.

## 6. Video

See the final output video for results - project_video_output.mp4

The results are not perfect, and possible improvements are discussed below.


## 7. Discussion

Here, several challenges and potential improvements are discussed.

**Classification**
Classification was a source of frustration during this project. I didn't find the use of Linear SVM classifier using HOG features with or without color histograms to be incredibly robust, and I'm relatively certain that the classification process could be improved through the use of neural networks and machine learning. Now, this might require that a GPU is available to execute the model once it's trained in order to process enough frames (depending on the complexity of the network of course), but this shouldn't be an issue.

**Training Data**
It could very well be that the frustration I experienced with the SVM classifier could have been resolved with additional data (and additional data would be an asset for a neural network approach as well). While ultimately I decided not to pull in additional data (found [here](https://github.com/udacity/self-driving-car/tree/master/annotations), for example) this may improve performance.

**Vehicle Centroid and Bounding Boxes**
Bounding boxes in the video change rapidly, and in some cases, multiple bounding boxes appear very close to each other. This is due to the admittedly unsophisticated technique that I used to simply accrue "heat" over time. A better approach for tracking vehicles over time could resolve this issue. For example, one could attempt to track vehicle centroids and bounding boxes over time, and use filters and other techniques to ensure that they don't change rapidly. For example, it's unreasonable to expect that the bounding box for a vehicle could reduce in size by half in one frame (though it's possible and does occur in what I've implemented), and one could use that knowledge to calculate more stable centroids and bounding boxes that don't change so rapidly over time.

**Tuning and Applicability**
Admittedly, it would be tough to argue that the pipeline is not *tuned* for this video. I spent quite a bit of time determining appropriate bounding box sizes, densities and placements, as well as heatmapping parameters. In many cases, I could tune the pipeline to perform excellently in one section of the video, but it would then fail in another section. Therefore, the final result ended up being a bit of a balancing act to ensure good behavior across the whole video. I wouldn't be surprised if the pipeline didn't perform as well on a different video, especially one with different environmental conditions (quality of daylight, etc). 

[//]: # (Image References)

[image1]: ./md_imgs/hog_veh.png "img"
[image2]: ./md_imgs/hog_nonveh.png "img"
[image3]: ./md_imgs/sample_imgs.png "img"
[image4]: ./md_imgs/scale_1p1.png "img"
[image5]: ./md_imgs/scale_1p1_hit.png "img"
[image6]: ./md_imgs/scale_1p5.png "img"
[image7]: ./md_imgs/scale_1p5_hit.png "img"
[image8]: ./md_imgs/scale_1p8.png "img"
[image9]: ./md_imgs/scale_1p8_hit.png "img"
[image10]: ./md_imgs/scale_2p2.png "img"
[image11]: ./md_imgs/scale_2p2_hit.png "img"
[image12]: ./md_imgs/all_boxes.png "img"
[image13]: ./md_imgs/all_hits.png "img"
[image14]: ./md_imgs/heatmap.png "img"
[image15]: ./md_imgs/labels.png "img"
