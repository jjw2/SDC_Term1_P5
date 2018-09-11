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
As with other projects in this program that involve machine learning, data processing and classifier training proved to be the critical portion of the exercise. I began by extracting HOG features alone, without making use of the color histogram and spatial binning techniques discussed in the lecture. The section below summarizes the final parameters of my HOG features extraction, with some rationale. The code can be found in the section titled "HOG FEATURES."

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

XXX The figures below show examples of the HOG features extraction for vehicle and non-vehicle images.


## 2. Classifier Training
Once HOG features were extracted, I trained a LinearSVC classifier from the sklearn module. With the parameters I've noted above, training took under a minute, and an accuracy if >98% was achieved. Obviously, the accuracy varies each time the classifier is run, since I used the train test split function from sklearn seeded with a random state. I provided the ability to save off models and reload them if desired.

Interestingly, I found that the use of a scaler function from the sklearn module (preprocessing/StandardScaler) reduced the accuracy of the classifier if only HOG features were extracted. I understand that a scaler may be necessary when combining different features (such as HOG and color histogram features) because otherwise one set of features could hold more "weight" during training, but leaving out the scaler when using only HOG features definitely reduced false positives when running the video.

The code for classifier training can be found in the section labeled "Classifier Training"





[//]: # (Image References)

[image1]: ./md_imgs/img.png "img"
