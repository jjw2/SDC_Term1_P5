import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
from dask.array.ufunc import floor


########## GLOBAL PARAMETERS ##########

# Load features and classifier if true; otherwise, extract features and train 
LOAD = True

# Filename for loading or saving features and classifer
svc_filename = "saves/svc.pkl"
feat_filename = "saves/features.pkl"

# Plot images throughout
PLOT = True

# Process the video if true; False used only for plotting/debugging
VIDEO = True
OUTPUT_VIDEO = 'project_video_output1.mp4'

# HOG parameters
HOG_COLOR = "YUV"
HOG_ORIENT = 11
HOG_PIX = 16
HOG_CELLS = 2
HOG_CHANNEL = "All"

# Sliding window parameters
# [0] -> ystart
# [1] -> ystop
# [2] -> scale
# [3] -> cell spacing
# [4] -> xmargin - narrows x search region by this number of pixels on each side of the image
SCALE_PARAMS = [[390, 470, 1.1, 1.0, 250],
                [400, 480, 1.1, 1.0, 250],
                [415, 500, 1.1, 1.0, 250],
                [390, 500, 1.5, 1.4, 0],
                [420, 530, 1.5, 1.4, 0],
                [390, 540, 1.8, 1.25, 0],
                [410, 570, 1.8, 1.25, 0],
                [420, 600, 2.0, 1.0, 0]]




########################################
########## PLOTTING UTILITIES ##########
########################################

# Genearl function for plotting an array of images.
def plot_imgs(X, title=[], subtitle=[], cols=2, cmap='brg', size=(11,3)):
    
    num_cols = cols
    num_plots = len(X)
    num_rows = int(math.ceil(num_plots/2))
    
    plotNum = 1
    plt.figure(figsize = size)

    for i in range(num_plots):
        plt.subplot(num_rows, num_cols, plotNum)
        plt.imshow(X[i], cmap=cmap)
        if(subtitle):
            plt.title(subtitle[i])
        plotNum = plotNum + 1
    
    if(title):
        plt.suptitle(title)
        
    plt.show()


##########################################
########## IMPORT TRAINING DATA ##########
##########################################

import glob 

veh_imgs = glob.glob("data/vehicles/**/*.png")
nonveh_imgs = glob.glob("data/non-vehicles/**/*.png")

print("Dataset contains", len(veh_imgs), "vehicles and", len(nonveh_imgs), "non-vehicles.")


##################################
########## HOG FEATURES ##########
##################################

from skimage.feature import hog 

# Returns hog features and images
# Adapted from Udacity course work
def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=True,
                     feature_vec=True):
    
    if vis:
        hog_features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block),
                                  block_norm= 'L2-Hys', transform_sqrt=False, 
                                  visualise= True, feature_vector= feature_vec)
    
        return hog_features, hog_image
    else:
        hog_features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block),
                                  block_norm= 'L2-Hys', transform_sqrt=False, 
                                  visualise= False, feature_vector= feature_vec)
        return hog_features


# Pick some images from the dataset for plotting
veh_idx = 0;
nonveh_idx = 10;

# Read in the image
veh_img = mpimg.imread(veh_imgs[veh_idx])
nonveh_img = mpimg.imread(nonveh_imgs[nonveh_idx])

veh_gray = cv2.cvtColor(veh_img, cv2.COLOR_RGB2GRAY)
nonveh_gray = cv2.cvtColor(nonveh_img, cv2.COLOR_RGB2GRAY)

# Call function with vis=True to see an image output
features, veh_hog_img = get_hog_features(veh_gray, orient= 11, 
                        pix_per_cell= 16, cell_per_block= 2, 
                        vis=True, feature_vec=False)

features, nonveh_hog_img = get_hog_features(nonveh_gray, orient= 11, 
                        pix_per_cell= 16, cell_per_block= 2, 
                        vis=True, feature_vec=False)


# Plot the examples
PLOT
if PLOT:
    fig = plt.figure(figsize = (6, 2.5))
    plt.subplot(121)
    plt.imshow(veh_img, cmap='gray')
    plt.title('Example Vehicle Image')
    plt.subplot(122)
    plt.imshow(veh_hog_img, cmap='gray')
    plt.title('HOG Visualization')
    plt.show()
    
    fig = plt.figure(figsize = (6, 2.5))
    plt.subplot(121)
    plt.imshow(nonveh_img, cmap='gray')
    plt.title('Example Non-Vehicle Image')
    plt.subplot(122)
    plt.imshow(nonveh_hog_img, cmap='gray')
    plt.title('HOG Visualization')
    plt.show()
    

#COLOR FEATURES
"""
NOTE: while I experimented with extracting and using color features, I found that this didn't help improve
accuracy during video processing. I've left the code here, but it is unused.
"""

def color_hist(img, nbins=32, bins_range=(0, 1)):
    
    # Convert and scale image
    image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    image = image.astype(np.float32)/255.0
    
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(image[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(image[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(image[:,:,2], bins=nbins, range=bins_range)

    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))

    return hist_features

#######################################
######### FEATURE EXTRACTION ##########
#######################################

def extract_features(imgs, cspace='RGB', orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        # Read in each one by one
        image = mpimg.imread(file)
        
        #print(np.amax(image))
        # apply color conversion if other than 'RGB'
        if cspace != 'RGB':
            if cspace == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif cspace == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif cspace == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif cspace == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif cspace == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
            elif cspace == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        else: feature_image = np.copy(image)      

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
        
        
        #color_features = color_hist(image)
        
        # Append the new feature vector to the features list
        #features.append(np.concatenate((hog_features, color_features)))
        features.append(hog_features)
    # Return list of feature vectors
    return features


########################################
########## FEATURE EXTRACTION ##########
########################################

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
import time 
import pickle



print('Using:', HOG_ORIENT,'orientations', HOG_PIX,
    'pixels per cell and', HOG_CELLS, 'cells per block')

if LOAD:
    
    X_train, X_test, y_train, y_test = pickle.load(open(feat_filename, 'rb'))
    print("Features loaded from", feat_filename)
    
else:

    print("Extracting features...")
    t=time.time()
    veh_feat = extract_features(veh_imgs, cspace=HOG_COLOR, orient=HOG_ORIENT, 
                            pix_per_cell=HOG_PIX, cell_per_block=HOG_CELLS, 
                            hog_channel=HOG_CHANNEL)
    nonveh_feat = extract_features(nonveh_imgs, cspace=HOG_COLOR, orient=HOG_ORIENT, 
                            pix_per_cell=HOG_PIX, cell_per_block=HOG_CELLS, 
                            hog_channel=HOG_CHANNEL)
    t2 = time.time()
    print(round(t2-t, 2), 'seconds to extract features...')
    
    # Create an array stack of feature vectors
    X = np.vstack((veh_feat, nonveh_feat)).astype(np.float64)
    
    # Define the labels vector
    y = np.hstack((np.ones(len(veh_feat)), np.zeros(len(nonveh_feat))))
    
    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=rand_state)
        
        
    pickle.dump([X_train, X_test, y_train, y_test], open(feat_filename, "wb"))
    print("Features saved to", feat_filename)


print('Feature vector length:', len(X_train[0]))


"""
I found that when using only HOG features, applying a scaler resulted in more false positives.
If additional features (such as color histogram features) were added, I'd apply the scalar.
See the README for more discussion.
"""

# Fit a per-column scaler on only the training data
# Data was saved before this operation because we want the scaler to be available for use later
#X_scaler = StandardScaler().fit(X_train)

# Apply the scaler to X
#X_train = X_scaler.transform(X_train)
#X_test = X_scaler.transform(X_test)
    
    

#########################################
########## CLASSIFIER TRAINING ##########
#########################################

# Train or load a classifer

if LOAD:
    print("Loading classifier", svc_filename)
    svc = pickle.load(open(svc_filename, "rb" ))

else:
    print("Training classifier...")
    # Use a linear SVC 
    svc = LinearSVC()
    # Check the training time for the SVC
    t=time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2-t, 2), "seconds to train SVC.")
    
    
    pickle.dump(svc, open(svc_filename, "wb" ))
    print("Classifier saved to", svc_filename)
    

# Check the score of the SVC
print('Test Accuracy of classifier = ', round(svc.score(X_test, y_test), 4))

####################################################
########## FINDING CARS IN A SINGLE FRAME ##########
####################################################

from moviepy.editor import VideoFileClip

# Extract frames from the video for plotting
times = [26, 30]
clip1 = VideoFileClip("project_video.mp4")

raw_imgs = []
for time in times:
    raw_imgs.append(clip1.get_frame(time))

if PLOT:
    plot_imgs(raw_imgs, title="Sample Images")


# Helper function to draw bounding boxes
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6, rainbow=False):
    # Make a copy of the image
    draw_img = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Choose a random color
        if rainbow:
            color = (np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255))
            
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(draw_img, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return draw_img

bboxes = [((275, 572), (380, 510)), ((488, 563), (549, 518)), ((554, 543), (582, 522)), 
          ((601, 555), (646, 522)), ((657, 545), (685, 517)), ((849, 678), (1135, 512))]



# Helper function to conver color spaces
def convert_color(img, conv='RGB2YCrCb'):
    if conv == 'RGB2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if conv == 'BGR2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    if conv == 'RGB2LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
    if conv == 'RGB2YUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    if conv == 'RGB2HSV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)


# Function that implements sliding window technique to classify vehicles in images
def find_cars(img, params, svc, orient, pix_per_cell, cell_per_block, scaler=None):
    
    # Normalize the image
    img = img.astype(np.float32)/255.0
    
    # List of all the boxes searched
    boxes = []
    
    # List of detected vehicles
    detections = []
    
    for param in params:
        ystart = param[0]
        ystop = param[1]
        scale = param[2]
        cell_step = param[3]
        xmargin = param[4]
        
        # Apply a margin to the left and right sides of if the image, if arg is passed
        if xmargin == 0:
            clr_tosearch = img[ystart:ystop,:,:]
        else:
            clr_tosearch = img[ystart:ystop,xmargin:-xmargin,:]
            
        # Convert to YUV space
        ctrans_tosearch = convert_color(clr_tosearch, conv='RGB2YUV')

        
        # Rescale images if scale is not 1
        if scale != 1:
            imshape = ctrans_tosearch.shape
            ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
            #clr_tosearch = cv2.resize(clr_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))

        # Separate channels to extract HOG features
        ch1 = ctrans_tosearch[:,:,0]
        ch2 = ctrans_tosearch[:,:,1]
        ch3 = ctrans_tosearch[:,:,2]
    
        # Define blocks and steps
        nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
        nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1 
        
        # Images are 64x64 pixels
        # Calculate number of sliding windows in x and y directions
        window = 64
        nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
        cells_per_step_x = cell_step  
        cells_per_step_y = cell_step
        nxsteps = int(floor((nxblocks - nblocks_per_window) / cells_per_step_x + 1))
        nysteps = int(floor((nyblocks - nblocks_per_window) / cells_per_step_y + 1))
        
        # Compute individual channel HOG features for the entire image
        # This is done once here befores sliding windows are applied in order to save execution time
        hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=False)
        hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=False)
        hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=False)
        
        # Step through each window and apply the classifier
        for xb in range(nxsteps):
            for yb in range(nysteps):
                ypos = int(yb*cells_per_step_y)
                xpos = int(xb*cells_per_step_x)
                
                # Extract HOG for this patch
                hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
    
                xleft = xpos*pix_per_cell
                ytop = ypos*pix_per_cell
    
                # Uncomment code to use color histogram features
                
                # Extract the image patch
                #subimg = cv2.resize(clr_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
              
                # Get color features
                #spatial_features = bin_spatial(subimg, size=spatial_size)
                #color_features = color_hist(subimg)
    
                # Scale features and make a prediction
                if scaler is not None:
                    #test_features = scaler.transform(np.hstack((hog_features, color_features)).reshape(1, -1))
                    test_features = scaler.transform(hog_features).reshape(1, -1)
                else:
                    test_features = hog_features.reshape(1,-1)
               
                test_prediction = svc.predict(test_features)
                
                xbox_left = np.int((xleft+xmargin)*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                
                boxes.append(((xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart)))
                
                if test_prediction == 1:
                    
                    detections.append(((xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart)))
                
    return detections, boxes


# Apply the find_cars function to our test images
all_box_imgs = []
hit_box_imgs = []
hit_boxes=[]

for img in raw_imgs:
    hit_box, all_box = find_cars(img, SCALE_PARAMS, svc, HOG_ORIENT, HOG_PIX, HOG_CELLS)
    hit_boxes.append(hit_box)
    all_box_imgs.append(draw_boxes(img, all_box, rainbow=True))
    hit_box_imgs.append(draw_boxes(img, hit_box))


if PLOT:
    plot_imgs(all_box_imgs, title="All Boxes", cols=2)
    plot_imgs(hit_box_imgs, title="All Hits", cols=2)




##################################################
########## HEATMAPPING AND THRESHOLDING ##########
##################################################

from scipy.ndimage.measurements import label

def add_heat(heatmap, bbox_list, addval=1.0):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += addval

    # Return updated heatmap
    return heatmap# Iterate through list of bboxes


def heat_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap


def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    output = np.copy(img)
    
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(output, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return output


heatmap_imgs = []
label_imgs = []
for i in range(len(hit_boxes)):
    img = hit_box_imgs[i]
    rects = hit_boxes[i]
    
    heat = np.zeros_like(img[:,:,0]).astype(np.float)
    heat = add_heat(heat, rects)
    heat = heat_threshold(heat, 1)
    heatmap =  np.clip(heat, 0, 255)
    heatmap_imgs.append(heatmap)
    labels = label(heatmap)
    label_imgs.append(draw_labeled_bboxes(raw_imgs[i], labels))


if PLOT:   
    plot_imgs(heatmap_imgs, title="Heatmapped Images", cols=2)
    plot_imgs(label_imgs, title="Labelled Images", cols=2)
    

# Combine above functions into one function
def create_labels(img, rects, thresh):
    heat = np.zeros_like(img[:,:,0]).astype(np.float)
    heat = add_heat(heat, rects)
    heat = heat_threshold(heat, thresh)
    heatmap =  np.clip(heat, 0, 255)
    labels = label(heatmap)
    output = draw_labeled_bboxes(img, labels)
    return output 
    


######################################
########## VEHICLE TRACKING ##########
######################################
"""
The VehicleTracker class below is sloppy in that its using several functions that are not
a part of the class (i.e.: all the functions above). Ideally, I would make all of these functions 
class methods, but at this point, and given that I developed the above code step-by-step as I 
progressed through the lessons, I'm goign to leave it as is. If this were to be deployed, I would
enforce proper encapsulation.  
"""

class VehicleTracker:
    def __init__(self, classifier):
        
        # Pass the classifier and scaler as initialization arguments.
        self.classifier = classifier
        
        # HOG parameters - duplicated from above
        self.colorspace = HOG_COLOR
        self.orient = HOG_ORIENT
        self.pix_per_cell = HOG_PIX
        self.cell_per_block = HOG_CELLS
        self.hog_channel = HOG_CHANNEL
        self.scale_params = SCALE_PARAMS 

        
        # Heat map thershold for heat accumulation acros frames
        self.temporal_heat_thresh = 22

        # Labels from last good run
        self.labels = None
        
        # Size of history buffer
        self.hist_buff_size = 10
        self.rect_hist = [] # this will end up being a list of lists...
        self.hist_scale = 0.95
        


    # Returns bounding boxes for an individual image
    def get_rects(self, img):
        rects, _ = find_cars(img, self.scale_params, self.classifier, self.orient, self.pix_per_cell, self.cell_per_block)
        return rects
    
    def push_rects(self, rects):
        # Add rectangles to buffer
        self.rect_hist.append(rects)
        
        # Throw out first list element if buffer is full
        if(len(self.rect_hist) > self.hist_buff_size):
            self.rect_hist = self.rect_hist[1:]
            

    def accum_heat(self):
        
        heatmap = np.zeros_like(img[:,:,0]).astype(np.float)
        mult = 1
        # Loop through rectangle history and create heat map
        # NOTE re: use of reversed here
        for rects in reversed(self.rect_hist):
            heatmap = add_heat(heatmap, rects, mult)
            mult = mult*self.hist_scale
            
        return heatmap
        
        
    
    def proc_img(self, img):
        # Get rectangles
        frame_rects = self.get_rects(img)
        
        # If we found recangles in this frame, do things
        if (len(frame_rects) > 0):
            self.push_rects(frame_rects)
        
            # Heatmap for accumulating heat across frames
            
            heatmap = self.accum_heat()
        
            heatmap = heat_threshold(heatmap, self.temporal_heat_thresh)
            heatmap = np.clip(heatmap, 0, 255)
        
            self.labels = label(heatmap)
        
        if self.labels is not None:
            output = draw_labeled_bboxes(img, self.labels)
        else:
            output = img
            
        return output



if VIDEO:
    tracker = VehicleTracker(svc)
    output_file1 = OUTPUT_VIDEO
    output_clip1 = clip1.fl_image(lambda image: tracker.proc_img(image))
    output_clip1.write_videofile(output_file1, audio=False)


