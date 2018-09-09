import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
from dask.array.ufunc import floor

########## PLOTTING UTILITIES ##########

def plot_imgs(X, title=[], cols = 2, cmap='brg', h_mult = 2.5):
    
    num_cols = cols
    num_plots = len(X)
    num_rows = int(math.ceil(num_plots/2))
    
    plotNum = 1
    plt.figure(figsize = (12, num_rows*h_mult))
    for i in range(num_plots):
        plt.subplot(num_rows, num_cols, plotNum)
        plt.imshow(X[i], cmap=cmap)
        if(title):
            plt.title(title[i])
        plotNum = plotNum + 1
        
    plt.show()


    
########## IMPORT TRAINING DATA ##########

import glob 

#veh_imgs = glob.glob("data_lesson/vehicles/**/*.jpeg")
#nonveh_imgs = glob.glob("data_lesson/non-vehicles/**/*.jpeg")
veh_imgs = glob.glob("data/vehicles/**/*.png")
nonveh_imgs = glob.glob("data/non-vehicles/**/*.png")

print("Dataset contains", len(veh_imgs), "vehicles and", len(nonveh_imgs), "non-vehicles.")



########## HOG FEATURES ##########

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


# Generate a random index to look at a car image
test_idx = 2;
# Read in the image
image = mpimg.imread(veh_imgs[test_idx])

print("Image size:", image.shape)

gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# Call our function with vis=True to see an image output
features, hog_image = get_hog_features(gray, orient= 9, 
                        pix_per_cell= 8, cell_per_block= 2, 
                        vis=True, feature_vec=False)


# Plot the examples

plot2 = False

if plot2:
    fig = plt.figure()
    plt.subplot(121)
    plt.imshow(image, cmap='gray')
    plt.title('Example Car Image')
    plt.subplot(122)
    plt.imshow(hog_image, cmap='gray')
    plt.title('HOG Visualization')
    plt.show()
    
    
    
    
    
########## COLOR FEATURES ##########

def color_hist(img, nbins=32, bins_range=(0, 1)):
    
    
    image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    
    image = image.astype(np.float32)/255.0
    
    
    # Compute the histogram of the color channels separately
    #channel1_hist = np.histogram(image[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(image[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(image[:,:,2], bins=nbins, range=bins_range)
    
    

    
    # Concatenate the histograms into a single feature vector
    #hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    hist_features = np.concatenate((channel2_hist[0], channel3_hist[0]))
    #hist_features = channel1_hist[0]
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

######### FEATURE EXTRACTION ##########

"""

"""


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



########## FEATURE EXTRACTION ##########

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
import time 
import pickle

# Feature extraction parameters

LOAD = True
VIDEO = True

HOG_COLOR = "YUV"
HOG_ORIENT = 11
HOG_PIX = 16
HOG_CELLS = 2


colorspace = HOG_COLOR # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = HOG_ORIENT
pix_per_cell = HOG_PIX
cell_per_block = HOG_CELLS
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"

print('Using:',orient,'orientations',pix_per_cell,
    'pixels per cell and', cell_per_block,'cells per block')

load_feat = LOAD
feat_filename = "saves/features.pkl"

if load_feat:
    
    X_train, X_test, y_train, y_test = pickle.load(open(feat_filename, 'rb'))
    print("Features loaded from", feat_filename)
    
else:

    print("Extracting features...")
    t=time.time()
    veh_feat = extract_features(veh_imgs, cspace=colorspace, orient=orient, 
                            pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
                            hog_channel=hog_channel)
    nonveh_feat = extract_features(nonveh_imgs, cspace=colorspace, orient=orient, 
                            pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
                            hog_channel=hog_channel)
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


# Fit a per-column scaler on only the training data
# Data was saved before this operation because we want the scaler to be available for use later
X_scaler = StandardScaler().fit(X_train)

# Apply the scaler to X
X_train = X_scaler.transform(X_train)
X_test = X_scaler.transform(X_test)
    
    
print('Feature vector length:', len(X_train[0]))



########## CLASSIFIER TRAINING ##########

# choose to train or load a classifer
load_svc = LOAD
svc_filename = "saves/svc.pkl"

if load_svc:
    
    print("Loading classifier", svc_filename)
    svc = pickle.load(open(svc_filename, "rb" ))

else:
    print("Training SVC classifier...")
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
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))



########## FIND CARS IN SINGLE IMAGE ##########

# First, grab some images out of the video
from moviepy.editor import VideoFileClip

# Times for frames to extract
#times = [15, 20, 25, 30, 35, 40]
times = [24, 24.4, 24.8, 25.2, 25.6, 26, 26.4]
clip1 = VideoFileClip("project_video.mp4")

raw_imgs = []

for time in times:
    raw_imgs.append(clip1.get_frame(time))

plot0 = False
if plot0:
    plot_imgs(raw_imgs, cols=3)


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


image = mpimg.imread('bbox-example-image.jpg')
result = draw_boxes(image, bboxes)

plot1 = False
if plot1:
    plt.imshow(result)
    plt.show()



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


def find_cars(img, params, svc, scaler, orient, pix_per_cell, cell_per_block):
    
    # Normalize the image
    img = img.astype(np.float32)/255.0
    
    
    # List of all the boxes searched
    boxes = []
    
    # Store indices of detected vehicles
    detections = []
    
    for param in params:
        ystart = param[0]
        ystop = param[1]
        scale = param[2]
        cell_step = param[3]
        xmargin = param[4]
        
        #print("ystart", ystart)
        #print("ystop", ystop)
        #print("scale", scale)
        
        if xmargin == 0:
            clr_tosearch = img[ystart:ystop,:,:]
        else:
            clr_tosearch = img[ystart:ystop,xmargin:-xmargin,:]
        ctrans_tosearch = convert_color(clr_tosearch, conv='RGB2YUV')
        #plt.imshow(ctrans_tosearch)
        #plt.show()
        
        #print("Search window shape before scaling:", ctrans_tosearch.shape)
        
        if scale != 1:
            imshape = ctrans_tosearch.shape
            ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
            #clr_tosearch = cv2.resize(clr_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
        
        #print("Search window shape after scaling:", ctrans_tosearch.shape)
        #plt.imshow(ctrans_tosearch)
        #plt.plot()
            
        ch1 = ctrans_tosearch[:,:,0]
        ch2 = ctrans_tosearch[:,:,1]
        ch3 = ctrans_tosearch[:,:,2]
    
        # Define blocks and steps as above
        nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
        nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1 
        
        # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
        window = 64
        nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
        cells_per_step_x = cell_step  
        cells_per_step_y = cell_step
        nxsteps = int(floor((nxblocks - nblocks_per_window) / cells_per_step_x + 1))
        nysteps = int(floor((nyblocks - nblocks_per_window) / cells_per_step_y + 1))
        
        #print("nxsteps:", nxsteps)
        #print("nysteps:", nysteps)
        
        # Compute individual channel HOG features for the entire image
        hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=False)
        hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=False)
        hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=False)
        
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
    
                # Extract the image patch
                #subimg = cv2.resize(clr_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
              
                # Get color features
                #spatial_features = bin_spatial(subimg, size=spatial_size)
                #color_features = color_hist(subimg)
    
                # Scale features and make a prediction
                #test_features = scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))    
                #test_features = scaler.transform(np.hstack((hog_features, color_features)).reshape(1, -1))
                test_features = scaler.transform(hog_features.reshape(1,-1))
                test_prediction = svc.predict(test_features)
                
                xbox_left = np.int((xleft+xmargin)*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                
                boxes.append(((xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart)))
                
                if test_prediction == 1:
                    
                    detections.append(((xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart)))
                
    return detections, boxes


# Some of this was defined above, but redefining here for ease of use/clarity
colorspace = HOG_COLOR
orient = HOG_ORIENT
pix_per_cell = HOG_PIX
cell_per_block = HOG_CELLS
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
# svc used from above
# scaler used from above

#               [ystart, ystop, scale, cell step]


scale_params = [[400, 470, 1.0, 1.5, 100],
                [415, 485, 1.0, 1.5, 100],
                [400, 500, 1.5, 1.25, 50],
                [430, 510, 1.5, 1.25, 50],
                [400, 530, 2.0, 1.0, 25],
                [430, 560, 2.0, 1.0, 25],
                [400, 650, 3.5, 1.0, 0],
                [450, 700, 3.5, 1.0, 0]]


"""
scale_params =  [[400, 480, 1.25, 1.5],  # Scale 1.0 -> search rectangle is 64x64 pixels.
                 [410, 490, 1.25, 1.5],
                 [380, 500, 1.5, 1.5], # Scale 1.5 -> search rectangle is 96x96 pixels.
                 [400, 515, 1.5, 1.5],
                 [380, 520, 1.75, 1.25], # Scale 2.0 -> search rectangle is 128x128 pixels.
                 [415, 545, 1.75, 1.25],
                 [390, 570, 2.0, 1.0], # Scale 2.5 -> search rectangle is 160x160 pixels.
                 [430, 610, 2.0, 1.0]]
"""

all_box_imgs = []
hit_box_imgs = []
hit_boxes=[]
for img in raw_imgs:
    
    hit_box, all_box = find_cars(img, scale_params, svc, X_scaler, orient, pix_per_cell, cell_per_block)
    hit_boxes.append(hit_box)
    all_box_imgs.append(draw_boxes(img, all_box, rainbow=True))
    hit_box_imgs.append(draw_boxes(img, hit_box))
    
plot3 = True
if plot3:
    plot_imgs(all_box_imgs, cols=3, h_mult=3.0)
    plot_imgs(hit_box_imgs, cols=3, h_mult=3.0)





########## HEATMAPPING AND THRESHOLDING ##########

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


plot4 = False
if plot4:   
    plot_imgs(heatmap_imgs, cols=3)
    plot_imgs(label_imgs, cols=3)
    

# Combine above functions into one function
def create_labels(img, rects, thresh):
    heat = np.zeros_like(img[:,:,0]).astype(np.float)
    heat = add_heat(heat, rects)
    heat = heat_threshold(heat, thresh)
    heatmap =  np.clip(heat, 0, 255)
    labels = label(heatmap)
    output = draw_labeled_bboxes(img, labels)
    return output 
    
    
########## CREATE CLASS TO TRACK VEHICLES ##########
"""
The VehicleTracker class below is sloppy in that its using a several functions that are not
a part of the class (i.e.: all the functions above). Ideally, I would make all of these functions 
class methods, but at this point, and given that I developed the above code step-by-step as I 
progressed through the lessons, I'm goign to leave it as is. If this were to be deployed, I would
enforce proper encapsulation.  
"""

class VehicleTracker:
    def __init__(self, classifier, scaler):
        
        # Pass the classifier and scaler as initialization arguments.
        self.classifier = classifier
        self.scaler = scaler
        
        # HOG parameters - duplicated from above
        self.colorspace = HOG_COLOR
        self.orient = HOG_ORIENT
        self.pix_per_cell = HOG_PIX
        self.cell_per_block = HOG_CELLS
        self.hog_channel = "ALL"
        self.scale_params = scale_params # Scale 2.5 -> search rectangle is 160x160 pixels.

                            
        
        # Heat map threshold for an individual image
        self.frame_heat_thresh = 2
        
        # Heat map thershold for heat accumulation acros frames
        self.temporal_heat_thresh = 10

        # Labels from last good run
        self.labels = None
        
        # Size of history buffer
        self.hist_buff_size = 15
        self.rect_hist = [] # this will end up being a list of lists...
        self.hist_scale = 0.85
        


    # Returns bounding boxes for an individual image
    def get_rects(self, img):
        rects, _ = find_cars(img, self.scale_params, self.classifier, self.scaler, self.orient, self.pix_per_cell, self.cell_per_block)
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
            
            """
            plt.figure()
            plt.title("Before thresholding")
            plt.imshow(heatmap)
            plt.show()
            """
            
            heatmap = heat_threshold(heatmap, self.temporal_heat_thresh)
            heatmap = np.clip(heatmap, 0, 255)
        
            """
            plt.figure()
            plt.title("After thresholding")
            plt.imshow(heatmap)
            plt.show()
            """
            
            self.labels = label(heatmap)
        
        if self.labels is not None:
            output = draw_labeled_bboxes(img, self.labels)
        else:
            output = img
            
        return output



proc_video = VIDEO

if proc_video:
    tracker = VehicleTracker(svc, X_scaler)
    
    output_file1 = 'project_video_output.mp4'
    #output_clip1 = clip1.fl_image(lambda image: tracker.proc_img(image)).subclip(24,28)
    output_clip1 = clip1.fl_image(lambda image: tracker.proc_img(image)).subclip(10, 20)
    output_clip1.write_videofile(output_file1, audio=False)


