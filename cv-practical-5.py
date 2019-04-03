import os
import cv2
from keras.models import Sequential
#from keras.layers.core import Dense, Activation, Flatten
#from keras.layers.convolutional import Convolution2D, MaxPooling2D

def loadImage(filename):
    # Load image with flag "Greyscale"
    image = cv2.imread(filename,0)

    # Get height and witdh, but skip channels
    height, width = image.shape[:2]

    # Assign smaller value to crop_dim
    if height < width:
      crop_dim = height
    else:
      crop_dim = width
    
    # Determine centerpoints for width and height
    height_middle = height/2
    width_middle = width/2

    # Determine crop size in both directions away from center
    crop_dim_half = crop_dim/2

    # Determine crop positions
    height_lower_boundary = height_middle - crop_dim_half
    height_upper_boundary = height_middle + crop_dim_half
    width_lower_boundary = width_middle - crop_dim_half
    width_upper_boundary = width_middle + crop_dim_half

    # Crop image to lower dimension size from center point
    cropped_image = image[int(height_lower_boundary):int(height_upper_boundary), int(width_lower_boundary):int(width_upper_boundary)]

    # Rezize to 48x48
    resized_image = cv2.resize(cropped_image, (48, 48))

    # Normalize image by dividing with 255 to make all values between 0 and 1
    normalized_image = resized_image/255.0
    
    return (normalized_image)

def getDatasetFilenames(setName):
    # Open set file and save all filenames
    with open(setName) as file:
        set_files = file.readlines()

    # Remove all endline characters from filenames
    set_files = [x.strip() for x in set_files]

    return set_files

def getDatasetLabels(file_list):
    label_list = []
    
    for file in file_list:
        label_list.append(file.rpartition('_')[0])

    return label_list

def getImagesFromFilenames(filename_list):
    image_list = []
    
    for filename in filename_list:
        image_list.append(loadImage("data\\images\\" + filename))

    return image_list

## Part 1: Split filenames into training and test sets

training_set_filenames = getDatasetFilenames("data\\image-splits\\train.txt")
testing_set_filenames = getDatasetFilenames("data\\image-splits\\test.txt")
training_set_labels = getDatasetLabels(training_set_filenames)
testing_set_labels = getDatasetLabels(testing_set_filenames)

## TODO: Part 2: Load and preprocess images

training_set_images = getImagesFromFilenames(training_set_filenames)
testing_set_images = getImagesFromFilenames(testing_set_filenames)

## TODO: one-hot encode label lists

## TODO: Part 3: Construct model


#model = Sequential()


## TODO: Part 4: Train model



## TODO: Part 5: Print results
