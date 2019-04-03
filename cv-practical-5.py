import os
import cv2
from keras.models import Sequential
#from keras.layers.core import Dense, Activation, Flatten
#from keras.layers.convolutional import Convolution2D, MaxPooling2D

def loadImage(filename):
    # [1] Get the file category

    # TODO
    # category = 

    # [2] Load the image in greyscale with opencv.

    # Load image with flag "Greyscale"
    image = cv2.imread(filename,0)

    # [3] Find the dimension that is the smallest between the height and the width and assign it to the crop_drim variable.

    # Get height and witdh, but skip channels
    height, width = image.shape[:2]

    # Assign smaller value to crop_dim
    if height < width:
      crop_dim = height
    else:
      crop_dim = width
      
    #print('h: ' + str(height) + ', w:' + str(width) + ', cd: ' + str(crop_dim))
    
    # [4] Crop the centre of the image based on the crop_dim dimension for both the height and width.

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

    # [5] Resize the image to 48 x 48 and divide it with 255.0 to normalise it to floating point format.

    # Rezize to 48x48
    resized_image = cv2.resize(cropped_image, (48, 48))

    cv2.imshow('image', resized_image)
    cv2.waitKey(0)

    # Normalize image by dividing with 255 to make all values between 0 and 1
    #normalized_image = resized_image/255.0
    
    #return (normalized_image,label)

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

## Part 1: Split filenames into training and test sets



training_set_files = getDatasetFilenames("data\\image-splits\\train.txt")
testing_set_files = getDatasetFilenames("data\\image-splits\\test.txt")
training_set_labels = getDatasetLabels(training_set_files)
testing_set_labels = getDatasetLabels(testing_set_files)

print(training_set_labels)
print(testing_set_labels)

## TODO: Part 2: Load and preprocess images

loadImage("data\\images\\" + training_set_files[0])

## TODO: Part 3: Construct model

model = Sequential()


## TODO: Part 4: Train model



## TODO: Part 5: Print results
