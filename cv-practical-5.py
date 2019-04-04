import os
import cv2
import random
import math
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD

BATCH_SIZE = 64
TRAINING_SET_SIZE = 4000
TESTING_SET_SIZE = 5532

NUMBER_OF_CLASSES = 40
NUMBER_OF_EPOCHS = 20
TOTAL_TRAINING_BATCHES = math.ceil(TRAINING_SET_SIZE / BATCH_SIZE)
TOTAL_TESTING_BATCHES = math.ceil(TESTING_SET_SIZE / BATCH_SIZE)
DATASET_PATH = "data\\images\\"
IMAGE_DIMENSION = 48

def loadImage(filename):
    #Save label
    label = filename.rpartition('_')[0].split('\\')[-1]

    # Load image with flag "Greyscale"
    image = cv2.imread(DATASET_PATH + filename, 0)

    # Get height and width, but skip channels
    height, width = image.shape[:2]

    # Assign smaller value to crop_dim
    if height < width:
      crop_dim = height
    else:
      crop_dim = width
    
    # Determine centerpoints for width and height
    height_middle = height / 2
    width_middle = width / 2

    # Determine crop size in both directions away from center
    crop_dim_half = crop_dim / 2

    # Determine crop positions
    height_lower_boundary = height_middle - crop_dim_half
    height_upper_boundary = height_middle + crop_dim_half
    width_lower_boundary = width_middle - crop_dim_half
    width_upper_boundary = width_middle + crop_dim_half

    # Crop image to lower dimension size from center point
    cropped_image = image[int(height_lower_boundary):int(height_upper_boundary), int(width_lower_boundary):int(width_upper_boundary)]

    # Rezize to 48x48
    resized_image = cv2.resize(cropped_image, (IMAGE_DIMENSION, IMAGE_DIMENSION))

    # Normalize image by dividing with 255 to make all values between 0 and 1
    normalized_image = resized_image / 255.0
    
    # Needed so that the network recognizes the shape
    reshaped_image = np.reshape(normalized_image, (IMAGE_DIMENSION, IMAGE_DIMENSION, 1))

    return (reshaped_image, label)

def getDatasetFilenames(setName):
    # Open set file and save all filenames
    with open(setName) as file:
        set_files = file.readlines()

    # Remove all endline characters from filenames
    set_files = [filename.strip() for filename in set_files]

    return set_files

# returns unique labels dictionary in ['name' => 'number'] format
def getDatasetLabels(file_list):
    label_list = []
    
    for file in file_list:
        label_list.append(file.rpartition('_')[0])

    label_list = list(set(label_list))
    label_dictionary = { label_list[i] : i for i in range(0, len(label_list) ) }

    return label_dictionary

def DataGenerator(image_set_filenames, batch_size, class_label_dictionary):
  while 1:
    # Ensure randomisation per epoch
    random.shuffle(image_set_filenames)

    X = []
    Y = []
    
    for i in range(len(image_set_filenames)):
      #Load image
      image_info = loadImage(image_set_filenames[i])

      #Append image data to X
      X.append(image_info[0])

      #Append image label one-hot vector to Y
      Y.append(np.eye(NUMBER_OF_CLASSES)[class_label_dictionary[image_info[1]]])

      #Comparing count and batch size to see if batch size reached
      #We can emit using count by simply using i+1, as that's exactly
      #what count would amount to
      if (i+1) % batch_size == 0:
        X = np.array(X)
        Y = np.array(Y)

        #Returning X and Y for this batch  
        yield X, Y

        #Resetting X and Y
        X = []
        Y = []

## Part 1: Split filenames into training and test sets

training_set_filenames = getDatasetFilenames("data\\image-splits\\train.txt")
testing_set_filenames = getDatasetFilenames("data\\image-splits\\test.txt")

## Part 2: Create Generators

training_generator = DataGenerator(training_set_filenames, BATCH_SIZE, getDatasetLabels(training_set_filenames))
testing_generator = DataGenerator(testing_set_filenames, BATCH_SIZE, getDatasetLabels(training_set_filenames))

## TODO: Part 3: Construct model

model = Sequential()

model.add(Convolution2D(32, 3, 3, input_shape = (IMAGE_DIMENSION, IMAGE_DIMENSION, 1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(40))
model.add(Activation('softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

## TODO: Part 4: Train model

print('')
print('Starting training!')
print('')

model.fit_generator(generator=training_generator, epochs=NUMBER_OF_EPOCHS, steps_per_epoch=TOTAL_TRAINING_BATCHES)

print('')
print('Training Completed!')
print('')

## TODO: Part 5: Print results

score = model.evaluate_generator(generator=testing_generator, steps=TOTAL_TESTING_BATCHES)

print('')
print('Test score: ' +  str(score[0]))
print('Test accuracy:' +  str(score[1]))
print('')