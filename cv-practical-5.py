import os
from keras.models import Sequential
#from keras.layers.core import Dense, Activation, Flatten
#from keras.layers.convolutional import Convolution2D, MaxPooling2D

## Part 1: Split filenames into training and test sets

# Open training set file and save all filenames
with open("data\\image-splits\\train.txt") as f:
    training_set_files = f.readlines()

# Remove all endline characters from filenames
training_set_files = [x.strip() for x in training_set_files] 

# Open testing set file and save all filenames
with open("data\\image-splits\\test.txt") as f:
    test_set_files = f.readlines()

# Remove all endline characters from filenames
test_set_files = [x.strip() for x in test_set_files]

## TODO: Part 2: Load and preprocess images



## TODO: Part 3: Construct model

model = Sequential()



## TODO: Part 4: Train model



## TODO: Part 5: Print results
