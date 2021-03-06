import os
import cv2
import random
import math
import numpy as np
import gc
import csv

from keras import backend as K
from keras.models import Sequential, Model
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD, Adam
from keras.utils import plot_model
from keras.callbacks import CSVLogger, ModelCheckpoint, LearningRateScheduler
from keras.models import load_model
from keras import applications
from keras import regularizers

from sklearn.metrics import confusion_matrix

BATCH_SIZE = 64
TRAINING_SET_SIZE = 4000
TESTING_SET_SIZE = 5532
TOTAL_TRAINING_BATCHES = math.ceil(TRAINING_SET_SIZE / BATCH_SIZE)
TOTAL_TESTING_BATCHES = math.ceil(TESTING_SET_SIZE / BATCH_SIZE)

NUMBER_OF_CLASSES = 40
NUMBER_OF_EPOCHS = 50
IMAGE_DIMENSION = 48
# IMAGE_DIMENSION = 96 # only for MobileNetV2

DATASET_PATH = "data/images/"
TRAINING_DATA_FILE = "data/image-splits/train.txt"
TESTING_DATA_FILE = "data/image-splits/test.txt"
NETWORK_STRUCTURE_FILE = 'results/network-structure.png'
TRAINING_LOG_FILE = 'results/training_log.csv'
BASIC_CLASSIFIER_FILE = 'results/basic_classifier.h5'
PRETRAINED_CLASSIFIER_FILE = 'results/pretrained_classifier.h5'
CONFUSION_MATRIX_FILE = 'results/confusion_matrix.csv'
AUTOMATIC_MODEL_SEARCH_LOG_FILE = 'results/automatic_model_search_log.csv'

CONFIGURABLE_MODEL_WEIGHTS_FILE = 'results/configurable_model_weights.h5'
BEST_AUTOMATIC_MODEL_FILE = 'results/best_automatic_model.h5'

def loadImage(filename, greyscale=True):
    # Save label
    label = filename.rpartition('_')[0].split('\\')[-1]

    # Load image with flag "Greyscale"
    if (greyscale):
      image = cv2.imread(DATASET_PATH + filename, 0)
    else:
      image = cv2.imread(DATASET_PATH + filename, 1)

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
    if (greyscale):
      reshaped_image = np.reshape(normalized_image, (IMAGE_DIMENSION, IMAGE_DIMENSION, 1))
    else:
      reshaped_image = np.reshape(normalized_image, (IMAGE_DIMENSION, IMAGE_DIMENSION, 3))

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
    label_list.sort()
    label_dictionary = { label_list[i] : i for i in range(0, len(label_list) ) }

    return label_dictionary

def getDataGenerator(image_set_filenames, batch_size, class_labels, randomize=True, greyscale=True):
  while 1:
    # Ensure randomisation per epoch (use only for training)
    if randomize:
      random.shuffle(image_set_filenames)

    X = []
    Y = []
    
    for i in range(len(image_set_filenames)):
      #Load image
      image_info = loadImage(image_set_filenames[i], greyscale=greyscale)

      #Append image data to X
      X.append(image_info[0])

      #Append image label one-hot vector to Y
      Y.append(np.eye(NUMBER_OF_CLASSES)[class_labels[image_info[1]]])

      #Comparing count and batch size to see if batch size reached
      #We can emit using count by simply using i+1, as that's exactly
      #what count would amount to
      if (i + 1) % batch_size == 0:
        X = np.array(X)
        Y = np.array(Y)

        #Returning X and Y for this batch  
        yield X, Y

        #Resetting X and Y
        X = []
        Y = []

def getActualDatasetLabels(image_set_filenames, class_labels):
  actual_labels = []

  for filename in image_set_filenames:
    label = filename.rpartition('_')[0].split('\\')[-1]
    actual_labels.append(class_labels[label])

  return actual_labels

def createBasicClassifier(plot=False, regulizer=True, custom_learning_rate=True):
  classifier = Sequential()

  if regulizer:
    classifier.add(Conv2D(32, (3, 3), input_shape = (IMAGE_DIMENSION, IMAGE_DIMENSION, 1), kernel_regularizer=regularizers.l2(0.01)))
  else:
    classifier.add(Conv2D(32, (3, 3), input_shape = (IMAGE_DIMENSION, IMAGE_DIMENSION, 1)))
  
  classifier.add(BatchNormalization())
  classifier.add(Activation('relu'))
  classifier.add(MaxPooling2D(pool_size=(2, 2)))

  classifier.add(Dropout(0.5))

  if regulizer:
    classifier.add(Conv2D(32, (3, 3), kernel_regularizer=regularizers.l2(0.01)))
  else:
    classifier.add(Conv2D(32, (3, 3)))
  
  classifier.add(BatchNormalization())
  classifier.add(Activation('relu'))
  classifier.add(MaxPooling2D(pool_size=(2, 2)))

  classifier.add(Dropout(0.5))

  classifier.add(Flatten())
  if regulizer:
    classifier.add(Dense(40, activation="softmax", kernel_regularizer=regularizers.l2(0.01)))
  else:
    classifier.add(Dense(40, activation="softmax"))

  if (custom_learning_rate):
    optimizer_function = SGD(lr=0.0, momentum=0.8, decay=0.0, nesterov=False)
    # optimizer_function = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
  else:
    optimizer_function = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

  classifier.compile(optimizer=optimizer_function, loss='categorical_crossentropy', metrics=['accuracy'])
  
  # plot cnn structure to the file
  if (plot):
    plot_model(classifier, show_shapes=True, to_file=NETWORK_STRUCTURE_FILE)

  return classifier

def createPretrainedClassifier(plot=True):
  base_model = applications.vgg19.VGG19(include_top=False, weights='imagenet', input_shape=(IMAGE_DIMENSION, IMAGE_DIMENSION, 3), pooling='avg')
  # base_model = applications.mobilenet_v2.MobileNetV2(include_top=False, weights='imagenet', input_shape=(IMAGE_DIMENSION, IMAGE_DIMENSION, 3), pooling='avg')

  for layer in base_model.layers:
    layer.trainable = False

  # add a new top layer
  x = base_model.output

  # x = Dense(512, activation='relu', name='dense_1')(x)
  # x = Dense(256, activation='relu', name='dense_2')(x)
  predictions = Dense(NUMBER_OF_CLASSES, activation='softmax', name='final_layer')(x)
  
  # this is the model we will train
  classifier = Model(inputs=base_model.input, outputs=predictions)

  optimizer_function = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
  classifier.compile(optimizer=optimizer_function, loss='categorical_crossentropy', metrics=['accuracy'])

  # plot cnn structure to the file
  if (plot):
    plot_model(classifier, show_shapes=True, to_file=NETWORK_STRUCTURE_FILE)

  return classifier

# AUTOMATIC CLASSIFIER SEARCH CONCEPT

# different configurations of custom learning rate
# different regularization values on the last Dense layer
# different amount (from 0 to 2) of Dense layers (with RELU), and with different amount of nodes

# feature vector:
# [ learning_rate, regularization_value, amount_of_dense_layers, amount_of_nodes_per layer[2]] ]

# form a vector of different configurations
# compile a new classifier
# train for X epochs
# evaluate it and save validation accuracy

# find out which parameter configuration vector yielded the best validation accuracy
#
def reconfigureClassifier(model_weights_file, learning_rate, regularization_value, amount_of_dense_layers, amount_of_nodes_per_layer):
  base_model = applications.vgg19.VGG19(include_top=False, weights='imagenet', input_shape=(IMAGE_DIMENSION, IMAGE_DIMENSION, 3), pooling='avg')

  for layer in base_model.layers:
    layer.trainable = False

  # add a new top layer
  x = base_model.output

  # create dense layers
  for i in range(amount_of_dense_layers):
    x = Dense(amount_of_nodes_per_layer[i], activation='relu', name='dense_' + str(i))(x)
  
  predictions = Dense(
    NUMBER_OF_CLASSES,
    activation='softmax',
    kernel_regularizer=regularizers.l2(regularization_value),
    name='final_layer'
  )(x)
  
  # this is the model we will train
  classifier = Model(inputs=base_model.input, outputs=predictions)

  classifier.load_weights(model_weights_file, by_name=True, reshape=True, skip_mismatch=True)

  optimizer_function = SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
  classifier.compile(optimizer=optimizer_function, loss='categorical_crossentropy', metrics=['accuracy'])

  return classifier

def testClassifier(classifier, greyscale=True):
  print('Starting classifier testing...')

  score = classifier.evaluate_generator(
    generator=getDataGenerator(testing_set_filenames, BATCH_SIZE, class_labels, randomize=False, greyscale=greyscale),
    steps=TOTAL_TESTING_BATCHES
  )
  print('Test score: ' +  str(score[0]))
  print('Test accuracy:' +  str(score[1]))

  print('Starting prediction of classes...')
  predictions = classifier.predict_generator(
    generator=getDataGenerator(testing_set_filenames, BATCH_SIZE, class_labels, randomize=False, greyscale=greyscale),
    steps=TOTAL_TESTING_BATCHES
  )

  actual_testing_labels = getActualDatasetLabels(testing_set_filenames, class_labels)
  matrix = confusion_matrix(
    actual_testing_labels,
    predictions[:len(actual_testing_labels)].argmax(axis=1)
  )

  total_true_guesses = 0
  for i in range(len(class_labels)):
    total_true_guesses += matrix[i][i]
  print('Prediction accuracy (based on confusion matrix): ', total_true_guesses / len(actual_testing_labels))

  print('Writing confusion matrix to the file...')
  np.savetxt(
    CONFUSION_MATRIX_FILE,
    np.asarray(matrix),
    fmt="%d",
    delimiter=","
  )
  print('Classes have been predicted, confusion matrix written into the file successfully!')

INITIAL_LEARNING_RATE = 0.0125
LEARNING_RATE_DECAY = 0.035
def expDecay(epoch):
   initial_learning_rate = INITIAL_LEARNING_RATE
   k = LEARNING_RATE_DECAY
   learning_rate = initial_learning_rate * math.exp(-k * epoch)

   return learning_rate

## ---------------------------------------------------------------------------------------------
## MAIN APPLICATION
## ---------------------------------------------------------------------------------------------

training_set_filenames = getDatasetFilenames(TRAINING_DATA_FILE)
testing_set_filenames = getDatasetFilenames(TESTING_DATA_FILE)

class_labels = getDatasetLabels(testing_set_filenames)

while (True):
  user_input = input(
    '''
    0 - exit

    1 - train basic classifier
    2 - test basic classifier

    3 - train pre-trained classifier (VGG19)
    4 - test pre-trained classifier (VGG19)

    5 - automatic model creation
    6 - test automatic model
    Option:
    '''
  )

  # exit
  if (user_input == '0'):
    exit()

  # train basic classifier
  if (user_input == '1'):
    print('Starting classifier training...')

    classifier = createBasicClassifier(plot=True)

    lrate = LearningRateScheduler(expDecay, verbose=1)

    training_logger = CSVLogger(TRAINING_LOG_FILE, append=False, separator=',')
    classifier_recorder = ModelCheckpoint(BASIC_CLASSIFIER_FILE, save_best_only=True, monitor='val_acc', mode='max')
    classifier.fit_generator(
      generator=getDataGenerator(training_set_filenames, BATCH_SIZE, class_labels),
      validation_data=getDataGenerator(testing_set_filenames, BATCH_SIZE, class_labels, randomize=False),
      validation_steps=TOTAL_TESTING_BATCHES,
      epochs=NUMBER_OF_EPOCHS,
      steps_per_epoch=TOTAL_TRAINING_BATCHES,
      callbacks=[lrate, training_logger, classifier_recorder]
    )
    print('Classifier has been trained succesfully!')

  # basic classifier testing
  if (user_input == '2'):
    testClassifier(
      classifier=load_model(BASIC_CLASSIFIER_FILE),
      greyscale=True
    )

  # train pretrained clssifier
  if (user_input == '3'):
    print('Starting to train pretrained classifier...')

    classifier = createPretrainedClassifier(plot=True)

    training_logger = CSVLogger(TRAINING_LOG_FILE, append=False, separator=',')
    classifier_recorder = ModelCheckpoint(PRETRAINED_CLASSIFIER_FILE, save_best_only=True, monitor='val_acc', mode='max')
    classifier.fit_generator(
      generator=getDataGenerator(training_set_filenames, BATCH_SIZE, class_labels, greyscale=False),
      validation_data=getDataGenerator(testing_set_filenames, BATCH_SIZE, class_labels, randomize=False, greyscale=False),
      validation_steps=TOTAL_TESTING_BATCHES,
      epochs=NUMBER_OF_EPOCHS,
      steps_per_epoch=TOTAL_TRAINING_BATCHES,
      callbacks=[training_logger, classifier_recorder]
    )
    print('Classifier has been trained succesfully!')

  # pre-trained classifier testing
  if (user_input == '4'):
    testClassifier(
      classifier=load_model(PRETRAINED_CLASSIFIER_FILE),
      greyscale=False
    )

  # automatic model creation (finding best parameters)
  if (user_input == '5'):
    print('Starting automatic classifier creation...')

    # prepare log file
    with open(AUTOMATIC_MODEL_SEARCH_LOG_FILE, 'w') as f:
      writer = csv.writer(f)
      writer.writerow([
        'Number of layers', 
        'Size of dense layer 1', 
        'Size of dense layer 2', 
        'Learning rate',
        'Regularization value',
        'Validation accuracy'
      ])

    best_validation_accuracy = 0

    # create initial classifier which we are going to tweak
    classifier = createPretrainedClassifier(plot=False)

    # automatic model search parameters
    epochs_without_improvment = 0
    break_threshold = 2
    is_network_structure_changed = False
    extra_epochs = 5 # extra epochs to train if network layer structure has changed

    # always compare two subsequent epochs to see if it is rising and do not break the loop in this case even threshold was exceeded!
    previous_validation_accuracy = 0

    # introduced break out of the for loop if few sequence of configurations did not improve (using same break_threshold)
    configurations_without_improvment = 0

    # params of model search
    initial_learning_rate = 0.02
    learning_rate_step = -0.0025

    initial_regularization_value = 0.0175
    regularization_value_step = -0.0025

    # initial_amount_of_dense_layers = 0
    initial_amount_of_nodes_per_layer_1 = 512
    amount_of_nodes_per_layer_1_step = -128

    initial_amount_of_nodes_per_layer_2 = 256
    amount_of_nodes_per_layer_2_step = -64

    for i in range(0, 3): # dense layers
      is_network_structure_changed = True

      for j in range(0, 4): # parameters per first layer
        is_network_structure_changed = True

        for k in range(0, 4): # parameters per second layer
          is_network_structure_changed = True

          for l in range(0, 7): # parameters per learning rate
            for r in range(0, 7): # parameters per regularization value

              if (not(i == 0 and j == 0 and k == 0 and l == 0 and r == 0)):
                print('')
                print('Reconfiguring classifier:')
                print('Feature vector: [' + str(i) + ', ' + str(j) + ', ' + str(k) + ', ' + str(l) + ', ' + str(r) + ']')
                classifier = reconfigureClassifier(
                  model_weights_file=CONFIGURABLE_MODEL_WEIGHTS_FILE,
                  learning_rate=initial_learning_rate + l * learning_rate_step,
                  regularization_value=initial_regularization_value + r * regularization_value_step,
                  amount_of_dense_layers=i,
                  amount_of_nodes_per_layer=[
                    initial_amount_of_nodes_per_layer_1 + j * amount_of_nodes_per_layer_1_step,
                    initial_amount_of_nodes_per_layer_2 + k * amount_of_nodes_per_layer_2_step
                  ]
                )
              
              # give some additional epoch to explore when network layer structure changes
              if (is_network_structure_changed):
                print('Network layer structure has changed, training for ' + str(extra_epochs) + ' extra epochs.')
                epochs_without_improvment -= extra_epochs
                is_network_structure_changed = False

              while (True):
                classifier.fit_generator(
                  generator=getDataGenerator(training_set_filenames, BATCH_SIZE, class_labels, greyscale=False),
                  epochs=1,
                  steps_per_epoch=TOTAL_TRAINING_BATCHES,
                )
                classifier.save_weights(CONFIGURABLE_MODEL_WEIGHTS_FILE)

                score = classifier.evaluate_generator(
                  generator=getDataGenerator(testing_set_filenames, BATCH_SIZE, class_labels, randomize=False, greyscale=False),
                  steps=TOTAL_TESTING_BATCHES
                )
                print('------------------------------------- Test accuracy:' +  str(score[1]))

                # fill log
                with open(AUTOMATIC_MODEL_SEARCH_LOG_FILE, 'a') as f:
                  writer = csv.writer(f)
                  writer.writerow([
                    str(i), 
                    str(initial_amount_of_nodes_per_layer_1 + j * amount_of_nodes_per_layer_1_step), 
                    str(initial_amount_of_nodes_per_layer_2 + k * amount_of_nodes_per_layer_2_step), 
                    str(initial_learning_rate + l * learning_rate_step),
                    str(initial_regularization_value + r * regularization_value_step),
                    str(score[1])
                  ])

                if (score[1] > best_validation_accuracy):
                  print('!!! NEW BEST MODEL ENCOUNTERED !!!')
                  best_validation_accuracy = score[1]
                  classifier.save(BEST_AUTOMATIC_MODEL_FILE)
                  epochs_without_improvment = -extra_epochs
                  configurations_without_improvment = 0

                # break the loop if threshold exceeded and validation accuracy is not rising
                if (epochs_without_improvment >= break_threshold and previous_validation_accuracy > score[1]):
                  print('Current configuration is not improving, breaking loop!')
                  epochs_without_improvment = 0
                  previous_validation_accuracy = score[1]
                  break
                previous_validation_accuracy = score[1]
                epochs_without_improvment += 1

              if (configurations_without_improvment >= break_threshold):
                print('Sequence of configurations is not improving, breaking loop!')
                configurations_without_improvment = 0
                break
              configurations_without_improvment += 1

              # release GPU memmory
              K.clear_session()
              del classifier
              for q in range(0, 4):
                gc.collect()

          if (i < 2): break
        if (i == 0): break

    print('Automatic model search completed successfully!')

  # test best automatic model found
  if (user_input == '6'):
    testClassifier(
      classifier=load_model(BEST_AUTOMATIC_MODEL_FILE),
      greyscale=False
    )