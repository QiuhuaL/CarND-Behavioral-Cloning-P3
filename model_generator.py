import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential,Model
from keras.layers import Flatten, Dense, Lambda,Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
import os
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

lines = []

# read the example images, and recorded new data from wo clockwise loops, two counterclockwise loops,
# one loop from second trail and
# augumented images from recovering scenes
dir_data = './data/driving_log_all.csv'
with open(dir_data) as csvfile:
    reader = csv.reader(csvfile)
    next(reader, None)
    for line in reader:
        lines.append(line)

train_samples, validation_samples = train_test_split(lines, test_size=0.2)

def generator(samples, batch_size = 32):
    num_samples = len(samples)
    correction = 0.2
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            measurements = []
            for batch_sample in batch_samples:
                # center images
                source_path = batch_sample[0]
                # the sample images have relative path, and the newly recorded images have absolute path
                foldername = source_path.split('/')[0]
                if foldername == 'IMG':
                    filename = source_path.split('/')[-1]
                    filename = './data/IMG/' + filename
                else:
                    filename = source_path
                image = cv2.imread(filename)
                measurement = float(batch_sample[3])
                images.append(image)
                measurements.append(measurement)
                # flip the images
                image_flipped = np.fliplr(image)
                measurement_flipped = -measurement
                images.append(image_flipped)
                measurements.append(measurement_flipped)

                # left images
                source_path = batch_sample[1]
                if foldername == 'IMG':
                    filename = source_path.split('/')[-1]
                    filename = './data/IMG/' + filename
                else:
                    filename = source_path
                image = cv2.imread(filename)
                images.append(image)
                measurements.append(measurement + correction)
                # flip the images
                image_flipped = np.fliplr(image)
                measurement_flipped = - (measurement + correction)
                images.append(image_flipped)
                measurements.append(measurement_flipped)

                # right images
                source_path = batch_sample[2]
                if foldername == 'IMG':
                    filename = source_path.split('/')[-1]
                    filename = './data/IMG/' + filename
                else:
                    filename = source_path
                image = cv2.imread(filename)
                images.append(image)
                measurements.append(measurement - correction)
                # flip the images
                image_flipped = np.fliplr(image)
                measurement_flipped = -(measurement - correction)
                images.append(image_flipped)
                measurements.append(measurement_flipped)

            X_train = np.array(images)
            y_train = np.array(measurements)

            yield shuffle(X_train, y_train)


# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

## Build the model

# Image pre-processing: normalization and cropping
model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(3,160,320)))

# LeNet
LeNet = False
if LeNet:
    model.add(Convolution2D(6,(5,5),activation='relu'))
    model.add(MaxPooling2D())
    model.add(Convolution2D(6,(5,5),activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(120))
    model.add(Dense(84))
    model.add(Dense(1))

# Traffic Sign Classifier
TrafficNet = False
if TrafficNet:
    model.add(Convolution2D(16, (5, 5), strides=(2,2), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Convolution2D(32, (5, 5), strides=(2,2), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Convolution2D(64, (5, 5), strides=(2, 2), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Dense(128))
    model.add(Dense(64))
    model.add(Dense(32))
    model.add(Dense(1))

# nVidia
nVidia = True
if nVidia:
    model.add(Convolution2D(24, (5, 5), activation="relu", strides=(2, 2)))
    model.add(Convolution2D(36, (5, 5), activation="relu", strides=(2, 2)))
    model.add(Convolution2D(48, (5, 5), activation="relu", strides=(2, 2)))
    model.add(Convolution2D(64, (3, 3), activation="relu"))
    model.add(Convolution2D(64, (3, 3), activation="relu"))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))


## Train the model
model.compile(loss='mse',optimizer='adam')
history_object = model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples),epochs = 5, verbose = 1)

model.save('model_generator.h5')

## Plot Training and Validation MSE Loss vs. Epochs
print(history_object.history.keys())
print('Loss')
print(history_object.history['loss'])
print('Validation Loss')
print(history_object.history['val_loss'])
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title(' Mean Squared Error Loss')
plt.xlabel('Number of Epochs')
plt.legend(['Training set', 'Validation set'], loc='upper right')
plt.show()
plt.savefig('TrainingLoss_model_generator.JPG')
