import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential,Model
from keras.layers import Flatten, Dense, Lambda,Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

images = []
measurements = []
lines = []

# read the example images, and recorded new data from wo clockwise loops, two counterclockwise loops,
# one loop from second trail and
# augumented images from recovering scenes
dir_data = './data/driving_log.csv'
with open(dir_data) as csvfile:
    reader = csv.reader(csvfile)
    next(reader, None)
    for line in reader:
        lines.append(line)
for line in lines:
    source_path = line[0]
    # the sample images have relative path, and the newly recorded images have absolute path
    foldername = source_path.split('/')[0]
    if foldername =='IMG':
        filename = source_path.split('/')[-1]
        filename = './data/IMG/' + filename
    else:
        filename = source_path
    image = cv2.imread(filename)
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)
    # flip the images 
    image_flipped = np.fliplr(image)
    measurement_flipped = -measurement
    images.append(image_flipped)
    measurements.append(measurement_flipped)

# read the recorded files for additional images near sharper left turns
use_sharpleft_data = True
if use_sharpleft_data:
    lines = []
    dir_recover = './data/driving_log_record_SharpLeft.csv'
    with open(dir_recover) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)
    for line in lines:
        file_path_name = line[0]
        image = cv2.imread(file_path_name)
        images.append(image)
        measurement = float(line[3])
        measurements.append(measurement)


## Training data and labels
X_train = np.array(images)
y_train = np.array(measurements)

del images 
del measurements

print(X_train.shape)
print(y_train.shape)

## Build the model
# Image pre-processing: normalization and cropping
model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(3,160,320)))

# LeNet
LeNet = False
if LeNet:
    model.add(Convolution2D(6,(5,5),activation= "relu"))
    model.add(MaxPooling2D())
    model.add(Convolution2D(6,(5,5),activation= "relu"))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(120))
    model.add(Dense(84))
    model.add(Dense(1))

# Traffic Sign Classifier
TrafficNet = True
if TrafficNet:
    model.add(Convolution2D(16, (5, 5), strides=(2,2), activation= "relu"))
    model.add(MaxPooling2D())
    model.add(Convolution2D(32, (5, 5), strides=(2,2), activation= "relu"))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Dense(128))
    model.add(Dense(64))
    model.add(Dense(32))
    model.add(Dense(1))

# nVidia
nVidia = False
if nVidia:
    model.add(Convolution2D(24, (5, 5), activation="relu", strides=(2, 2)))
    model.add(Convolution2D(36, (5, 5), activation="relu", strides=(2, 2)))
    model.add(Convolution2D(48, (5, 5), activation="relu", strides=(2, 2)))
    model.add(Convolution2D(64, (3, 3), activation= "relu"))
    model.add(Convolution2D(64, (3, 3), activation= "relu"))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))

## Train the model
model.compile(loss='mse',optimizer='adam')
history_object = model.fit(X_train,y_train,validation_split = 0.2, shuffle = True, epochs = 8, verbose = 1)
model.save('model_TrafficNet.h5')

## Plot Training and Validation MSE Loss vs. Number of Epochs
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title(' Mean Squared Error Loss')
plt.xlabel('Number of Epochs')
plt.legend(['Training set', 'Validation set'], loc='upper right')
plt.show()
plt.savefig('TrainingLoss.JPG')
