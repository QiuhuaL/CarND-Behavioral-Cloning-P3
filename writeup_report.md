# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image_model]: ./examples/NVIDIA_model.png "Model Visualization"
[image_clockwise]: ./examples/center_clockwise.jpg "Clockwise Driving Image"
[image_counterclockwise]: ./examples/center_counterclock.jpg "Counter-Clockwise Driving Image"
[image_mountain]: ./examples/center_mountain.jpg "Image from track 2"
[image_recover1]: ./examples/recover1.jpg "Recovery Image"
[image_recover2]: ./examples/recover2.jpg "Recovery Image"
[image_recover3]: ./examples/recover3.jpg "Recovery Image"
[image_recover4]: ./examples/recover4.jpg "Recovery Image"
[image_normal]: ./examples/normal.jpg "Normal Image"
[image_flipped]: ./examples/flipped.jpg "Flipped Image"
[image_original]: ./examples/original.jpg "Original Image"
[image_normalized]: ./examples/normalized.jpg "Normalized Image"
[image_cropped]: ./examples/cropped.jpg "Cropped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed
For this project, I tried different modesl, including the LeNet model, the model that I built for the second project, traffic sign classifier and the NVIDIA model.

Test results on self-driving show that both the traffic sign classifier and the NVIDIA model were able to finish one lap safely, but the traffic sign classifier has several cases where the car wandered to the side of the road and recovered back later, while testing on the model from NVIDIA model show that the car was able to stay in the center lane for the whole loop.  For the final result of this project, I chose to use the NVIDIA model (model.py lines 106 to 115).

The model consists of 4 convolutional layers with the filter size of 5x5 or 3x3 and depths between 24 and 64 (model.py lines 106-110), 4 fully connected layers with dimension of output 100,50,10,and 1.

The model includes RELU layers to introduce nonlinearity (code line 106-110), and the data is normalized in the model using a Keras lambda layer (code line 73). 

#### 2. Attempts to reduce overfitting in the model

The data set was splitted into training and validation data sets, with a ration of 80%:20%.The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 118-130), where I plotted the mean squared error loss on the training and validation data sets versus the number of epochs. The number of epochs chosen for the final model was 5.

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 118).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of two laps of clockwise center lane driving , two laps of counter-clockwise center lane driving, recovering from the left and right sides of the road, one lap of center land driving from the mountain track, and the example data set provided by Udacity.  The images were also flipped to augument the traing data set.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to learn the steering behavior and stay as much as to the center of the lane as possible in the self-driving automation mode.

My first step was to try the convolution neural network model LeNet to see how it goes (model.py line 78-85). I thought this model might be appropriate because LeNet has been widely used model and has various layers to learn the differnet level of features. I then tried with the traffic sign classifier model that I built for project 2 (model.py line 90-101), that is based on LeNet. With the training data that I have, this network got pretty good results on the testing run on track 1, where the car was able to drive the whole lap on track 1 safely, with several cases where the car wandered to the side of the road, but recovered later(run1_TrafficNet.mp4). The NVIDIA model has been proven succesful for real automatic driving tasks and I finally decided to try this model (model.py line 106-115) and achieved the best testing result on lane 1 among the three models that I tried, where the car always stays in the lane on track 1.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set and chose the number of epochs that achives good loss on the validation set. 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track when there is a sharp turn to the left. The car kept moving off the road to the dirt ground.  To overcome this, I recorded more data near the sharp turns.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road. The final video result using the NVIDIA model was recorded in the file video.mp4. 

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following sequential layers:

|Layer (type)           |   Description	                      |
|:---------------------:|:--------------------------------------:|
| Lambda                | Image Normalization   |       
| cropping2d            | Image Cropping  |
| Convolution2D         | kernel:5x5, depth:24, max pooling 2x2, RELU|
| Convolution2D         | kernel:5x5, depth:48, max pooling 2x2, RELU| 
| Convolution2D         | kernel:3x3, depth:64, RELU    |     
| Convolution2D         | kernel:3x3, depth:64, RELU    |     
| Flatten               | fully connected, output size 1164        |
| Dense                 | fully connected, output size 100          |   
| Dense                 | fully connected, output size 50               |
| Dense                 | fulluy connected, output size 10                 |
| Dense                 | fulluy connected, output size 1     |           

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)
![alt text][image_model]

The lamda layer is to noralize the images and the cropping is used to choose an area of interest that excludes the sky and/or the hood of the car. Here is an example of an input image and its cropped version after passing through the normlization layer and the Cropping2D layer:
![alt text][image_original]
![alt text][image_normalized]
![alt text][image_cropped]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps of clockwise center lane driving on track one. Here is an example image of clockwise center lane driving:

![alt text][image_clockwise]


I then recorded two laps of counter-clockwise center lane driving on track one. Here is an example image of counter-clockwisecenter lane driving:

![alt text][image_counterclockwise]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover driving from the sides. These images show what a recovery looks like starting from driving past the right lane line and back to the center of the lane  :

![alt text][image_recover1]
![alt text][image_recover2]
![alt text][image_recover3]
![alt text][image_recover4]

Then I collected one lap of driving on track 2 to get more data.

![alt text][image_mountain]

Besides the new images collected, I also used the example data set provided by Udacity to increase the the number of images for training.

To further augment the data sat, I also flipped images and angles thinking that this would reduce the left or right turn bias.  For example, here is an image that has then been flipped:

![alt text][image_normal]
![alt text][image_flipped]

When I tried with the above images from the center cemera, I found that the car was generally driving automatically well, but have the problem of wandering off to the side of the road when there are sharp turns, so I collected more data around those areas from track 1. 

After the collection process, I had 54345 number of data points from the center images. I then preprocessed this data by a lamda layer normalization and 2D cropping to choose an area of interest that excludes the sky and/or the hood of the car.

I finally randomly shuffled the data set and put 20% of the data into a validation set. The model was trained with 43476 samples and validated on 10869 samples. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by MSE loss funcion on the training and testing data set. I used an adam optimizer so that manually training the learning rate wasn't necessary.

The results are surprisingly well with the center images, so the model.py that I submitted for review are only using the images from the center camera. 

#### 4. Extra Expriment With Images From All Three Cameras and the Generator
But out of curiosity, I did try to use the images from all three cemeras and use the generator to load data and preprocess it on the fly, in batch size of 32. The model with the images from all three cameras and with a generator (model_generator.py). 

For the images from the left cemera, I used the correction = 0.2 to the steering measurement and correction = 0.2 for the images from the right camera.

I found that the generator greatly reduced the memory consumption, which is great for future usage with large training data set. Comparing to loading all images at once, the use of generator did slow down the whole training process when loading and processing the images batch by batch. 
# CarND-Behavioral-Cloning-P3
