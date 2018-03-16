# **Behavioral Cloning** 

## Gary Holness

### My implementation of Behavioral Cloning for the Driving task

### Note:  I only implemented autonomous driving for the easy track

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:

* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./steeringAngleDist.jpg "Priors over Steering Angles"
[image2]: ./sharpSteeringAngleDist.jpg "Priors for Top 5-percent Steering Angles"

## High Level Description

#### The key observation is scucessfully getting the car to drive autonomously around the track was to watch the steering actions predicted by the trained network.

My observation was that the steering predictions were always very
small in magnitude.  In MATLAB, I did an analysis of the steering
angles for the training data and found that most of the steering angles
were minor variations from zero.  The reason for this stems from the
fact that, because the track is mostly straight, most of the training
data involved minor steering angles involved in keeping the car in
the center of the road.

Additionally, when driving around the track starting in the initial
position in the simulator, driving around the track involved moslty
turning left.    This skewed the data towards camera views (center,
left, right) that depicted the side of the road (curb?) primarily
on the right extreme of the image data.

Yet still, the image data includes a large amount of pixel information
associated with scenery that is not critical to the driving task.
Objects such as trees, kills, marsh grass, and water, are not critical
to driving.

Addressing these data issues involved a large number of augmentations
to the data:

1.  Driving around track multiple times (4 times) to give enough data
    for learning a good model.  With more examples the idea is that
    the driving would be smoother.

2.  Corrective actions were added by pausing recording, driving to
    the right edge of the roadway, unpausing, and recording the
    corrective action of turning left to bring the car back to
    the center of the roadway.   This was repeated on the left
    edge of the roadway.   Then an entire lap was recorded by
    alternating the corrective action on the right side, then
    left side, etc.  This served to generate more dramatic steering
    actions needed to teach the network how to recover back to
    the center of the roadway above and beyond simple corrections
    maintaining position once in the center of the roadway.
    Doing an entire lap's worth of corrective actions alternating
    between the right and left sides of the track generated
    data that teaches the network how to recover to the roadway
    center in all parts of the track.

3.  I turned around the car and drove it in the other direction
    around the track.  This resulting in data exemplars consisting
    of steering actions where the turns were mostly right hand turns.
    I drove in the other direction for 4 laps.  I followed this with
    a complete lap of alternating corrective actions going around
    the track in the reverse direction.

4.  Separately in MATLAB I did an analysis of the steering angles in
    the recorded data set.  I observed that the prior distribution over
    large steering angles had very little probability mass.  This explained
    why my trained model kept running of the side of the track.  To fix this,
    using the log file in MATLAB, I produced a version of the data set that
    contains the top 5% largest magnitude steering angles.  This was done
    for the 5% most negative (less than 5-th percentile) and the 5% most
    positive (greater than 95-th percentile).

5.  Asymmetric corrections.  When adding the steering actions associated with
    the right hand camera, because it is closer to the right edge of the
    track, the corrective action should be larger.  The reason from this is
    because of the relationship between arch-length, turn radius, and angle
    traversed.  The right hand side of the car must traverse a longer arch
    length.  As such, when correcting, it has further to travel so the
    correction shoud be larger so that the effect is to reduce the turn
    radius more aggressively because it has further to travel.

6.  I cropped pixels at bottom of image to remove car dashboard and also
    cropped from the top to eliminate the environmental features not
    important to driving.

7.  I generally slowed down when driving so that more images are taken.


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.  I have provided extensive amount of comments describing my implementation and rationale
behind the implementation.

### Model Architecture and Training Strategy

#### 1. I implemented both LeNet and the NVIDIA network. I ended up using the NVIDIA network for my submission

My NVIDIA Network, lines 310-324, consists of 3 5x5 Convolution layers followed by 2 3x3 Convolution layers.  These provide identification of basic features and complex aggregates of basic features.  The Convolution layers are follwed by Flattening and the network ends in 4 Dense layers of dimension 100, 50, 10,
and finally 1.

The input to the NVIDIA network (lines 288-290) does Sequential inpt followed by a Lambda layer that
normalizes the RGB data followed by Cropping. The normalization and Cropping serve to improve performance of the network by giving better pixel values as well as removing pixel information not critical to driving.

My NVIDIA Network Layers are..

Convolution2D:  24-filters, 5x5, subsample=(2,2), activation=relu
Convolution2D:  36-filters, 5x5, subsample=(2,2), activation=relu
Convolution2D:  48-filters, 3x3, subsample=(2,2), activation=relu
Convolution2D:  64-filters, 3x3, subsample=(2,2), activation=relu
Convolution2D:  64-filters, 3x3, subsample=(2,2), activation=relu
MaxPooling2D: 
Dropout:  drop_prob=0.1
Flatten
Dense:  100
Dropout: drop_prob=0.1
Dense: 50
Dropout: drop_prob=0.1
Dense: 10
Dense: 1

The MaxPooling adds nonlinearity and the dropout performs regularization.
The relu activation introduces nonlinearity at the convolutions.


The model includes RELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer (code line 18). 

#### 2. Attempts to reduce overfitting in the model

My model uses dropout layers to reduce overfitting. This is done at the Dense layers 
closer to the output, because I chose to reduce overfitting at the representation of
more complex features closer to the output of the network rather than the simple features
closer to the input of the network.  

I validated the model using 20% of the data. This was done (line 327) at the call to model.fit().
I also shuffle the data and run for a total of 5 EPOCS.  For each version of the model, I tested
it by running the simulator to ensure that the vehicle stays on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 327)

#### 4. Appropriate training data

A very large amount of my time was spent gathering the training data.  I drove the car around the track for four laps.  In addition I drove an extra lap (5th) by pausing and unpausing the record button and perfoming recovery actions alternating between the right hand edge of the road and the left hand edge of the road.  I did this recovery for an entire lap.  I also turned the car around and recorded driving data for four laps in the opposite direction around the track.  I then recorded a 5th lap again doing recovery actionsalternating between the left and and right hand side around the entire track.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

I implemented both LeNet and the NVIDIA driving network.  This consisted of five convolution layers
followed by flattening and four dense layers.   I added MaxPooling after the last convolution layer
and I inserted Dropout between the Dense layers.

The convolution layers ware appropriate in order to perform feature extration.  Using so many layers
essentially built up increasingly complex assemblages of features.  Using Dropout at the Dense Layers
is appropriate because I felt that regularization should happen at the complex assemblage and not
the basic features. 

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Creation of the Training Set & Training Process

For a very many versions of my model, the car's behavior was such that it would drift off the side of the track. This occured in spite of inclusion of corrective/recovery actions in the data.  Upon closer observation of the trained network, I noticed that the steering angles tended to be quite small.  It dawned on me that the track is mostly straight.  When driving straight, one makes mostly subtle steering corrections.
But when the car approached a curve, it eventually drifted off the course because the trained model was unable to predict a larger steering angle.  This piqued my curiosity and I decided to take the log file and analyze it in MATLAB.  Within MATLAB, I computed a histogram for the distribution of the steering angles.  The histogram had 20 bins.

[image1]

This distribution had a mode at very small steering angle and not much probability mass for the larger postive and negative steering angles.  I needed to boost the priors associated with larger magnitude (positive/negative) steering angles.  So, in MATLAB, I produced a second version of the data log file by isolating data points whose steering angles were either less than the 5-th percentile (most negative 5%) or  whose steering angles were greater than the 05-th percentile (most positive 5%).  I produced a histogram of this segregated data set.

[image2]

I then went to my model and loaded the regular data set and the big-angle data set separately. When I construct the training set, I oversample the big-angle data set in order to boost the priors for large steering angles.  This appears to have worked very well for me resulting in much better driving performance.

I did implement data augmentation by flipping the images, but found it was not necessary because my driving in the reverse direction and boosting priors for large steering angle corrections was sufficient. The code is there but I turned off augmentation using a boolean conditional.

In all I had a total of 90,567 data points.  Of this, 72,453 were used for training and 18114 were used for validation.  My training epocs are as follows for training loss and validation loss. This is from TensorFlow running on Amazon AWS...


72453/72453 [==============================] - 97s - loss: 0.0160 - val_loss: 0.0371
Epoch 2/5
72453/72453 [==============================] - 93s - loss: 0.0113 - val_loss: 0.0270
Epoch 3/5
72453/72453 [==============================] - 94s - loss: 0.0093 - val_loss: 0.0161
Epoch 4/5
72453/72453 [==============================] - 93s - loss: 0.0081 - val_loss: 0.0150
Epoch 5/5
72453/72453 [==============================] - 94s - loss: 0.0074 - val_loss: 0.0130

As you can see, my validation loss decreases over each of 5 training EPOCS. I used only 5 epocs
for reasons of time and also I felt loss was small enough (2 decimal places) given the
amount of training and validation data (its a tiny mean squared error over so many instances).

