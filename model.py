#####
# Gary Holness
# gary.holness@gmail.com
#
# Autonomous Driving Assignment
#
# The key observation is scucessfully getting the car to drive
# autonomously around the track was to watch the steering actions
# predicted by the trained network.
#
# My observation was that the steering predictions were always very
# small in magnitude.  In MATLAB, I did an analysis of the steering
# angles for the training data and found that most of the steering angles
# were minor variations from zero.  The reason for this stems from the
# fact that, because the track is mostly straight, most of the training
# data involved minor steering angles involved in keeping the car in
# the center of the road.
#
# Additionally, when driving around the track starting in the initial
# position in the simulator, driving around the track involved moslty
# turning left.    This skewed the data towards camera views (center,
# left, right) that depicted the side of the road (curb?) primarily
# on the right extreme of the image data. 
#
# Yet still, the image data includes a large amount of pixel information
# associated with scenery that is not critical to the driving task.
# Objects such as trees, kills, marsh grass, and water, are not critical
# to driving.  
#
# Addressing these data issues involved a large number of augmentations
# to the data:
#
# 1.  Driving around track multiple times (4 times) to give enough data
#     for learning a good model.  With more examples the idea is that
#     the driving would be smoother.
#
# 2.  Corrective actions were added by pausing recording, driving to 
#     the right edge of the roadway, unpausing, and recording the
#     corrective action of turning left to bring the car back to 
#     the center of the roadway.   This was repeated on the left
#     edge of the roadway.   Then an entire lap was recorded by
#     alternating the corrective action on the right side, then
#     left side, etc.  This served to generate more dramatic steering
#     actions needed to teach the network how to recover back to
#     the center of the roadway above and beyond simple corrections
#     maintaining position once in the center of the roadway.
#     Doing an entire lap's worth of corrective actions alternating
#     between the right and left sides of the track generated 
#     data that teaches the network how to recover to the roadway
#     center in all parts of the track.
#
#  3. I turned around the car and drove it in the other direction
#       around the track.  This resulting in data exemplars consisting
#       of steering actions where the turns were mostly right hand turns.
#       I drove in the other direction for 4 laps.  I followed this with
#       a complete lap of alternating corrective actions going around
#       the track in the reverse direction.
#
#   4.  Separately in MATLAB I did an analysis of the steering angles in
#       the recorded data set.  I observed that the prior distribution over
#       large steering angles had very little probability mass.  This explained
#       why my trained model kept running of the side of the track.  To fix this,
#       using the log file in MATLAB, I produced a version of the data set that
#       contains the top 5% largest magnitude steering angles.  This was done
#       for the 5% most negative (less than 5-th percentile) and the 5% most
#       positive (greater than 95-th percentile).
#
#   5.  Asymmetric corrections.  When adding the steering actions associated with
#       the right hand camera, because it is closer to the right edge of the
#       track, the corrective action should be larger.  The reason from this is
#       because of the relationship between arch-length, turn radius, and angle
#       traversed.  The right hand side of the car must traverse a longer arch
#       length.  As such, when correcting, it has further to travel so the
#       correction shoud be larger so that the effect is to reduce the turn
#       radius more aggressively because it has further to travel.
#
#   6.  I cropped pixels at bottom of image to remove car dashboard and also
#       cropped from the top to eliminate the environmental features not
#       important to driving.
#
#   7.  I generally slowed down when driving so that more images are taken.
#
#   All of this is marked in sections in the code below.
##
import csv
import cv2
import numpy as np
import sklearn as sk

DEBUG_PRINT= False

IMROWS= 160
IMCOLS= 320
RGBDEPTH= 3
FLATTENED_SIZE= IMROWS * IMCOLS

####
# set size of batch to be 1/BATCH_SIZE_DIV of the
# training set and validation set
##
BATCH_SIZE_DIV= 50

EPOCHS=5

drop_prob=0.1

correction= 0.15

TOP_CROP=70
BOTTOM_CROP=10
LEFT_CROP=0
RIGHT_CROP=0

lines= []

lines_big= []

LENET= False

with open('./data/driving_log.csv') as csvfile:
  reader= csv.reader(csvfile)
  for line in reader:
    lines.append(line)
   
#####
# Separately in MATLAB, I isolated the data associated with the
# largest magitude steering actions.  This was done by taking the
# data whose steering action was either smaller than the 5-th
# percentile steering angle (most negative or smallest 5%) or
# greater than the 95-th percentile (most positive or largest
# 5%).
#
# read lines for top 5pct largest magnintued steering angles
#
# later these are oversampled to change the priors so that
# examples associated with larger magnitude steering angles
# are boosted.
##
with open('./data/driving_log_5_pct.csv') as csvfile:
   reader= csv.reader(csvfile)
   for line in reader:
     lines_big.append(line)


#####
# Compare the difference between the samples and the
# top 5% steering angle samples.  Whatever this difference,
# boost the proportion of top 5% steering angle examples by
# oversampling their entries.
#
# By doing this with the CSV file, it makes it easier to create
# the images as well as a single generator later.
###

num_lines= len(lines)

if DEBUG_PRINT:
  print("len(num_lines)= ", num_lines)

num_lines_big = len(lines_big)

if DEBUG_PRINT:
  print("len(num_lines_big)= ", num_lines_big)

num_samples = num_lines - num_lines_big

#####
# Randomly select 
##
if ((num_lines - num_lines_big) > 0):
   num_samples= int(round((num_lines - num_lines_big)*5/7))

   for i in range(num_samples):
     sel= np.random.randint(num_lines_big)
     line= lines_big[sel]
     lines.append(line)


   
images = []
measurements = []

augmented_images= []
augmented_measurements= []


  #####
  # Load the images for the data set. 
  #
  # Add the camera images for the left, right, and
  # center cameras
  #
  # index 0: center image
  # index 1: left image
  # index 2: right image
  ###
for line in lines:
  for i in range(3):
    source_path= line[i]
    filename = source_path.split('/')[-1]
    current_path = './data/IMG/' + filename
    image = cv2.imread(current_path)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    images.append(image)

  measurement = float(line[3])
  # get steering measurement
  #####
  # append steering measurement
  # then append synthesized steering measurement for left image
  # finally, append synthesized steering measurment for right image
  #
  # did not need correction here
  ##
  measurements.append(measurement)
  #measurements.append(measurement)
  #measurements.append(measurement)
  measurements.append(measurement + correction)
  measurements.append(measurement - (correction*1.0))



def generator(samples, batch_size=32):
  num_samples= len(samples)

  while 1:
    sk.utils.shuffle(samples)

    for offset in range(0, num_samples, batch_size):
      batch_samples= samples[offset:offset + batch_size]
      
      images= []
      measurements= []

      for batch_sample in batch_samples:
        for i in range(3):
          source_path= batch_sample[i]
          filename = source_path.split('/')[-1]
          current_path= './data/IMG/' + filename
          image= cv2.imread(current_path)
          image= cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
          images.append(image)

        measurement= float(batch_sample[3])
        measurements.append(measurement)
        measurements.append(measurement + correction)
        measurements.append(measurement - correction)
        #measurements.append(measurement + correction)
        #measurements.append(measurement - (correction *1.0))
       
      X_train= np.array(images)
      y_train= np.array(measurements)
   
      if DEBUG_PRINT:
        print("len(X_train)= ",len(X_train))
        print("len(y_train)= ",len(y_train))

      yield sk.utils.shuffle(X_train, y_train)
    


#####
# I ended up not needing the augmentation that
# was introduced by flipping the images.  Driving
# around the track in the other direction proved
# better results for me.  This especially because
# I did an entire corrective lap in the other direction.
##
DO_AUGMENTATION= False

if DO_AUGMENTATION:
   for image, measurements in zip(images, measurements):
      augmented_images.append(image)
      augmented_measurements.append(measurement)
      flipped_version_img= cv2.flip(image,1)
      flipped_version_meas= measurement * -1.0
      augmented_images.append(flipped_version_img)
      augmented_measurements.append(flipped_version_meas)
else:
   augmented_images = images
   augmented_measurements = measurements

X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Activation, Dropout, Lambda

from keras.layers.convolutional import Convolution2D, Convolution1D, Cropping2D
from keras.layers.pooling import MaxPooling2D

from sklearn.model_selection import train_test_split

train_samples, validation_samples = train_test_split(lines, test_size=0.2)

train_batch_size= int(round(len(train_samples)/BATCH_SIZE_DIV))
val_batch_size= int(round(len(validation_samples)/BATCH_SIZE_DIV))
train_generator = generator(train_samples, batch_size=train_batch_size)
validation_generator= generator(validation_samples, batch_size=val_batch_size);
#####
# Pixels of image were normalized (divide by 255) and mean centered (subtract 0.5).
# This is done in tensorflow/Keras so that it is efficient on GPU, but also means
# any image can be put into the model thus making it more flexible.
##
model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((TOP_CROP,BOTTOM_CROP),(LEFT_CROP,RIGHT_CROP))))


#####
# Can selecte LENET.  I experimented wiht it but chose not to use it
##
if LENET:
  model.add(Convolution2D(6,5,5,activation='relu'))
  model.add(MaxPooling2D())
  model.add(Convolution2D(16,5,5,activation='relu'))
  model.add(MaxPooling2D())
  model.add(Flatten())
  model.add(Dense(120))
  model.add(Dense(84))
  model.add(Dense(1))

#####
# I actually used the NVIDIA Driving network. I added a single MaxPool
# but used Dropout to regularize it.
##
else:
  model.add(Convolution2D(24,5,5,subsample=(2,2),activation='relu'))
  model.add(Convolution2D(36,5,5,subsample=(2,2),activation='relu'))
  model.add(Convolution2D(48,5,5,subsample=(2,2),activation='relu'))
  model.add(Convolution2D(64,3,3,activation='relu'))
  model.add(Convolution2D(64,3,3,activation='relu'))
  model.add(MaxPooling2D())
  model.add(Dropout(drop_prob))
  model.add(Flatten())
  model.add(Dense(100))
  model.add(Dropout(drop_prob))
  model.add(Dense(50))
  model.add(Dropout(drop_prob))
  model.add(Dense(10))
  model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch= len(train_samples), \
                    validation_data= validation_generator, nb_val_samples= len(validation_samples), nb_epoch=EPOCHS)

model.save('model.h5')
