#Import libraries
import csv
import cv2
import matplotlib.pyplot as plt
import random
import math
import numpy as np
import sklearn

lines = []
#Read the csv file which contains the driving data from Udacity
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    #eliminate the information of first row
    for index, line in enumerate(reader):
        if index==0:
            continue
        else:
            lines.append(line)
print (len(lines))

#Read the csv file which contains the driving data of counter-clockwise lap
with open('data/driving_reverse.csv') as csvfile:
    reader = csv.reader(csvfile)
    for index, line in enumerate(reader):
        lines.append(line)
print (len(lines))

#Read the csv file which contains the driving data of recovery lap
with open('data/driving_recovery.csv') as csvfile:
    reader = csv.reader(csvfile)
    for index, line in enumerate(reader):
        lines.append(line)
print (len(lines))

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
train_samples, validation_samples = train_test_split(lines, test_size=0.2)

def generator(samples, correction=0.2, batch_size=32, threshold=0.6):
    num_samples = len(samples)
    print (num_samples)
    while True: # Loop forever so the generator never terminates
        shuffle(samples)
        last_batch = num_samples//batch_size
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                steering_angle = float(batch_sample[3])
                file_paths = ['data/IMG/'+batch_sample[i].split('/')[-1] for i in range(3)]
                
                #import all the images from three cameras
                for file_path in file_paths:
                    img = cv2.imread(file_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)    #Convert the colorspace from BGR to RGB
                    images.append(img)
                    images.append(cv2.flip(img,1))      #Collect more data by flipping the image

                #Low Pass filter for steering angle to prevent from too much jitter
                if math.fabs(steering_angle) < threshold:
                    pass
                else:
                    if steering_angle>0:
                        steering_angle = threshold
                    else:
                        steering_angle = -threshold
                         
                #Steering angles from center camera
                angles.append(steering_angle)
                angles.append(-steering_angle)  #Steering angles should be opposed for flipped images
                #Steering angles from left camera
                angles.append(steering_angle+correction)
                angles.append(-(steering_angle+correction))
                #Steering angles from right camera
                angles.append(steering_angle-correction)
                angles.append(-(steering_angle-correction))

            #Convert data type from list to numpy.array
            X_data = np.array(images)
            y_data = np.array(angles)
            yield shuffle(X_data, y_data)

# next(train_generator)
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

#resize the original image from 160x320x3 to 32x32x3 and then normalize it
import keras
import gc
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator

# datagen = ImageDataGenerator(vertical_flip=True)
# datagen.fit(X_data)

def resize(x):
    from keras.backend import tf as ktf
    return ktf.image.resize_images(x, (64, 64))

model = Sequential()
model.add(Lambda(resize, input_shape=(160,320,3), output_shape=(64,64,3)))
model.add(Lambda(lambda x: x/255.0 - 0.5))
model.add(Cropping2D(cropping=((20,10), (0,0))))

#simple testing network
# model.add(Flatten())
# model.add(Dense(1))

#LeNet 
# model.add(Convolution2D(6, 5,5, activation='relu'))
# model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Convolution2D(16, 5,5, activation='relu'))
# model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Flatten())
# model.add(Dense(120, activation='relu'))
# # model.add(Dropout(0.5))
# model.add(Dense(84, activation='relu'))
# # model.add(Dropout(0.5))
# model.add(Dense(1))


#NVIDIA Network
model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))  #30x30x24
model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu'))  #13x13x36
model.add(Convolution2D(48,3,3, activation='relu'))  #
model.add(Convolution2D(64,2,2, activation='relu'))  #
model.add(Convolution2D(64,2,2, activation='relu'))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(1))
          
model.compile(loss='mse', optimizer='adam')

history_object = model.fit_generator(train_generator, samples_per_epoch=6*len(train_samples), validation_data=validation_generator,nb_val_samples=len(validation_samples), nb_epoch=10)

print (history_object)
model.save('model.h5')
gc.collect()

#Plot the loss 
plt.subplot(1,1,1)
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')

plt.show()