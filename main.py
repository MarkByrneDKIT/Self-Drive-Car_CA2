import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
import random
from PIL import Image
import cv2
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
import pandas as pd
import cv2
import os
import ntpath
from matplotlib import image as mpimg
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from imgaug import augmenters as iaa

# Get the file name from the path
def path_leaf(path):
    head, tail = ntpath.split(path) # split the path
    return tail     # return the tail

#preprocess the image
def preprocess_img(img):
    img = mpimg.imread(img)
    img = img[60:135, :, :] # crop the image
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)  # convert to YUV
    img = cv2.GaussianBlur(img, (3, 3), 0)  # blur the image
    img = cv2.resize(img, (200, 66))    # resize the image
    img = img/255   # normalize the image
    return img  # return the preprocessed image


def preprocess_img_no_imread(img):
    img = img[60:135, :, :]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img/255
    return img


# zoom the image
def zoom(image_to_zoom):
    zoom_func = iaa.Affine(scale=(1, 1.3))  # zoom from 100% (no zoom) to 130%
    z_image = zoom_func.augment_image(image_to_zoom)    # apply the zoom
    return z_image  # return the zoomed image

# pan the image
def pan(image_to_pan):
    pan_func = iaa.Affine(translate_percent={"x":(-0.1, 0.1), "y":(-0.1, 0.1)}) # pan left-right by 10% and up-down by 10%
    pan_image = pan_func.augment_image(image_to_pan)    # apply the pan
    return pan_image    # return the panned image

# random brightness
def img_random_brightness(image_to_brighten):
    bright_func = iaa.Multiply((0.2, 1.2))  # change brightness, doesn't affect dark images much
    bright_image = bright_func.augment_image(image_to_brighten).astype("uint8")   # apply the brightness
    return bright_image # return the brightened image

# random flip
def img_random_flip(image_to_flip, steering_angle):
    # 0 - flip horizontal, 1 flip vertical, -1 combo of both
    flipped_image = cv2.flip(image_to_flip, -1) # flip the image
    steering_angle = -steering_angle    # negate the steering angle
    return flipped_image, steering_angle    # return the flipped image and the negated steering angle

# random augment
def random_augment(image_to_augment, steering_angle):
    augment_image = mpimg.imread(image_to_augment)  # read the image
    if np.random.rand() < 0.5:  # randomly flip the image
        augment_image = zoom(augment_image).astype("uint8") # zoom the image
    if np.random.rand() < 0.5:  # randomly flip the image
        augment_image = pan(augment_image).astype("uint8")  # pan the image
    # if np.random.rand() < 0.5:
    #     augment_image = img_random_brightness(augment_image)
    if np.random.rand() < 0.5:  # randomly flip the image
        augment_image, steering_angle = img_random_flip(augment_image, steering_angle)  # flip the image
        augment_image = augment_image.astype("uint8")   # convert the image to uint8
    return augment_image, steering_angle    # return the augmented image and the steering angle

# Load the image and steering angle
def load_steering_img(datadir, df): 
    image_path = [] # empty list to store the image path
    steering = []   # empty list to store the steering angle
    for i in range(len(df)):    # loop through the dataframe
        indexed_data = data.iloc[i] # get the indexed data
        centre, left, right = indexed_data[0], indexed_data[1], indexed_data[2] # get the image path
        image_path.append(os.path.join(datadir, centre.strip()))    # append the image path to the list
        steering.append(float(indexed_data[3])) # append the steering angle to the list
    image_paths = np.array(image_path)  # convert the image path list to numpy array
    steerings = np.array(steering)  # convert the steering angle list to numpy array
    return image_paths, steerings   # return the image path and steering angle

# Batch generator
def batch_generator(image_paths, steering_ang, batch_size, is_training):
    while True: # loop forever
        batch_img = []  # empty list to store the image
        batch_steering = [] # empty list to store the steering angle
        for i in range(batch_size): # loop through the batch size
            random_index = random.randint(0, len(image_paths)-1)    # get a random index
            if is_training: # if training
                im, steering = random_augment(image_paths[random_index], steering_ang[random_index])    # augment the image
            else:   # if not training
                im = mpimg.imread(image_paths[random_index])    # read the image
                steering = steering_ang[random_index]   # get the steering angle

            im = preprocess_img_no_imread(im) # preprocess the image
            batch_img.append(im)    # append the image to the list
            batch_steering.append(steering) # append the steering angle to the list
        yield np.asarray(batch_img), np.asarray(batch_steering)   # return the batch image and steering angle

# Model
def nvidia_model():
    model = Sequential()
    model.add(Convolution2D(24, (5, 5), strides=(2, 2), input_shape=(66, 200, 3), activation='elu'))
    model.add(Convolution2D(36, (5, 5), strides=(2, 2), activation='elu'))
    model.add(Convolution2D(48, (5, 5), strides=(2, 2), activation='elu'))
    model.add(Convolution2D(64, (3, 3), activation='elu'))
    model.add(Convolution2D(64, (3, 3), activation='elu'))
    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))
    optimizer = Adam(learning_rate=0.0001)
    model.compile(loss='mse', optimizer=optimizer)
    return model


print("STARTING UP...")
# Read in the Data from the driving_log.csv file
datadir = "E:\\CA2\\CA2\\Recordings\\Track_1_V4"  # Path to the data folder
columns = ['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed']   # Column names for the data
data = pd.read_csv(os.path.join(datadir, 'driving_log.csv'), names = columns)   # Read in the data
pd.set_option('display.max_columns', 7) # Display all columns

print("Num of Center: ", data['center'])


data['center'] = data['center'].apply(path_leaf)    # Get the file name for the center image
data['left'] = data['left'].apply(path_leaf)    # Get the file name for the left image
data['right'] = data['right'].apply(path_leaf)  # Get the file name for the right image


num_bins = 25   
samples_per_bin = 200   
hist, bins = np.histogram(data['steering'], num_bins)   
centre = (bins[:-1] + bins[1:])*0.5  
plt.bar(centre, hist, width=0.05)  
plt.plot((np.min(data['steering']), np.max(data['steering'])), (samples_per_bin, samples_per_bin))  
plt.show()  # Show the plot

remove_list=[]  # 
print('Total data: ', len(data))    

for j in range(num_bins):  
    list_ = []  
    for i in range(len(data['steering'])): 
        if bins[j] <= data['steering'][i] <= bins[j+1]: #    
            list_.append(i) 
    list_ = shuffle(list_) 
    list_ = list_[samples_per_bin:] 
    remove_list.extend(list_)   

print("Removed Images: ", len(remove_list))  # Print the number of images to remove
data.drop(data.index[remove_list], inplace=True)    # Drop the images from the data
print("Remaining Images: ", len(data)) # Print the number of remaining images

hist, bins = np.histogram(data['steering'], num_bins)   
plt.bar(centre, hist, width=0.05)  
plt.plot((np.min(data['steering']), np.max(data['steering'])), (samples_per_bin, samples_per_bin)) 
plt.show()

image_paths, steerings = load_steering_img(datadir+'/IMG', data)    # Load the images and steering angles
X_train, X_valid, y_train, y_valid = train_test_split(image_paths, steerings, test_size=0.2, random_state=6)    # Split the data into training and validation sets
print(f"Training samples {len(X_train)}\n Validation samples {len(X_valid)}")   # Print the number of training and validation samples

fig, axes = plt.subplots(1,2, figsize=(12,4))   
axes[0].hist(y_train, bins=num_bins, width=0.05, color='blue')  
axes[0].set_title("Training set")   
axes[1].hist(y_valid, bins=num_bins, width=0.05, color='red')
axes[1].set_title("Validation set")
plt.show()

image = image_paths[100]
original_image = mpimg.imread(image)
preprocessed_image = preprocess_img(image)
fig, axes = plt.subplots(1,2, figsize=(15,10))
fig.tight_layout()
axes[0].imshow(original_image)
axes[0].set_title("Original image")
axes[1].imshow(preprocessed_image)
axes[1].set_title("Preprocessed image")
plt.show()

#X_train = np.array(list(map(preprocess_img, X_train)))
#X_valid = np.array(list(map(preprocess_img, X_valid)))

# plt.imshow(X_train[random.randint(0, len(X_train)-1)])
# plt.axis('off')
# plt.show()
# print(X_train.shape)

image = image_paths[random.randint(0, 1000)]    
original_image = mpimg.imread(image)   
zoomed_image = zoom(original_image) 
fig, axs = plt.subplots(1, 2, figsize=(15, 10)) 
fig.tight_layout()
axs[0].imshow(original_image)
axs[0].set_title("Original Image")
axs[1].imshow(zoomed_image)
axs[1].set_title("Zoomed Image")
plt.show()

image = image_paths[random.randint(0, 1000)]
original_image = mpimg.imread(image)
panned_image = pan(original_image)
fig, axs = plt.subplots(1, 2, figsize=(15, 10))
fig.tight_layout()
axs[0].imshow(original_image)
axs[0].set_title("Original Image")
axs[1].imshow(panned_image)
axs[1].set_title("Panned Image")
plt.show()

image = image_paths[random.randint(0, 1000)]
original_image = mpimg.imread(image)
bright_image = img_random_brightness(original_image)
fig, axs = plt.subplots(1, 2, figsize=(15, 10))
fig.tight_layout()
axs[0].imshow(original_image)
axs[0].set_title("Original Image")
axs[1].imshow(bright_image)
axs[1].set_title("Bright Image")
plt.show()

random_index = random.randint(0, 1000)
image = image_paths[random_index]
steering_angle = steerings[random_index]
original_image = mpimg.imread(image)
flipped_image, flipped_angle = img_random_flip(original_image, steering_angle)
fig, axs = plt.subplots(1, 2, figsize=(15, 10))
fig.tight_layout()
axs[0].imshow(original_image)
axs[0].set_title("Original Image - " + "Steering Angle: " + str(steering_angle))
axs[1].imshow(flipped_image)
axs[1].set_title("Flipped Image"+ "Steering Angle: " + str(flipped_angle))
plt.show()

ncols = 2
nrows = 10
fig, axs = plt.subplots(nrows, ncols, figsize=(15, 50))
fig.tight_layout()
for i in range(10):
    rand_num = random.randint(0, len(image_paths) - 1)
    random_image = image_paths[rand_num]
    random_steering = steerings[rand_num]
    original_image = mpimg.imread(random_image)
    augmented_image, steering_angle = random_augment(random_image, random_steering)
    axs[i][0].imshow(original_image)
    axs[i][0].set_title("Original Image")
    axs[i][1].imshow(augmented_image)
    axs[i][1].set_title("Augmented Image")
plt.show()

model = nvidia_model()
print(model.summary())

# train 300 40 times, 300  validate 200 steps generate 200 images
history = model.fit(batch_generator(X_train, y_train, 200, 1), steps_per_epoch=160, epochs=26, validation_data=batch_generator(X_valid, y_valid, 160, 0), validation_steps=200, verbose=1, shuffle=1)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(["Training", "Validation"])
#plt.ylim(0, 1)
plt.title('Loss')
plt.xlabel("Epoch")
plt.show()




model.save('model.h5')
print("Model saved successfully")

