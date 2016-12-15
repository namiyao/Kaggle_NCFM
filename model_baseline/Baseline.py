
# coding: utf-8

# In[1]:

import os, random, glob
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt
from matplotlib import ticker
import seaborn as sns
get_ipython().magic(u'matplotlib inline')

from keras.models import Sequential
from keras.layers import Convolution2D, BatchNormalization, LeakyReLU, AveragePooling2D, Flatten, Dropout, Dense
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras import backend as K


# In[8]:

TRAIN_DIR = '../data/train/'
TEST_DIR = '../data/test_stg1/'
FISH_CLASSES = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']
ROWS = 256
COLS = 256
BatchSize=64


# In[3]:

#Loading data

import pickle
if os.path.exists('../data/data_train_{}_{}.pickle'.format(ROWS, COLS)):
    print ('Exist data_train_{}_{}.pickle. Loading data from file.'.format(ROWS, COLS))
    with open('../data/data_train_{}_{}.pickle'.format(ROWS, COLS), 'rb') as f:
        data_train = pickle.load(f)
    X_train = data_train['X_train']
    y_train = data_train['y_train']
else:
    print ('Loading data from original images. Generating data_train_{}_{}.pickle.'.format(ROWS, COLS))

    def get_images(fish):
        """Load files from train folder"""
        fish_dir = TRAIN_DIR+'{}'.format(fish)
        images = [fish+'/'+im for im in os.listdir(fish_dir)]
        return images

    def read_image(src):
        """Read and resize individual images"""
        im = Image.open(src)
        im = im.resize((COLS, ROWS), Image.BILINEAR)
        im = np.asarray(im)
        return im

    files = []
    y_train = []

    for fish in FISH_CLASSES:
        fish_files = get_images(fish)
        files.extend(fish_files)

        y_fish = np.tile(fish, len(fish_files))
        y_train.extend(y_fish)
        #print("{0} photos of {1}".format(len(fish_files), fish))

    y_train = np.array(y_train)
    X_train = np.ndarray((len(files), ROWS, COLS, 3), dtype=np.uint8)

    for i, im in enumerate(files): 
        X_train[i] = read_image(TRAIN_DIR+im)
        if i%1000 == 0: print('Processed {} of {}'.format(i, len(files)))

    #X_train = X_train / 255.
    #print(X_train.shape)

    # One Hot Encoding Labels
    y_train = LabelEncoder().fit_transform(y_train)
    y_train = np_utils.to_categorical(y_train)

    #save data to file
    data_train = {'X_train': X_train,'y_train': y_train }

    with open('../data/data_train_{}_{}.pickle'.format(ROWS, COLS), 'wb') as f:
        pickle.dump(data_train, f)

X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=22, stratify=y_train)


# In[10]:

#create model

optimizer = Adam(lr=1e-4)

def create_model():
    model = Sequential()

    model.add(Convolution2D(32, 3, 3, border_mode='same', dim_ordering='tf', input_shape=(ROWS, COLS, 3)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.33))
    model.add(Convolution2D(32, 3, 3, border_mode='same', dim_ordering='tf', subsample=(2, 2)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.33))
    
    model.add(Convolution2D(64, 3, 3, border_mode='same', dim_ordering='tf'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.33))
    model.add(Convolution2D(64, 3, 3, border_mode='same', dim_ordering='tf', subsample=(2, 2)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.33))

    model.add(Convolution2D(128, 3, 3, border_mode='same', dim_ordering='tf'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.33))
    model.add(Convolution2D(128, 3, 3, border_mode='same', dim_ordering='tf'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.33))
    model.add(Convolution2D(128, 3, 3, border_mode='same', dim_ordering='tf', subsample=(2, 2)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.33))
    
    model.add(Convolution2D(256, 3, 3, border_mode='same', dim_ordering='tf'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.33))
    model.add(Convolution2D(256, 3, 3, border_mode='same', dim_ordering='tf'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.33))
    model.add(Convolution2D(256, 3, 3, border_mode='same', dim_ordering='tf', subsample=(2, 2)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.33))
    
    model.add(Convolution2D(256, 3, 3, border_mode='same', dim_ordering='tf'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.33))
    model.add(Convolution2D(256, 3, 3, border_mode='same', dim_ordering='tf'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.33))
    model.add(Convolution2D(256, 3, 3, border_mode='same', dim_ordering='tf', subsample=(2, 2)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.33))
    
    model.add(AveragePooling2D(pool_size=(7, 7), dim_ordering='tf'))
    model.add(Flatten())
    model.add(Dense(len(FISH_CLASSES), activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


# In[5]:

#data preprocessing

train_datagen = ImageDataGenerator(
    #featurewise_center=True,
    #featurewise_std_normalization=True,
    rescale=1./255,
    rotation_range=180,
    shear_range=np.pi/6.,
    zoom_range=[1,1.1],
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=True)

#train_datagen.fit(X_train)
train_generator = train_datagen.flow(X_train, y_train, batch_size=BatchSize, shuffle=True, seed=22)

valid_datagen = ImageDataGenerator(rescale=1./255)

valid_generator = valid_datagen.flow(X_valid, y_valid, batch_size=BatchSize, shuffle=True, seed=22)


# In[11]:

#callbacks

early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=15, verbose=1, mode='auto')        

if not os.path.exists('./checkpoints'):
    os.mkdir('./checkpoints')
files = glob.glob('./checkpoints/*')
for f in files:
    os.remove(f)
model_checkpoint = ModelCheckpoint(filepath='./checkpoints/weights.{epoch:03d}-{val_loss:.4f}.hdf5', monitor='val_loss', verbose=1, save_best_only=False, save_weights_only=False, mode='auto')
        
learningrate_schedule = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, verbose=1, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)

tensorboard = TensorBoard(log_dir='./logs', histogram_freq=10, write_graph=True, write_images=True)


# In[12]:

#training model

model = create_model()
model.fit_generator(train_generator, samples_per_epoch=len(X_train), nb_epoch=300, verbose=1, 
                    callbacks=[early_stopping, model_checkpoint, learningrate_schedule, tensorboard], 
                    validation_data=valid_generator, nb_val_samples=len(X_valid), nb_worker=4, pickle_safe=True)


# In[ ]:

#test submission

test_files = [im for im in os.listdir(TEST_DIR)]
test = np.ndarray((len(test_files), ROWS, COLS, CHANNELS), dtype=np.uint8)

for i, im in enumerate(test_files): 
    test[i] = read_image(TEST_DIR+im) / 255.
    
test_preds = model.predict(x, batch_size=BatchSize, verbose=1)

submission = pd.DataFrame(test_preds, columns=FISH_CLASSES)
submission.insert(0, 'image', test_files)
submission.head()

