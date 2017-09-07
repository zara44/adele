import os
import shutil
import random
import numpy as np
import pandas as pd

path = 'C:\\workspace\\kaggle\\distracted_driver\\data\\'

# stratify split
def split_train_valid_set(path):
    for categ in os.listdir(os.path.join(path,'train')):
        print(categ)
        ls = os.listdir(os.path.join(os.path.join(path ,'train'), categ))
        ls = random.sample(ls, int(len(ls)*0.2))
        for f in ls:
            print(f)
            shutil.move(os.path.join(os.path.join(os.path.join(path ,'train'), categ), f),os.path.join(os.path.join(os.path.join(path ,'valid'), categ), f))
    return

def generate_sample_dataset(path,foo):
    # foo = 'test'
    for categ in os.listdir(os.path.join(path,foo)):
        print(categ)
        ls = os.listdir(os.path.join(os.path.join(path ,foo), categ))
        ls = random.sample(ls, int(len(ls)*0.05))
        for f in ls:
            print(f)
            shutil.move(os.path.join(os.path.join(os.path.join(path ,foo), categ), f),os.path.join(os.path.join(os.path.join(os.path.join(path,'sample') ,foo), categ), f))
    return

from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
from keras.optimizers import SGD, Adadelta, Adagrad, Adam
from keras import utils
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D

# image generators

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
        os.path.join(os.path.join(path,'sample'),'test'),
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary')

train_datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)
train_generator = train_datagen.flow_from_directory(
        os.path.join(os.path.join(path,'sample'),'train'),
        target_size=(224, 224),
        batch_size=32)

valid_datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)
valid_generator = train_datagen.flow_from_directory(
        os.path.join(os.path.join(path,'sample'),'valid'),
        target_size=(224, 224),
        batch_size=32)


# vgg model
vgg_model = VGG16(include_top=True, weights=None, input_tensor=None, input_shape=None, pooling=None, classes=1000)
vgg_path = 'C:\\workspace\\model_weights'
vgg_model.load_weights(os.path.join(vgg_path,'vgg_imagenet_weights.h5'))

vgg_model.compile(optimizer=Adam(lr=0.001),loss='categorical_crossentropy', metrics=['accuracy'])


# commands for pop
# vgg_model.layers.pop()

# define vgg layers and then set the weights


input_shape = (224, 224, 3)

vgg_model2 = Sequential([
    Conv2D(64, (3, 3), input_shape=input_shape, padding='same',
           activation='relu'),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    Conv2D(128, (3, 3), activation='relu', padding='same',),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Conv2D(256, (3, 3), activation='relu', padding='same',),
    Conv2D(256, (3, 3), activation='relu', padding='same',),
    Conv2D(256, (3, 3), activation='relu', padding='same',),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Conv2D(512, (3, 3), activation='relu', padding='same',),
    Conv2D(512, (3, 3), activation='relu', padding='same',),
    Conv2D(512, (3, 3), activation='relu', padding='same',),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Conv2D(512, (3, 3), activation='relu', padding='same',),
    Conv2D(512, (3, 3), activation='relu', padding='same',),
    Conv2D(512, (3, 3), activation='relu', padding='same',),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Flatten(),
    Dense(4096, activation='relu'),
    Dense(4096, activation='relu'),
    Dense(1000, activation='softmax')
])

vgg_model2.summary()

vgg_model2.load_weights(os.path.join(vgg_path,'vgg_imagenet_weights.h5'))

vgg_model2.pop()

for layer in vgg_model2.layers: layer.trainable = False
vgg_model2.add(Dense(10, activation='softmax'))
vgg_model2.compile(optimizer=Adam(lr=0.001),loss='categorical_crossentropy', metrics=['accuracy'])

vgg_model2.fit_generator(train_generator,
                         steps_per_epoch=int(np.floor(train_generator.n/train_generator.batch_size)),
                         epochs=1,
                         validation_data=valid_generator,
                         validation_steps=int(np.floor(valid_generator.n/valid_generator.batch_size)))

result2 = vgg_model2.predict_generator(test_generator, int(np.floor(test_generator.n/test_generator.batch_size)))

vgg_model2.save_weights(os.path.join(vgg_path,'dd_vgg_localtrain.h5'))