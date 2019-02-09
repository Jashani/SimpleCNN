# Convolutional Neural Network

# Installing Theano Tensorflow Keras

# Part 1 - Building the CNN

# Importing libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Convolution2D(32, (3, 3), input_shape = (256, 256, 3), activation = 'relu'))

# Step 2 -  Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 2.5 - ANOTHER LAYER
classifier.add(Convolution2D(64, (5, 5), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full Connection
classifier.add(Dense(activation="relu", units=128))
classifier.add(Dropout(p = 0.3))
classifier.add(Dense(activation="sigmoid", units=1))

# Compiling the CNN
classifier.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])

# Part 2 - Fitting the CNN to the images
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.25,
        zoom_range=0.25,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(256, 256),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(256, 256),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(
        training_set,
        steps_per_epoch=(8000/32),
        epochs=25,
        validation_data=test_set,
        validation_steps=(2000/32))

# Single prediction
from keras.preprocessing import image
import numpy as np
from keras.models import load_model
classifier = load_model('Nice.h5')
test_image = image.load_img('dataset/single_prediction/cat_or_dog_2.png', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
test_image = test_image/255.0  
result = classifier.predict(test_image)
if result[0][0] > 0.5:
    prediction = 'dog'
    print ((result[0][0] - 0.5) * 200, "% dog.")
else:
    prediction = 'cat'
    print ((0.5 - result[0][0]) * 200, "% cat.")