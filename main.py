from matplotlib import pyplot as plt
from random import sample
from utils import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dropout, Flatten, Dense
import os


fd = 'Dataset/PetImages/'

FILTER_SIZE = 3
NUM_FILTERS = 32
INPUT_SIZE = 32
BATCH_SIZE = 16
MAXPOOL_SIZE = 2
STEPS_PER_EPOCH = 20000 // BATCH_SIZE
EPOCHS = 10

image_generator = ImageDataGenerator(rotation_range=30,
                                     width_shift_range=0.2,
                                     height_shift_range=0.2,
                                     zoom_range=0.2,
                                     horizontal_flip=True,
                                     fill_mode='nearest')

def explore_data(folder, verbose=True):
    _, _, images = next(os.walk(fd + folder))
    if verbose:
        fig, ax = plt.subplots(3,3, figsize=(20, 10))

        for idx, img in enumerate(sample(images, 9)):
            img_read = plt.imread(fd + folder + img)
            ax[int(idx/3), idx%3].imshow(img_read)
            ax[int(idx/3), idx%3].axis('off')
            ax[int(idx/3), idx%3].set_title(folder+img)
        plt.show()
    return images

def data_preprocessing():
    training_data_generator = ImageDataGenerator(rescale=1. / 255)
    training_set = training_data_generator. \
        flow_from_directory('Dataset/PetImages/Train/',
                            target_size=(INPUT_SIZE, INPUT_SIZE),
                            batch_size=BATCH_SIZE,
                            class_mode='binary')
    return training_set

def build_model():
    model = Sequential()

    model.add(Conv2D(NUM_FILTERS, (FILTER_SIZE, FILTER_SIZE),
                     input_shape=(INPUT_SIZE, INPUT_SIZE, 3),
                     activation='relu'))
    model.add(MaxPooling2D(pool_size = (MAXPOOL_SIZE, MAXPOOL_SIZE)))
    model.add(MaxPooling2D(pool_size=(MAXPOOL_SIZE, MAXPOOL_SIZE)))
    model.add(Flatten())
    model.add(Dense(units=128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy'])





if __name__ == '__main__':
    """ Helper function to split data in folder, run just once """
    # train_test_split('Dataset/PetImages/')
    """ Data exploration functions to view random images in the folders """
    # cat_images = explore_data('Cat/')
    # dog_images = explore_data('Dog/')
    pass