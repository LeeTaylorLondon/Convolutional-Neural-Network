from matplotlib import pyplot as plt
from random import sample
from utils import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os


fd = 'Dataset/PetImages/'

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


train_test_split('Dataset/PetImages')

image_generator = ImageDataGenerator(rotation_range=30,
                                     width_shift_range=0.2,
                                     height_shift_range=0.2,
                                     zoom_range=0.2,
                                     horizontal_flip=True,
                                     fill_mode='nearest')

fig, ax = plt.subplots(2,3, figsize=(20,10))
# all_images = []


if __name__ == '__main__':
    cat_images = explore_data('Cat/')
    dog_images = explore_data('Dog/')
    pass