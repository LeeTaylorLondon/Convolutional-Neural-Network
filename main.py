from matplotlib import pyplot as plt
from random import sample
from utils import train_test_split
import os


cat_fd = 'Dataset/PetImages/Cat/'

def explore_cat_data():
    _, _, cat_images = next(os.walk(cat_fd))

    fig, ax = plt.subplots(3,3, figsize=(20, 10))

    for idx, img in enumerate(sample(cat_images, 9)):
        img_read = plt.imread(cat_fd + img)
        ax[int(idx/3), idx%3].imshow(img_read)
        ax[int(idx/3), idx%3].axis('off')
        ax[int(idx/3), idx%3].set_title('Cat/'+img)
    plt.show()


train_test_split(cat_fd)


if __name__ == '__main__':
    # explore_cat_data()
    pass