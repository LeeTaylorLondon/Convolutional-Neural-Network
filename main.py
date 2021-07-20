from matplotlib import pyplot as plt
from random import sample
import os


def explore_cat_data():
    cat_fd = 'Dataset/PetImages/Cat/'
    _, _, cat_images = next(os.walk(cat_fd))

    fig, ax = plt.subplots(3,3, figsize=(20, 10))

    for idx, img in enumerate(sample(cat_images, 9)):
        img_read = plt.imread(cat_fd + img)
        ax[int(idx/3), idx%3].imshow(img_read)
        ax[int(idx/3), idx%3].axis('off')
        ax[int(idx/3), idx%3].set_title('Cat/'+img)
    plt.show()


if __name__ == '__main__':
    explore_cat_data()