import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import random
from skimage import exposure
from skimage.util import random_noise
from skimage import transform
from cv2 import resize

def main():

    img = mpimg.imread('./pexels-shanekell-1363873.jpg')
    plt.imshow(img)
    plt.show()

    # RESCALING
    img_rescale = resize(img, (500,350))
    plt.imshow(img_rescale)
    plt.show()

    # FLIPPING
    horizontal_flip = np.fliplr(img)
    plt.imshow(horizontal_flip)
    plt.show()

    vertical_flip = np.flipud(img)
    plt.imshow(vertical_flip)
    plt.show()

    # ROTATION
    trans_img = transform.rotate(img, random.uniform(-40,40))
    plt.imshow(trans_img)
    plt.show()

    # ADDING NOISE
    img_noise = random_noise(img, mode="s&p", clip=True)
    plt.imshow(img_noise)
    plt.show()

if __name__ == "__main__":
    main()