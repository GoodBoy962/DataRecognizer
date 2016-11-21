import numpy as np
import pandas as pa

import matplotlib.pyplot as plt
import matplotlib.cm as cm

data = pa.read_csv('data/test.csv')

images = data.iloc[:, 0:].values
images = images.astype(np.float)
images = np.multiply(images, 1.0 / 255.0)

IMAGE_TO_DISPLAY = 5737


## display image
def display(img):
    one_image = img.reshape(28, 28)
    plt.axis('off')
    plt.imshow(one_image, cmap=cm.binary)


display(images[IMAGE_TO_DISPLAY])
