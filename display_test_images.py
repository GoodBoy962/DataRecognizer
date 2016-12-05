import numpy as np
import pandas as pa

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pip._vendor.distlib.compat import raw_input

data = pa.read_csv('data/test.csv')

images = data.iloc[:, 0:].values
images = images.astype(np.float)
images = np.multiply(images, 1.0 / 255.0)


## display image
def display():
    one_image = images[25997].reshape(28, 28)
    plt.axis('off')
    plt.imshow(one_image, cmap=cm.binary)


# while True:
    # i = raw_input("Enter image id, to display")
# display(images[25999])
# i = raw_input("next digit")

display()
