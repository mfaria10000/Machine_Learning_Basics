import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
learn = tf.contrib.learn

mnist = learn.datasets.load_dataset('mnist')

train_data = mnist.train.images
train_lables = np.asarray(mnist.train.labels,dtype=np.int32)

test_data = mnist.test.images
test_labels = np.asarray(mnist.test.labels,dtype=np.int32)

img = test_data[1]
plt.title('Image index: %d | label: %d ' % (1, test_labels[1]))
plt.imshow(img.reshape((28, 28)), cmap=plt.cm.gray_r)
plt.show()
