"""
Utility functions
"""

from PIL import Image
from tensorflow import pad
from tensorflow.keras.layers import Layer

import numpy as np


class ImageProcessor:
    """
    Class for processing, loading and saving document images
    """
    def __init__(self,
                 file_path_clean
                 ):
        self.dataset_name = dataset_name
        self.img_res = img_res

    def load_data(self, domain, batch_size=1, is_testing=False):
        data_type = "train%s" % domain if not is_testing else "test%s" % domain
        path = glob('./%s/%s/*' % (self.dataset_name, data_type))

        batch_images = np.random.choice(path, size=batch_size)

        imgs = []
        for img_path in batch_images:
            img = self.imread(img_path)
            #if not is_testing:
                #img = resize(img, self.img_res)

                #if np.random.random() > 0.5:
                #    img = np.fliplr(img)
            #else:
            #    img = resize(img, self.img_res)
            imgs.append(img)

        imgs = np.array(imgs) / 127.5 - 1.

        return imgs

    def load_batch(self, batch_size=1, is_testing=False):
        data_type = "train" if not is_testing else "test"
        path_A = glob('./%s/%sA/*' % (self.dataset_name, data_type))
        path_B = glob('./%s/%sB/*' % (self.dataset_name, data_type))

        self.n_batches = int(min(len(path_A), len(path_B)) / batch_size)
        total_samples = self.n_batches * batch_size

        # Sample n_batches * batch_size from each path list so that model sees all
        # samples from both domains
        path_A = np.random.choice(path_A, total_samples, replace=False)
        path_B = np.random.choice(path_B, total_samples, replace=False)

        for i in range(self.n_batches-1):
            batch_A = path_A[i*batch_size:(i+1)*batch_size]
            batch_B = path_B[i*batch_size:(i+1)*batch_size]
            imgs_A, imgs_B = [], []
            for img_A, img_B in zip(batch_A, batch_B):
                img_A = self.imread(img_A)
                img_B = self.imread(img_B)

                #img_A = resize(img_A, self.img_res)
                #img_B = resize(img_B, self.img_res)

                #if not is_testing and np.random.random() > 0.5:
                        #img_A = np.fliplr(img_A)
                        #img_B = np.fliplr(img_B)

                imgs_A.append(img_A)
                imgs_B.append(img_B)

            imgs_A = np.array(imgs_A) / 127.5 - 1.
            imgs_B = np.array(imgs_B) / 127.5 - 1.

            yield imgs_A, imgs_B

    def load_img(self, path):
        img = self.imread(path)
        img = img / 127.5 - 1.
        return img[np.newaxis, :, :, :]

    def imread(self, path):
        return np.array(Image.open(path).convert('L').resize(size=self.img_res))


class ConstantPadding2D(Layer):
    """
    Class for building constant padding layer in Keras / TensorFlow
    """
    def __init__(self, padding: tuple = (1, 1), constant: int = 0, **kwargs):
        """
        :param padding: tuple
            Padding size

        :param constant: int
            Constant value

        :param kwargs: dict
            Key-word arguments for configuring neural network layer in keras
        """
        self.padding = tuple(padding)
        self.constant = constant
        super(ConstantPadding2D, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape) -> tuple:
        """
        Compute output shape

        :param input_shape: tuple
            Image input shape

        :return tuple
            3-dimensional output shape
        """
        return (input_shape[0], input_shape[1] + 2 * self.padding[0], input_shape[2] + 2 * self.padding[1], input_shape[3])

    def call(self, input_tensor, mask=None):
        """
        Apply constant padding when calling

        :param input_tensor:
            Input tensor

        :param mask:

        """
        _padding_width, _padding_height = self.padding
        return pad(input_tensor, [[0, 0], [_padding_height, _padding_height], [_padding_width, _padding_width], [0, 0]], mode='CONSTANT', constant_values=self.constant)


class ReflectionPadding2D(Layer):
    """
    Class for building reflection padding layer in Keras / TensorFlow
    """
    def __init__(self, padding: tuple = (1, 1), **kwargs):
        """
        :param padding: tuple
            Padding size

        :param kwargs: dict
            Key-word arguments for configuring neural network layer in keras
        """
        self.padding = tuple(padding)
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape: tuple) -> tuple:
        """
        Compute output shape

        :param input_shape: tuple
            Image input shape

        :return tuple
            3-dimensional output shape
        """
        return (input_shape[0], input_shape[1] + 2 * self.padding[0], input_shape[2] + 2 * self.padding[1], input_shape[3])

    def call(self, input_tensor, mask=None):
        """
        Apply reflection padding when calling

        :param input_tensor:
            Input tensor

        :param mask:

        """
        _padding_width, _padding_height = self.padding
        return pad(input_tensor, [[0, 0], [_padding_height, _padding_height], [_padding_width, _padding_width], [0, 0]], 'REFLECT')


class ReplicationPadding2D(Layer):
    """
    Class for building replication padding layer in Keras / TensorFlow
    """
    def __init__(self, padding: tuple = (1, 1), **kwargs):
        """
        :param padding: tuple
            Padding size

        :param kwargs: dict
            Key-word arguments for configuring neural network layer in keras
        """
        self.padding = tuple(padding)
        super(ReplicationPadding2D, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape) -> tuple:
        """
        Compute output shape

        :param input_shape: tuple
            Image input shape

        :return tuple
            3-dimensional output shape
        """
        return (input_shape[0], input_shape[1] + 2 * self.padding[0], input_shape[2] + 2 * self.padding[1], input_shape[3])

    def call(self, input_tensor, mask=None):
        """
        Apply replication padding when calling

        :param input_tensor:
            Input tensor

        :param mask:

        """
        _padding_width, _padding_height = self.padding
        return pad(input_tensor, [[0, 0], [_padding_height, _padding_height], [_padding_width, _padding_width], [0, 0]], 'SYMMETRIC')
