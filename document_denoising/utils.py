"""
Utility functions
"""

from glob import glob
from tensorflow import pad
from tensorflow.keras.layers import Layer
from typing import List, Tuple

import cv2
import numpy as np


class ImageProcessor:
    """
    Class for processing, loading and saving document images
    """
    def __init__(self,
                 file_path_clean_images: str,
                 file_path_noisy_images: str,
                 n_channels: int = 0,
                 batch_size: int = 1,
                 image_resolution: tuple = None,
                 flip: bool = True,
                 crop: Tuple[Tuple[int, int], Tuple[int, int]] = None
                 ):
        """
        :param file_path_clean_images: str
            Complete file path of the clean images

        :param file_path_noisy_images: str
            Complete file path of the noisy images

        :param n_channels: int
            Number of channels of the image
                -> 0: gray
                -> 3: color (rgb)

        :param batch_size: int
            Batch size

        :param image_resolution: tuple
            Force resizing image into given resolution

        :param flip: bool
            Whether to flip image based on probability distribution or not

        :parm crop: Tuple[Tuple[int, int], Tuple[int, int]]
            Define image cropping
        """
        self.file_path_clean_images: str = file_path_clean_images
        self.file_path_noisy_images: str = file_path_noisy_images
        self.n_channels: int = cv2.IMREAD_COLOR if n_channels > 1 else cv2.IMREAD_GRAYSCALE
        self.batch_size: int = batch_size
        self.n_batches: int = 0
        self.image_resolution: tuple = image_resolution
        self.flip: bool = flip
        self.crop: Tuple[Tuple[int, int], Tuple[int, int]] = crop

    @staticmethod
    def _normalize(images: np.array) -> np.array:
        """
        Normalize images
        """
        return images / 127.5 - 1.0

    def _read_image(self, file_path: str) -> np.array:
        """
        Read image

        :param file_path: str
            Complete file path of the image to read

        :return np.array
            Image
        """
        _image: cv2 = cv2.imread(filename=file_path, flags=self.n_channels)
        if self.crop is not None:
            _iamge = _image[self.crop[0][0]:self.crop[0][1], self.crop[1][0]:self.crop[1][1]]
        if self.image_resolution is not None:
            _image = cv2.resize(src=_image, dsize=self.image_resolution, dst=None, fx=None, fy=None, interpolation=None)
        if self.flip:
            if np.random.random() > 0.5:
                if np.random.random() > 0.5:
                    _direction: int = 0
                else:
                    _direction: int = 1
                _image = cv2.flip(src=_image, flipCode=_direction, dst=None)
        return _image

    def load_batch(self) -> Tuple[np.array, np.array]:
        """
        Load batch images for each group (clean & noisy) separately

        :return Tuple[np.array, np.array]
            Arrays of clean & noisy images
        """
        _file_paths_clean: List[str] = glob(f'./{self.file_path_clean_images}/*')
        _file_paths_noisy: List[str] = glob(f'./{self.file_path_noisy_images}/*')
        self.n_batches = int(min(len(_file_paths_clean), len(_file_paths_noisy)) / self.batch_size)
        _total_samples: int = self.n_batches * self.batch_size
        _file_paths_clean_sample: List[str] = np.random.choice(_file_paths_clean, _total_samples, replace=False)
        _file_paths_noisy_sample: List[str] = np.random.choice(_file_paths_noisy, _total_samples, replace=False)
        for i in range(0, self.n_batches - 1, 1):
            _batch_clean: List[str] = _file_paths_clean_sample[i * self.batch_size:(i + 1) * self.batch_size]
            _batch_noisy: List[str] = _file_paths_noisy_sample[i * self.batch_size:(i + 1) * self.batch_size]
            _images_clean, _images_noisy = [], []
            for path_image_clean, path_image_noisy in zip(_batch_clean, _batch_noisy):
                _images_clean.append(self._read_image(file_path=path_image_clean))
                _images_noisy.append(self._read_image(file_path=path_image_noisy))
            yield self._normalize(np.array(_images_clean)), self._normalize(np.array(_images_noisy))

    def load_images(self) -> Tuple[List[str], np.array]:
        """
        Load images without batching

        :return Tuple[List[str], np.array]
            List of image file names & array of loaded images
        """
        _file_paths_noisy: List[str] = glob(f'./{self.file_path_noisy_images}/*')
        _images_noisy: List[str] = []
        for path_image_noisy in _file_paths_noisy:
            _images_noisy.append(self._read_image(file_path=path_image_noisy))
        yield _file_paths_noisy, self._normalize(np.array(_images_noisy))

    @staticmethod
    def save_image(image: np.array, output_file_path: str):
        """
        Save image

        :param image: np.array
            Image to save

        :parm output_file_path: str
            Complete file path of the output image
        """
        cv2.imwrite(filename=output_file_path, img=image, params=None)


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
