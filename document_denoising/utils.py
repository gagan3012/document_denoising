"""
Utility functions
"""

from glob import glob
from tensorflow import pad
from tensorflow.keras.layers import Layer
from typing import List, Tuple

import cv2
import json
import math
import numpy as np
import os
import random


class ImageProcessor:
    """
    Class for processing, loading and saving document images
    """
    def __init__(self,
                 file_path_clean_images: str,
                 file_path_noisy_images: str,
                 n_channels: int = 1,
                 batch_size: int = 1,
                 image_resolution: tuple = None,
                 flip: bool = True,
                 crop: Tuple[Tuple[int, int], Tuple[int, int]] = None,
                 file_type: str = None
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

        :param file_type: str
            Specific image file type
        """
        self.file_path_clean_images: str = file_path_clean_images
        self.file_path_noisy_images: str = file_path_noisy_images
        self.file_type: str = '' if file_type is None else file_type
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
        _file_paths_clean: List[str] = glob(os.path.join('.', self.file_path_clean_images, f'*{self.file_type}'))
        _file_paths_noisy: List[str] = glob(os.path.join('.', self.file_path_noisy_images, f'*{self.file_type}'))
        #_file_paths_clean: List[str] = glob(f'./{self.file_path_clean_images}/*')
        #_file_paths_noisy: List[str] = glob(f'./{self.file_path_noisy_images}/*')
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

        :param output_file_path: str
            Complete file path of the output image
        """
        cv2.imwrite(filename=output_file_path, img=image, params=None)


class NoiseGenerator:
    """
    Class for generating noisy document images used in model training
    """
    def __init__(self,
                 file_path_input_clean_images: str,
                 file_path_output_noisy_images: str,
                 file_type: str = None
                 ):
        """
        :param file_path_input_clean_images: str
            Complete file path of clean input images

        :param file_path_output_noisy_images: str
            Complete file path of noisy output images

        :param file_type: str
            Specific image file type
        """
        self.file_path_input_clean_images: str = file_path_input_clean_images
        self.file_path_output_noisy_images: str = file_path_output_noisy_images
        self.file_type: str = '' if file_type is None else file_type
        self.labeling: dict = dict(file_name=[], label=[], label_encoding=[])
        self.image_file_names: List[str] = glob(os.path.join(self.file_path_input_clean_images, f'*{self.file_type}'))

    def blur(self, blur_type: str = 'average', kernel_size: tuple = (3, 3)):
        """
        Generate blurred noise images

        :param blur_type: str
            Blur type name
                -> average: Average
                -> median: Median

        :param kernel_size: tuple
            Kernel size
        """
        _blur_type: str = blur_type if blur_type in ['average', 'median'] else 'average'
        for image in self.image_file_names:
            _file_name: str = image.split('/')[-1]
            _file_type: str = _file_name.split('.')[-1]
            _image = cv2.imread(filename=image)
            for (k_x, k_y) in kernel_size:
                if _blur_type == 'average':
                    _blurred = cv2.blur(image, (k_x, k_y))
                elif _blur_type == 'median':
                    _blurred = cv2.medianBlur(image, k_x)
                else:
                    _blurred = None
                self.labeling['file_name'].append(os.path.join(self.file_path_output_noisy_images,
                                                               _file_name.split('.')[0],
                                                               '_blur',
                                                               _file_type
                                                               )
                                                  )
                self.labeling['label'].append('blur')
                self.labeling['label_encoding'].append('1')
                cv2.imwrite(filename=self.labeling['file_name'][-1], img=_blurred)

    def fade(self, brightness: int = 20, contrast: int = 20):
        """
        Generate faded noise images

        :param brightness: int
            Desired brightness change

        :param contrast: int
            Desired contrast change
        """
        _brithness: float = brightness if brightness > 0 else 20
        _contrast: float = contrast if contrast > 0 else 20
        for image in self.image_file_names:
            _file_name: str = image.split('/')[-1]
            _file_type: str = _file_name.split('.')[-1]
            _image = cv2.imread(filename=image)
            _contrast_diff: float = (100 - _contrast)
            if _contrast_diff <= 0.1:
                _contrast = 99.9
            _arg: float = math.pi * (((_contrast * _contrast) / 20000) + (3 * _contrast / 200)) / 4
            _slope: float = 1 + (math.sin(_arg) / math.cos(_arg))
            if _slope < 0:
                _slope = 0
            _pivot: float = (100 - _brithness) / 200
            _intercept_brigthness: float = _brithness / 100
            _interecept_contrast: float = _pivot * (1 - _slope)
            _intercept: float = _intercept_brigthness + _interecept_contrast
            _image = _image / 255.0
            _out: np.array = _slope * _image + _intercept
            _out[_out > 1] = 1
            _out[_out < 0] = 0
            self.labeling['file_name'].append(os.path.join(self.file_path_output_noisy_images,
                                                           _file_name.split('.')[0],
                                                           '_fade',
                                                           _file_type
                                                           )
                                              )
            self.labeling['label'].append('fade')
            self.labeling['label_encoding'].append('2')
            cv2.imwrite(filename=self.labeling['file_name'][-1], img=_image)

    def salt_pepper(self, number_of_pixel_edges: tuple = (300, 1000)):
        """
        Generate salt & pepper noise images

        :param number_of_pixel_edges: tuple
            Edges to draw number of pixels to coloring into white and black
        """
        for image in self.image_file_names:
            _file_name: str = image.split('/')[-1]
            _file_type: str = _file_name.split('.')[-1]
            _image = cv2.imread(filename=image)
            _height, _width = _image.shape
            _number_of_pixels: int = random.randint(number_of_pixel_edges[0], number_of_pixel_edges[1])
            # White pixels:
            for i in range(0, _number_of_pixels, 1):
                _y_coord = random.randint(0, _height - 1)
                _x_coord = random.randint(0, _width - 1)
                _image[_y_coord][_x_coord] = 255
            # Black pixels:
            for i in range(0, _number_of_pixels, 1):
                _y_coord = random.randint(0, _height - 1)
                _x_coord = random.randint(0, _width - 1)
                _image[_y_coord][_x_coord] = 0
            self.labeling['file_name'].append(os.path.join(self.file_path_output_noisy_images,
                                                           _file_name.split('.')[0],
                                                           '_salt_pepper',
                                                           _file_type
                                                           )
                                              )
            self.labeling['label'].append('salt_pepper')
            self.labeling['label_encoding'].append('3')
            cv2.imwrite(filename=self.labeling['file_name'][-1], img=_image)

    def save_labels(self, include_image_data):
        """
        Save noise type labeling

        :param include_image_data: bool
            Whether to include image data or not
        """
        if include_image_data:
            self.labeling['image'] = []
            for image in self.labeling.get('file_name'):
                _image = cv2.imread(filename=image)
                self.labeling['image'].append(_image)
        with open(self.file_path_output_noisy_images, 'w') as _file:
            json.dump(obj=self.labeling, fp=_file, ensure_ascii=False)
        if include_image_data:
            del self.labeling['image']

    def watermark(self, watermark_files: List[str]):
        """
        Generate watermarked noise images

        :param watermark_files: List[str]
            Complete file path of the files containing watermarks
        """
        for image in self.image_file_names:
            _file_name: str = image.split('/')[-1]
            _file_type: str = _file_name.split('.')[-1]
            _image = cv2.imread(filename=image)
            _height_image, _width_image, _ = _image.shape
            _center_y: int = int(_height_image / 2)
            _center_x: int = int(_width_image / 2)
            for watermark in watermark_files:
                _file_name_watermark: str = watermark.split('/')[-1]
                _file_type_watermark: str = _file_name_watermark.split('.')[-1]
                _watermark = cv2.imread(filename=watermark)
                _height_watermark, _width_watermark, _ = _watermark.shape
                _top_y: int = _center_y - int(_height_watermark / 2)
                _left_x: int = _center_x - int(_width_watermark / 2)
                _bottom_y: int = _top_y + _height_watermark
                _right_x: int = _left_x + _width_watermark
                _destination = _image[_top_y:_bottom_y, _left_x:_right_x]
                _result = cv2.addWeighted(_destination, 1, _watermark, 0.5, 0)
                _image[_top_y:_bottom_y, _left_x:_right_x] = _result
                self.labeling['file_name'].append(os.path.join(self.file_path_output_noisy_images,
                                                               _file_name.split('.')[0],
                                                               '_watermark_',
                                                               _file_name_watermark.split('.')[0],
                                                               _file_type
                                                               )
                                                  )
                self.labeling['label'].append('watermark')
                self.labeling['label_encoding'].append('4')
                cv2.imwrite(filename=self.labeling['file_name'][-1], img=_image)


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
