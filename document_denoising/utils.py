"""
Utility functions
"""

from glob import glob
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.engine import keras_tensor
from keras.initializers.initializers_v2 import (
    Constant, HeNormal, HeUniform, GlorotNormal, GlorotUniform, LecunNormal, LecunUniform, Ones, Orthogonal, RandomNormal, RandomUniform, TruncatedNormal, Zeros
)
from keras.layers import BatchNormalization, Dropout, Input, ReLU
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Activation, Dense, Dropout, Flatten, Input
from keras.models import Model, Sequential
from keras.preprocessing.image import DirectoryIterator, ImageDataGenerator
from skimage.transform import resize
from tensorflow import pad
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Layer
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from typing import List, Tuple

import cv2
import json
import keras
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import tensorflow as tf


def peek_signal_to_noise_ratio(first_image: np.array, second_image: np.array) -> float:
    """
    Calculate peek signal-to-noise ratio (psnr)

    :param first_image: np.array
        First image

    :param second_image: np.array
        Second image

    :return float
        Peek signal-to-noise ratio
    """
    return cv2.PSNR(src1=first_image, src2=second_image, R=None)


def train_noise_type_classifier(file_path_train_images: str,
                                file_path_validation_images: str,
                                file_path_model_output: str,
                                learning_rate: float = 0.001,
                                n_epoch: int = 10,
                                batch_size: int = 32,
                                image_width: int = 256,
                                image_height: int = 256,
                                n_channels: int = 3,
                                n_noise_types: int = 4
                                ):
    """
    Train model for classifying noise types (required in mixture of experts generator)

    :param file_path_train_images: str
        Complete file path for the training noise type images

    :param file_path_validation_images: str
        Complete file path for the validation noise type images

    :param file_path_model_output: str
        Complete file path for the trained model to save

    :param learning_rate: float
        Learning rate

    :param n_epoch: int
            Number of epochs to train

    :param batch_size: int
            Batch size

    :param image_height: int
        Height of the image

    :param image_width: int
        Width of the image

    :param n_channels: int
            Number of image channels
                -> 1: gray
                -> 3: color (rbg)

    :param n_noise_types: int
        Number of noise type classes
    """
    if len(file_path_train_images) == 0:
        raise FileNotFoundError('No training images found')
    if len(file_path_validation_images) == 0:
        raise FileNotFoundError('No validation images found')
    if len(file_path_model_output) == 0:
        raise FileNotFoundError('No path for the trained model output found')
    _learning_rate: float = learning_rate if learning_rate > 0 else 0.001
    _n_epoch: int = n_epoch if n_epoch > 0 else 10
    _batch_size: int = batch_size if batch_size > 0 else 32
    _image_width: int = image_width if image_width > 10 else 256
    _image_height: int = image_height if image_height > 10 else 256
    if 1 < n_channels < 4:
        _n_channels: int = 3
    else:
        _n_channels: int = 1
    _image_shape: tuple = (_image_width, _image_height, _n_channels)
    if n_noise_types < 2:
        raise ValueError(f'Not enough noise type classes ({n_noise_types})')
    # Build neural network model:
    _model: Sequential = Sequential()
    # 1. Convolutional Layer:
    _model.add(Conv2D(32, (2, 2), input_shape=_image_shape))
    _model.add(Activation('relu'))
    _model.add(MaxPooling2D(pool_size=(2, 2)))
    # 2. Convolutional Layer:
    _model.add(Conv2D(32, (2, 2)))
    _model.add(Activation('relu'))
    _model.add(MaxPooling2D(pool_size=(2, 2)))
    # 3. Convolutional Layer:
    _model.add(Conv2D(32, (2, 2)))
    _model.add(Activation('relu'))
    _model.add(MaxPooling2D(pool_size=(2, 2)))
    # 4. Convolutional Layer:
    _model.add(Conv2D(64, (2, 2)))
    _model.add(Activation('relu'))
    _model.add(MaxPooling2D(pool_size=(2, 2)))
    # 5. Convolutional Layer:
    _model.add(Conv2D(64, (2, 2)))
    _model.add(Activation('relu'))
    _model.add(MaxPooling2D(pool_size=(2, 2)))
    # 6. Convolutional Layer:
    _model.add(Conv2D(64, (2, 2)))
    _model.add(Activation('relu'))
    _model.add(MaxPooling2D(pool_size=(2, 2)))
    # 7. Convolutional Layer:
    _model.add(Conv2D(64, (2, 2)))
    _model.add(Activation('relu'))
    _model.add(MaxPooling2D(pool_size=(2, 2)))
    # MLP Layer:
    _model.add(Flatten())
    _model.add(Dense(64))
    _model.add(Activation('relu'))
    _model.add(Dropout(0.5))
    _model.add(Dense(n_noise_types))
    _model.add(Activation('softmax'))
    # Compile model:
    _model.compile(loss='categorical_crossentropy',
                   optimizer=Adam(learning_rate=_learning_rate,
                                  beta_1=0.9,
                                  beta_2=0.999,
                                  epsilon=1e-7,
                                  amsgrad=False
                                  ),
                   metrics=[tf.keras.metrics.Accuracy(),
                            tf.keras.metrics.Recall(),
                            tf.keras.metrics.TruePositives(),
                            tf.keras.metrics.FalsePositives(),
                            tf.keras.metrics.TrueNegatives(),
                            tf.keras.metrics.FalseNegatives()
                            ]
                   )
    # Load and preprocess image data (train & validation):
    _train_data_generator: ImageDataGenerator = ImageDataGenerator(rescale=1. / 255,
                                                                   shear_range=0.0,
                                                                   zoom_range=0.0,
                                                                   horizontal_flip=True
                                                                   )
    _validation_data_generator: ImageDataGenerator = ImageDataGenerator(rescale=1. / 255)
    _train_generator = _train_data_generator.flow_from_directory(directory=file_path_train_images,
                                                                 target_size=(_image_width, _image_height),
                                                                 batch_size=_batch_size,
                                                                 class_mode='categorical'
                                                                 )
    _validation_generator = _validation_data_generator.flow_from_directory(directory=file_path_validation_images,
                                                                           target_size=(_image_width, _image_height),
                                                                           batch_size=_batch_size,
                                                                           class_mode='categorical'
                                                                           )
    # Train neural network model:
    _model.fit(x=_train_generator,
               #steps_per_epoch=300 // batch_size,
               epochs=_n_epoch,
               validation_data=_validation_generator,
               #validation_steps=75 // _batch_size
               )
    # Save trained neural network model:
    _model.save_weights(filepath=os.path.join(file_path_model_output, 'noise_type_clf.h5'))


class ImageProcessor:
    """
    Class for processing, loading and saving document images
    """
    def __init__(self,
                 file_path_clean_images: str,
                 file_path_noisy_images: str,
                 file_path_multi_noisy_images: List[str] = None,
                 n_channels: int = 1,
                 batch_size: int = 1,
                 image_resolution: tuple = None,
                 normalize: bool = False,
                 flip: bool = True,
                 crop: Tuple[Tuple[int, int], Tuple[int, int]] = None,
                 file_type: str = None
                 ):
        """
        :param file_path_clean_images: str
            Complete file path of the clean images

        :param file_path_noisy_images: str
            Complete file path of the noisy images

        :param file_path_multi_noisy_images: List[str]
            Complete file paths of several noisy images

        :param n_channels: int
            Number of channels of the image
                -> 0: gray
                -> 3: color (rgb)

        :param batch_size: int
            Batch size

        :param image_resolution: tuple
            Force resizing image into given resolution

        :param normalize: bool
            Whether to normalize image (rescale to 0 - 1) or not

        :param flip: bool
            Whether to flip image based on probability distribution or not

        :parm crop: Tuple[Tuple[int, int], Tuple[int, int]]
            Define image cropping

        :param file_type: str
            Specific image file type
        """
        self.file_path_clean_images: str = file_path_clean_images
        self.file_path_noisy_images: str = file_path_noisy_images
        self.file_path_multi_noisy_images: List[str] = file_path_multi_noisy_images
        self.file_type: str = '' if file_type is None else file_type
        self.n_channels: int = cv2.IMREAD_COLOR if n_channels > 1 else cv2.IMREAD_GRAYSCALE
        self.batch_size: int = batch_size
        self.n_batches: int = 0
        self.image_resolution: tuple = image_resolution
        self.normalize: bool = normalize
        self.flip: bool = flip
        self.crop: Tuple[Tuple[int, int], Tuple[int, int]] = crop

    @staticmethod
    def _de_normalize(images: np.array) -> np.array:
        """
        De-normalize images

        :param images: np.array
            Images
        """
        return images / (1 / 255)

    @staticmethod
    def _normalize(images: np.array) -> np.array:
        """
        Normalize images

        :param images: np.array
            Images
        """
        return images * (1 / 255)

    def _read_image(self, file_path: str) -> np.array:
        """
        Read image

        :param file_path: str
            Complete file path of the image to read

        :return np.array
            Image
        """
        _image: np.array = cv2.imread(filename=file_path, flags=self.n_channels)
        if self.crop is not None:
            _image = _image[self.crop[0][0]:self.crop[0][1], self.crop[1][0]:self.crop[1][1]]
        if self.image_resolution is not None:
            _image = resize(image=_image, output_shape=self.image_resolution)
        if self.flip:
            if np.random.random() > 0.5:
                if np.random.random() > 0.5:
                    _direction: int = 0
                else:
                    _direction: int = 1
                _image = cv2.flip(src=_image, flipCode=_direction, dst=None)
        return _image

    def load_batch(self) -> Tuple[np.array, np.array, List[int]]:
        """
        Load batch images for each group (clean & noisy) separately

        :return Tuple[np.array, np.array, List[str]]
            Arrays of clean & noisy images as well as noise labels
        """
        if self.file_path_multi_noisy_images is None:
            _label: int = 0
            _file_paths_noisy: List[str] = glob(os.path.join('.', self.file_path_noisy_images, f'*{self.file_type}'))
        else:
            _label: int = random.randint(a=0, b=len(self.file_path_multi_noisy_images) - 1)
            _file_paths_noisy: List[str] = glob(os.path.join('.', self.file_path_multi_noisy_images[_label], f'*{self.file_type}'))
        _file_paths_clean: List[str] = glob(os.path.join('.', self.file_path_clean_images, f'*{self.file_type}'))
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
            if self.normalize:
                yield self._normalize(np.array(_images_clean)), self._normalize(np.array(_images_noisy)), _label
            else:
                yield np.array(_images_clean), np.array(_images_noisy), _label

    def load_all_images(self) -> Tuple[np.array, np.array]:
        """
        Load all images without batching
        """
        if self.file_path_multi_noisy_images is None:
            _label: int = 0
            _file_paths_noisy: List[str] = glob(os.path.join('.', self.file_path_noisy_images, f'*{self.file_type}'))
        else:
            _label: int = random.randint(a=0, b=len(self.file_path_multi_noisy_images) - 1)
            _file_paths_noisy: List[str] = glob(os.path.join('.', self.file_path_multi_noisy_images[_label], f'*{self.file_type}'))
        _file_paths_clean: List[str] = glob(os.path.join('.', self.file_path_clean_images, f'*{self.file_type}'))
        sorted(_file_paths_clean, reverse=False)
        sorted(_file_paths_noisy, reverse=False)
        _images_clean, _images_noisy = [], []
        for path_image_clean, path_image_noisy in zip(_file_paths_clean, _file_paths_noisy):
            _images_clean.append(self._read_image(file_path=path_image_clean))
            _images_noisy.append(self._read_image(file_path=path_image_noisy))
        if self.normalize:
            return self._normalize(np.array(_images_clean)), self._normalize(np.array(_images_noisy))
        else:
            return np.array(_images_clean), np.array(_images_noisy)

    def load_images(self, n_images: int = None) -> Tuple[str, np.array]:
        """
        Load images without batching

        :return Tuple[List[str], np.array]
            List of image file names & array of loaded images
        """
        _images_path, _images_noisy = [], []
        _file_paths_noisy: List[str] = glob(f'{self.file_path_noisy_images}/*')
        if n_images is not None and n_images > 0:
            for i in range(0, n_images, 1):
                _file_paths_noisy_sample: List[str] = np.random.choice(_file_paths_noisy, 1, replace=False)
                for path_image_noisy in _file_paths_noisy_sample:
                    _images_path.append(path_image_noisy)
                    _images_noisy.append(self._read_image(file_path=path_image_noisy))
                if self.normalize:
                    yield _images_path, self._normalize(np.array(_images_noisy))
                else:
                    yield _images_path, np.array(_images_noisy)
        else:
            self.n_batches = int(len(_file_paths_noisy))
            _total_samples: int = self.n_batches * self.batch_size
            _file_paths_noisy_sample: List[str] = np.random.choice(_file_paths_noisy, _total_samples, replace=False)
            for i in range(0, self.n_batches - 1, 1):
                _batch_noisy: List[str] = _file_paths_noisy_sample[i * self.batch_size:(i + 1) * self.batch_size]
                for path_image_noisy in _batch_noisy:
                    _images_path.append(path_image_noisy)
                    _images_noisy.append(self._read_image(file_path=path_image_noisy))
                if self.normalize:
                    yield _images_path, self._normalize(np.array(_images_noisy))
                else:
                    yield _images_path, np.array(_images_noisy)

    def save_image(self, image: np.array, output_file_path: str):
        """
        Save image

        :param image: np.array
            Image to save

        :param output_file_path: str
            Complete file path of the output image
        """
        if self.normalize:
            _image: np.array = self._de_normalize(images=image)
        else:
            _image: np.array = image
        fig = plt.figure()
        plt.imshow(X=_image, cmap='Greys_r')
        fig.savefig(output_file_path)
        plt.close()
        #cv2.imwrite(filename=output_file_path, img=_image, params=None)


class ImageSkew:
    """
    Class for handling skew in document images
    """
    def __init__(self,
                 file_path_input_images: str,
                 file_path_output_images: str = None,
                 denoise_by_blurring: bool = False
                 ):
        """
        :param file_path_input_images: str
            Complete file path of the input images

        :param file_path_output_images: str
            Complete file path of the output images

        :param denoise_by_blurring: bool
            Whether to denoise image by burring or not
        """
        self.denoise_by_blurring: bool = denoise_by_blurring
        if len(file_path_input_images) == 0:
            raise ValueError('Input file path is empty')
        self.file_path_input_images: str = file_path_input_images
        if self.file_path_input_images is None or len(file_path_output_images) == 0:
            self.file_path_output_images: str = file_path_input_images
        else:
            self.file_path_output_images: str = file_path_output_images
        self.image_processor: ImageProcessor = ImageProcessor(file_path_clean_images='',
                                                              file_path_noisy_images=self.file_path_input_images,
                                                              n_channels=1,
                                                              batch_size=1,
                                                              image_resolution=None,
                                                              normalize=False,
                                                              flip=False,
                                                              crop=None,
                                                              file_type=None
                                                              )

    @staticmethod
    def _rotate_image(image: np.array, angle: float) -> np.array:
        """
        Rotate image

        :param image: np.array
            Image

        :param angle: float
            Angle for rotation

        :return np.array
            Rotated images
        """
        _rotated_image: np.array = image
        (_height, _width) = _rotated_image.shape[:2]
        _center: tuple = (_width // 2, _height // 2)
        _rotation_matrix = cv2.getRotationMatrix2D(center=_center, angle=angle, scale=1.0)
        return cv2.warpAffine(src=_rotated_image,
                              M=_rotation_matrix,
                              dsize=(_width, _height),
                              flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE
                              )

    def deskew(self):
        """
        Deskew text in document image
        """
        for image_file_path, image in self.image_processor.load_images(n_images=1):
            _angle: float = self.get_angle(image=image)
            _image: np.array = self._rotate_image(image=image, angle=-1.0 * _angle)
            _file_name: str = image_file_path.split('/')[-1]
            _file_type: str = _file_name.split('.')[-1]
            _file_name = f'{_file_name.split(".")[0]}_deskewed.{_file_type}'
            _file_path: str = os.path.join(self.file_path_output_images, _file_name)
            self.image_processor.save_image(image=_image, output_file_path=_file_path)

    def get_angle(self, image: np.array) -> float:
        """
        Get angle of text in document image

        :param image: np.array
            Image

        :return float
            Angle of the text in document image
        """
        if self.denoise_by_blurring:
            _image: np.array = cv2.GaussianBlur(src=image, ksize=(9, 9), sigmaX=0, dst=None, sigmaY=None, borderType=None)
        else:
            _image: np.array = image
        _image = cv2.threshold(src=_image,
                               thresh=0,
                               maxval=255,
                               type=cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
                               dst=None
                               )[1]

        # Apply dilate to merge text into meaningful lines/paragraphs.
        # Use larger kernel on X axis to merge characters into single line, cancelling out any spaces.
        # But use smaller kernel on Y axis to separate between different blocks of text
        _kernel = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(30, 5), anchor=None)
        _image = cv2.dilate(src=_image, kernel=_kernel, iterations=5, borderType=None, borderValue=None)

        # Find all contours
        _contours, _hierarchy = cv2.findContours(image=_image,
                                                 mode=cv2.RETR_LIST,
                                                 method=cv2.CHAIN_APPROX_SIMPLE,
                                                 contours=None,
                                                 hierarchy=None,
                                                 offset=None
                                                 )
        _contours = sorted(_contours, key=cv2.contourArea, reverse=True)

        # Find largest contour and surround in min area box
        _largest_contour: float = _contours[0]
        _min_area_rectangle: tuple = cv2.minAreaRect(points=_largest_contour)

        # Determine the angle. Convert it to the value that was originally used to obtain skewed image
        _angle: float = _min_area_rectangle[-1]
        if _angle < -45:
            _angle = 90 + _angle
        return -1.0 * _angle

    def skew(self, angle: float):
        """
        Skew image

        :param angle: float
            Angle of the text in document image
        """
        for image_file_path, image in self.image_processor.load_images(n_images=1):
            _image: np.array = self._rotate_image(image=image, angle=angle)
            _file_name: str = image_file_path.split('/')[-1]
            _file_type: str = _file_name.split('.')[-1]
            _file_name = f'{_file_name.split(".")[0]}_skewed.{_file_type}'
            _file_path: str = os.path.join(self.file_path_output_images, _file_name)
            self.image_processor.save_image(image=_image, output_file_path=_file_path)


class NoiseGenerator:
    """
    Class for generating noisy document images used in model training
    """
    def __init__(self,
                 file_path_input_clean_images: str,
                 file_path_output_noisy_images: str,
                 file_type: str = None,
                 image_resolution: tuple = None
                 ):
        """
        :param file_path_input_clean_images: str
            Complete file path of clean input images

        :param file_path_output_noisy_images: str
            Complete file path of noisy output images

        :param file_type: str
            Specific image file type

        :param image_resolution: tuple
            Resizing image into given resolution
        """
        self.file_path_input_clean_images: str = file_path_input_clean_images
        self.file_path_output_noisy_images: str = file_path_output_noisy_images
        self.file_type: str = '' if file_type is None else file_type
        self.image_resolution: tuple = image_resolution
        self.labeling: dict = dict(file_name=[], label=[], label_encoding=[])
        self.image_file_names: List[str] = glob(os.path.join(self.file_path_input_clean_images, f'*{self.file_type}'))

    def blur(self, blur_type: str = 'average', kernel_size: tuple = (9, 9)):
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
            if _blur_type == 'average':
                _image = cv2.blur(_image, kernel_size)
            elif _blur_type == 'median':
                _image = cv2.medianBlur(_image, kernel_size[0])
            _output_file_path: str = os.path.join(self.file_path_output_noisy_images,
                                                  f"{_file_name.split('.')[0]}_blur.{_file_type}"
                                                  )
            if _output_file_path not in self.labeling['file_name']:
                self.labeling['file_name'].append(_output_file_path)
                self.labeling['label'].append('blur')
                self.labeling['label_encoding'].append('1')
            if self.image_resolution is not None:
                _image = cv2.resize(src=_image, dsize=self.image_resolution, dst=None, fx=None, fy=None, interpolation=None)
            cv2.imwrite(filename=_output_file_path, img=_image)

    def fade(self, brightness: int = 20):
        """
        Generate faded noise images

        :param brightness: int
            Desired brightness change
        """
        _brightness: float = brightness if brightness > 0 else 150
        for image in self.image_file_names:
            _file_name: str = image.split('/')[-1]
            _file_type: str = _file_name.split('.')[-1]
            _image = cv2.imread(filename=image)
            _hsv = cv2.cvtColor(_image, cv2.COLOR_BGR2HSV)
            _h, _s, _v = cv2.split(_hsv)
            _lim: float = 255 - _brightness
            _v[_v > _lim] = 255
            _v[_v <= _lim] += _brightness
            _final_hsv = cv2.merge((_h, _s, _v))
            _image = cv2.cvtColor(_final_hsv, cv2.COLOR_HSV2BGR)
            _output_file_path: str = os.path.join(self.file_path_output_noisy_images,
                                                  f"{_file_name.split('.')[0]}_fade.{_file_type}"
                                                  )
            if _output_file_path not in self.labeling['file_name']:
                self.labeling['file_name'].append(_output_file_path)
                self.labeling['label'].append('fade')
                self.labeling['label_encoding'].append('2')
            if self.image_resolution is not None:
                _image = cv2.resize(src=_image, dsize=self.image_resolution, dst=None, fx=None, fy=None, interpolation=None)
            cv2.imwrite(filename=_output_file_path, img=_image)

    def salt_pepper(self, number_of_pixel_edges: tuple = (100000, 500000)):
        """
        Generate salt & pepper noise images

        :param number_of_pixel_edges: tuple
            Edges to draw number of pixels to coloring into white and black
        """
        for image in self.image_file_names:
            _file_name: str = image.split('/')[-1]
            _file_type: str = _file_name.split('.')[-1]
            _image = cv2.imread(filename=image)
            _height, _width, _ = _image.shape
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
            _output_file_path: str = os.path.join(self.file_path_output_noisy_images,
                                                  f"{_file_name.split('.')[0]}_salt_pepper.{_file_type}"
                                                  )
            if _output_file_path not in self.labeling['file_name']:
                self.labeling['file_name'].append(_output_file_path)
                self.labeling['label'].append('salt_pepper')
                self.labeling['label_encoding'].append('3')
            if self.image_resolution is not None:
                _image = cv2.resize(src=_image, dsize=self.image_resolution, dst=None, fx=None, fy=None, interpolation=None)
            cv2.imwrite(filename=_output_file_path, img=_image)

    def save_labels(self):
        """
        Save noise type labeling
        """
        with open(os.path.join(self.file_path_output_noisy_images, 'noisy_document_images.json'), 'w') as _file:
            json.dump(obj=self.labeling, fp=_file, ensure_ascii=False)

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
                _output_file_path: str = os.path.join(self.file_path_output_noisy_images,
                                                      f"{_file_name.split('.')[0]}_watermark_{_file_name_watermark.split('.')[0]}.{_file_type}"
                                                      )
                if _output_file_path not in self.labeling['file_name']:
                    self.labeling['file_name'].append(_output_file_path)
                    self.labeling['label'].append('watermark')
                    self.labeling['label_encoding'].append('4')
                if self.image_resolution is not None:
                    _image = cv2.resize(src=_image, dsize=self.image_resolution, dst=None, fx=None, fy=None, interpolation=None)
                cv2.imwrite(filename=_output_file_path, img=_image)


class NoiseTypeClassifierException(Exception):
    """
    Class for handling exceptions for class NoiseTypeClassifier
    """
    pass


class NoiseTypeClassifier:
    """
    Class for training noise type classifier network
    """
    def __init__(self,
                 file_path_train_images: str,
                 file_path_validation_images: str,
                 n_channels: int = 1,
                 image_height: int = 256,
                 image_width: int = 256,
                 learning_rate: float = 0.001,
                 optimizer: str = 'adam',
                 initializer: str = 'he_normal',
                 batch_size: int = 32,
                 start_n_filters: int = 64,
                 up_sample_n_filters_period: int = 3,
                 n_conv_layers: int = 3,
                 dropout_rate_conv: float = 0.0,
                 fc_input_units: int = 64,
                 dropout_rate_fc: float = 0.5,
                 n_noise_types: int = 4,
                 print_model_architecture: bool = True
                 ):
        """
        :param file_path_train_images: str
            Complete file path of images used for training

        :param file_path_validation_images: str
            Complete file path of images used for evaluation

        :param n_channels: int
            Number of image channels
                -> 1: gray
                -> 3: color (rbg)

        :param image_height: int
            Height of the image

        :param image_width: int
            Width of the image

        :param learning_rate: float
            Learning rate

        :param initializer: str
            Name of the initializer used in convolutional layers
                -> constant: Constant value 2
                -> he_normal:
                -> he_uniform:
                -> glorot_normal: Xavier normal
                -> glorot_uniform: Xavier uniform
                -> lecun_normal: Lecun normal
                -> lecun_uniform:
                -> ones: Constant value 1
                -> orthogonal:
                -> random_normal:
                -> random_uniform:
                -> truncated_normal:
                -> zeros: Constant value 0

        :param batch_size: int
            Batch size

        :param start_n_filters: int
            Number of filters used in first convolutional layer in auto-encoder network

        :param up_sample_n_filters_period: int
            Number of layers until up-sampling number of filters

        :param n_conv_layers: int
            Number of convolutional layer in auto-encoder network

        :param dropout_rate_conv: float
            Dropout rate used after each convolutional layer

        :param fc_input_units: int
            Number of units (neurons) of the fully connected input layer

        :param dropout_rate_fc: float
            Dropout rate used after the fully connected input layer

        :param n_noise_types: int
            Number of different noise types to classify

        :param print_model_architecture: bool
            Whether to print architecture of cycle-gan model components (discriminators & generators) or not
        """
        if len(file_path_train_images) == 0:
            raise NoiseTypeClassifierException('No training images found')
        if len(file_path_validation_images) == 0:
            raise NoiseTypeClassifierException('No validation images found')
        self.file_path_train_images: str = file_path_train_images
        self.file_path_validation_images: str = file_path_validation_images
        self.learning_rate: float = learning_rate if learning_rate > 0 else 0.001
        self.batch_size: int = batch_size if batch_size > 0 else 32
        self.image_height: int = image_height if image_height > 10 else 256
        self.image_width: int = image_width if image_width > 10 else 256
        if 1 < n_channels < 4:
            self.n_channels: int = 3
        else:
            self.n_channels: int = 1
        self.image_shape: tuple = (self.image_width, self.image_height, self.n_channels)
        if n_noise_types < 2:
            raise NoiseTypeClassifierException(f'Not enough noise type classes ({n_noise_types})')
        self.n_noise_types: int = n_noise_types
        self.learning_rate: float = learning_rate if learning_rate > 0 else 0.001
        if optimizer == 'rmsprop':
            self.optimizer: RMSprop = RMSprop(learning_rate=self.learning_rate,
                                              rho=0.9,
                                              momentum=0.0,
                                              epsilon=1e-7,
                                              centered=False
                                              )
        elif optimizer == 'sgd':
            self.optimizer: SGD = SGD(learning_rate=self.learning_rate,
                                      momentum=0.0,
                                      nesterov=False
                                      )
        else:
            self.optimizer: Adam = Adam(learning_rate=self.learning_rate,
                                        beta_1=0.5,
                                        beta_2=0.999,
                                        epsilon=1e-7,
                                        amsgrad=False
                                        )
        self.batch_size: int = batch_size if batch_size > 0 else 1
        if self.batch_size == 1:
            self.normalizer = InstanceNormalization
        else:
            self.normalizer = BatchNormalization
        if initializer == 'constant':
            self.initializer: keras.initializers.initializers_v2 = Constant(value=2)
        elif initializer == 'he_normal':
            self.initializer: keras.initializers.initializers_v2 = HeNormal(seed=1234)
        elif initializer == 'he_uniform':
            self.initializer: keras.initializers.initializers_v2 = HeUniform(seed=1234)
        elif initializer == 'glorot_normal':
            self.initializer: keras.initializers.initializers_v2 = GlorotNormal(seed=1234)
        elif initializer == 'glorot_uniform':
            self.initializer: keras.initializers.initializers_v2 = GlorotUniform(seed=1234)
        elif initializer == 'lecun_normal':
            self.initializer: keras.initializers.initializers_v2 = LecunNormal(seed=1234)
        elif initializer == 'lecun_uniform':
            self.initializer: keras.initializers.initializers_v2 = LecunUniform(seed=1234)
        elif initializer == 'ones':
            self.initializer: keras.initializers.initializers_v2 = Ones()
        elif initializer == 'orthogonal':
            self.initializer: keras.initializers.initializers_v2 = Orthogonal(gain=1.0, seed=1234)
        elif initializer == 'random_normal':
            self.initializer: keras.initializers.initializers_v2 = RandomNormal(mean=0.0, stddev=0.2, seed=1234)
        elif initializer == 'random_uniform':
            self.initializer: keras.initializers.initializers_v2 = RandomUniform(minval=-0.05, maxval=0.05, seed=1234)
        elif initializer == 'truncated_normal':
            self.initializer: keras.initializers.initializers_v2 = TruncatedNormal(mean=0.0, stddev=0.05, seed=1234)
        elif initializer == 'zeros':
            self.initializer: keras.initializers.initializers_v2 = Zeros()
        self.start_n_filters: int = start_n_filters if start_n_filters > 0 else 32
        self.up_sample_n_filters_period: int = up_sample_n_filters_period if up_sample_n_filters_period > 0 else 3
        self.n_conv_layers: int = n_conv_layers if n_conv_layers > 0 else 3
        self.dropout_rate_conv: float = dropout_rate_conv if dropout_rate_conv >= 0 else 0.0
        self.fc_input_units: int = fc_input_units if fc_input_units > 0 else 64
        self.dropout_rate_fc: float = dropout_rate_fc if dropout_rate_fc >= 0 else 0.0
        self.print_model_architecture: bool = print_model_architecture
        self.model: Model = None
        # Load and preprocess image data (train & validation):
        self.train_data_generator: ImageDataGenerator = ImageDataGenerator(rescale=1. / 255,
                                                                           shear_range=0.0,
                                                                           zoom_range=0.0,
                                                                           horizontal_flip=True
                                                                           )
        self.validation_data_generator: ImageDataGenerator = ImageDataGenerator(rescale=1. / 255)
        self.train_generator: DirectoryIterator = self.train_data_generator.flow_from_directory(directory=self.file_path_train_images,
                                                                                                target_size=(self.image_width, self.image_height),
                                                                                                batch_size=self.batch_size,
                                                                                                class_mode='categorical'
                                                                                                )
        self.validation_generator: DirectoryIterator = self.validation_data_generator.flow_from_directory(directory=self.file_path_validation_images,
                                                                                                          target_size=(self.image_width, self.image_height),
                                                                                                          batch_size=self.batch_size,
                                                                                                          class_mode='categorical'
                                                                                                          )
        self.model: Model = None
        # Build neural network model:
        self._build_clf_network()

    def _build_clf_network(self):
        """
        Build classification model network
        """
        _input: Input = Input(shape=self.image_shape)
        _n_filters: int = self.start_n_filters
        _clf: keras_tensor.KerasTensor = Conv2D(filters=_n_filters,
                                                kernel_size=(2, 2),
                                                strides=(1, 1),
                                                padding='valid',
                                                kernel_initializer=self.initializer
                                                )(_input)
        _clf = ReLU(max_value=None, negative_slope=0, threshold=0)(_clf)
        _clf = MaxPooling2D(pool_size=(2, 2),
                            strides=None,
                            padding='valid',
                            data_format=None
                            )(_clf)
        # Convolutional Layers:
        for i in range(0, self.n_conv_layers - 1, 1):
            if (i + 1) % self.up_sample_n_filters_period == 0:
                _n_filters *= 2
            _clf = Conv2D(filters=_n_filters,
                          kernel_size=(2, 2),
                          strides=(1, 1),
                          padding='valid',
                          kernel_initializer=self.initializer
                          )(_clf)
            _clf = ReLU(max_value=None, negative_slope=0, threshold=0)(_clf)
            if self.dropout_rate_conv > 0:
                _clf = Dropout(rate=self.dropout_rate_conv, seed=1234)(_clf)
            _clf = MaxPooling2D(pool_size=(2, 2),
                                strides=None,
                                padding='valid',
                                data_format=None
                                )(_clf)
        # Fully Connected Layer:
        _clf = Flatten()(_clf)
        _clf = Dense(units=self.fc_input_units,
                     activation=None,
                     use_bias=True,
                     kernel_initializer=self.initializer,
                     bias_initializer='zeros',
                     kernel_regularizer=None,
                     bias_regularizer=None,
                     activity_regularizer=None,
                     kernel_constraint=None,
                     bias_constraint=None
                     )
        _clf = ReLU(max_value=None, negative_slope=0, threshold=0)(_clf)
        if self.dropout_rate_fc > 0:
            _clf = Dropout(rate=self.dropout_rate_fc, seed=1234)(_clf)
        _clf = Dense(units=self.n_noise_types,
                     activation=None,
                     use_bias=True,
                     kernel_initializer=self.initializer,
                     bias_initializer='zeros',
                     kernel_regularizer=None,
                     bias_regularizer=None,
                     activity_regularizer=None,
                     kernel_constraint=None,
                     bias_constraint=None
                     )(_clf)
        if self.n_noise_types == 2:
            # Binary Classification:
            _clf = Activation('sigmoid')(_clf)
        else:
            # Multi Classification:
            _clf = Activation('softmax')(_clf)
        self.model = Model(inputs=_input, outputs=_clf, name='classifier')
        if self.print_model_architecture:
            self.model.summary()
        self.model.compile(optimizer=self.optimizer,
                           loss='categorical_crossentropy',
                           metrics=[tf.keras.metrics.Accuracy(),
                                    tf.keras.metrics.Recall(),
                                    tf.keras.metrics.TruePositives(),
                                    tf.keras.metrics.FalsePositives(),
                                    tf.keras.metrics.TrueNegatives(),
                                    tf.keras.metrics.FalseNegatives()
                                    ],
                           loss_weights=None,
                           weighted_metrics=None,
                           run_eagerly=None,
                           steps_per_execution=None
                           )

    def _eval_training(self, file_path: str):
        """
        Evaluate current training by generating predictions based on test images

        :param file_path: str
            Complete file path to save test predictions
        """
        pass

    def _save_models(self, model_output_path: str):
        """
        Save cycle-gan models

        :param model_output_path: str
            Complete file path of the model output
        """
        self.model.save(filepath=os.path.join(model_output_path, 'noise_type_clf.h5'))

    def train(self,
              model_output_path: str,
              n_epoch: int = 100,
              early_stopping_patience: int = 0
              ):
        """
        Train classifier network model

        :param model_output_path: str
            Complete file path of the model output

        :param n_epoch: int
            Number of epochs to train

        :param early_stopping_patience: int
            Number of unchanged gradient epoch intervals for early stopping
        """
        if early_stopping_patience > 0:
            _callbacks: list = [EarlyStopping(monitor='val_loss',
                                              min_delta=0,
                                              patience=early_stopping_patience,
                                              verbose=0,
                                              mode='auto',
                                              baseline=None,
                                              restore_best_weights=False
                                              )
                                ]
        else:
            _callbacks: list = None
        self.model.fit(x=self.train_generator,
                       batch_size=self.batch_size,
                       epochs=n_epoch,
                       verbose='auto',
                       callbacks=_callbacks,
                       validation_split=0.0,
                       validation_data=self.validation_generator,
                       shuffle=True,
                       class_weight=None,
                       sample_weight=None,
                       initial_epoch=0,
                       steps_per_epoch=None,
                       validation_steps=None,
                       validation_batch_size=None,
                       validation_freq=1,
                       max_queue_size=10,
                       workers=1,
                       use_multiprocessing=False
                       )
        self._eval_training(file_path=model_output_path)
        self._save_models(model_output_path=model_output_path)


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
