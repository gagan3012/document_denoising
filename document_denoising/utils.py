"""
Utility functions
"""

from glob import glob
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Activation, Dense, Dropout, Flatten
from keras.models import Model, Sequential
from keras.preprocessing.image import ImageDataGenerator
from skimage.transform import resize
from tensorflow import pad
from tensorflow.keras.layers import Layer
from typing import List, Tuple

import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import tensorflow as tf


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
    def _normalize(images: np.array) -> np.array:
        """
        Normalize images
        """
        return (images / 127.5) - 1.0

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
            _image = _image[self.crop[0][0]:self.crop[0][1], self.crop[1][0]:self.crop[1][1]]
        if self.image_resolution is not None:
            #_image = cv2.resize(src=_image, dsize=self.image_resolution, dst=None, fx=None, fy=None, interpolation=None)
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

    def load_images(self, n_images: int = None) -> Tuple[str, np.array]:
        """
        Load images without batching

        :return Tuple[List[str], np.array]
            List of image file names & array of loaded images
        """
        _images_path, _images_noisy = [], []
        _file_paths_noisy: List[str] = glob(f'{self.file_path_noisy_images}/*')
        if n_images is not None and n_images > 0:
            _file_paths_noisy_sample: List[str] = np.random.choice(_file_paths_noisy, n_images, replace=False)
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

    @staticmethod
    def save_image(image: np.array, output_file_path: str):
        """
        Save image

        :param image: np.array
            Image to save

        :param output_file_path: str
            Complete file path of the output image
        """
        fig = plt.figure()
        plt.imshow(image, cmap='Greys_r')
        fig.savefig(output_file_path)
        plt.close()
        #cv2.imwrite(filename=output_file_path, img=image, params=None)


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


class NoiseTypeClassifier(Model):
    """
    Class for training noise type classifier
    """
    def __init__(self,
                 file_path_train_images: str,
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
        super(NoiseTypeClassifier, self).__init__()
        if len(file_path_train_images) == 0:
            raise NoiseTypeClassifierException('No training images found')
        if len(file_path_validation_images) == 0:
            raise NoiseTypeClassifierException('No validation images found')
        if len(file_path_model_output) == 0:
            raise NoiseTypeClassifierException('No path for the trained model output found')
        self.file_path_train_images: str = file_path_train_images
        self.file_path_validation_images: str = file_path_validation_images
        self.file_path_model_output: str = file_path_model_output
        self.learning_rate: float = learning_rate if learning_rate > 0 else 0.001
        self.n_epoch: int = n_epoch if n_epoch > 0 else 10
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
        # Build neural network model:
        self.model: Sequential = Sequential()
        # 1. Convolutional Layer:
        self.model.add(Conv2D(32, (2, 2), input_shape=self.image_shape))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        # 2. Convolutional Layer:
        self.model.add(Conv2D(32, (2, 2)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        # 3. Convolutional Layer:
        self.model.add(Conv2D(32, (2, 2)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        # 4. Convolutional Layer:
        self.model.add(Conv2D(64, (2, 2)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        # 5. Convolutional Layer:
        self.model.add(Conv2D(64, (2, 2)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        # 6. Convolutional Layer:
        self.model.add(Conv2D(64, (2, 2)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        # 7. Convolutional Layer:
        self.model.add(Conv2D(64, (2, 2)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        # MLP Layer:
        self.model.add(Flatten())
        self.model.add(Dense(64))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(self.n_noise_types))
        if self.n_noise_types == 2:
            self.model.add(Activation('sigmoid'))
        else:
            self.model.add(Activation('softmax'))

    def call(self, x):
        return self.model(x)

    def train_step(self, data):
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}


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
