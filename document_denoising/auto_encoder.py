"""
Build & train auto-encoder model for document image denoising using paired images
"""

from document_denoising.utils import ImageProcessor
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.engine import keras_tensor
from keras.initializers.initializers_v2 import (
    Constant, HeNormal, HeUniform, GlorotNormal, GlorotUniform, LecunNormal, LecunUniform, Ones, Orthogonal, RandomNormal, RandomUniform, TruncatedNormal, Zeros
)
from keras.layers import BatchNormalization, Dropout, Input, ReLU
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam, RMSprop, SGD

import json
import numpy as np
import keras
import os
import tensorflow as tf


class AutoEncoderException(Exception):
    """
    Class for handling exceptions for class AutoEncoder
    """
    pass


class AutoEncoder:
    """
    Class for building and training auto-encoder model
    """
    def __init__(self,
                 file_path_train_clean_images: str,
                 file_path_train_noisy_images: str,
                 n_channels: int = 1,
                 image_height: int = 256,
                 image_width: int = 256,
                 learning_rate: float = 0.001,
                 optimizer: str = 'adam',
                 initializer: str = 'he_normal',
                 batch_size: int = 32,
                 start_n_filters: int = 64,
                 n_conv_layers: int = 3,
                 dropout_rate: float = 0.0,
                 print_model_architecture: bool = True
                 ):
        """
        :param file_path_train_clean_images: str
            Complete file path of clean images for training

        :param file_path_train_noisy_images: str
            Complete file path of noisy images for training

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

        :param n_conv_layers: int
            Number of convolutional layer in auto-encoder network

        :param dropout_rate: float
            Dropout rate used after encoder and decoder in auto-encoder network

        :param print_model_architecture: bool
            Whether to print architecture of cycle-gan model components (discriminators & generators) or not
        """
        if len(file_path_train_clean_images) == 0:
            raise AutoEncoderException('File path for clean training document images is empty')
        if len(file_path_train_noisy_images) == 0:
            raise AutoEncoderException('File path for noisy training document images is empty')
        self.file_path_train_clean_data: str = file_path_train_clean_images
        self.file_path_train_noisy_data: str = file_path_train_noisy_images
        if 1 < n_channels < 4:
            self.n_channels: int = 3
        else:
            self.n_channels: int = 1
        self.image_height: int = image_height
        self.image_width: int = image_width
        self.image_shape: tuple = tuple([self.image_width, self.image_height, self.n_channels])
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
        self.n_conv_layers: int = n_conv_layers if n_conv_layers > 0 else 3
        self.dropout_rate: float = dropout_rate if dropout_rate >= 0 else 0.0
        self.print_model_architecture: bool = print_model_architecture
        self.model: Model = None
        self.image_processor: ImageProcessor = ImageProcessor(file_path_clean_images=self.file_path_train_clean_data,
                                                              file_path_noisy_images=self.file_path_train_noisy_data,
                                                              file_path_multi_noisy_images=None,
                                                              n_channels=self.n_channels,
                                                              batch_size=self.batch_size,
                                                              image_resolution=(self.image_width, self.image_height),
                                                              normalize=True if self.n_channels > 1 else False,
                                                              flip=True,
                                                              crop=None,
                                                              file_type=None
                                                              )
        self._build_auto_encoder_network()

    def _build_auto_encoder_network(self):
        """
        Build auto-encoder network
        """
        _n_filters: int = self.start_n_filters
        _input: Input = Input(shape=self.image_shape)
        # Encoder:
        _encoder: keras_tensor.KerasTensor = Conv2D(filters=_n_filters,
                                                    kernel_size=(3, 3),
                                                    strides=(1, 1),
                                                    padding='same',
                                                    kernel_initializer=self.initializer
                                                    )(_input)
        _encoder = ReLU(max_value=None, negative_slope=0, threshold=0)(_encoder)
        for _ in range(0, self.n_conv_layers - 1, 1):
            _n_filters *= 2
            _encoder = Conv2D(filters=_n_filters,
                              kernel_size=(3, 3),
                              strides=(1, 1),
                              padding='same',
                              kernel_initializer=self.initializer
                              )(_encoder)
            _encoder = ReLU(max_value=None, negative_slope=0, threshold=0)(_encoder)
        _encoder = self.normalizer()(_encoder)
        _encoder = MaxPooling2D(pool_size=(2, 2), strides=None, padding='same')(_encoder)
        if self.dropout_rate > 0:
            _encoder = Dropout(rate=self.dropout_rate)(_encoder)
        # Decoder:
        _decoder: keras_tensor.KerasTensor = Conv2D(filters=_n_filters,
                                                    kernel_size=(3, 3),
                                                    strides=(1, 1),
                                                    padding='same',
                                                    kernel_initializer=self.initializer
                                                    )(_encoder)
        _decoder = ReLU(max_value=None, negative_slope=0, threshold=0)(_decoder)
        for _ in range(0, self.n_conv_layers, 1):
            _n_filters /= 2
            _decoder = Conv2D(filters=_n_filters,
                              kernel_size=(3, 3),
                              strides=(1, 1),
                              padding='same',
                              kernel_initializer=self.initializer
                              )(_decoder)
            _decoder = ReLU(max_value=None, negative_slope=0, threshold=0)(_decoder)
        _decoder = self.normalizer()(_decoder)
        _decoder = UpSampling2D(size=(2, 2), interpolation='bilinear')(_decoder)
        _decoder = Conv2D(filters=1,
                          kernel_size=(3, 3),
                          strides=(1, 1),
                          padding='same',
                          kernel_initializer=self.initializer,
                          activation='sigmoid'
                          )(_decoder)
        self.model = Model(inputs=_input, outputs=_decoder, name='auto_encoder')
        if self.print_model_architecture:
            self.model.summary()
        self.model.compile(optimizer=self.optimizer,
                           loss='binary_crossentropy',
                           #metrics=[tf.keras.metrics.MeanSquaredError()],
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
        _image_processor: ImageProcessor = ImageProcessor(file_path_clean_images='',
                                                          file_path_noisy_images=self.file_path_train_noisy_data,
                                                          n_channels=self.n_channels,
                                                          batch_size=self.batch_size,
                                                          image_resolution=(self.image_width, self.image_height),
                                                          normalize=True if self.n_channels > 1 else False,
                                                          flip=False,
                                                          crop=None,
                                                          file_type=None
                                                          )
        for image_noisy_file_path, image_noisy in _image_processor.load_images(n_images=1):
            _fake_noisy: np.array = self.model.predict(image_noisy)
            _output_file_path_fake: str = os.path.join(file_path,
                                                       f"test_fake_{image_noisy_file_path[0].split('/')[-1]}"
                                                       )
            _output_file_path_noisy: str = os.path.join(file_path,
                                                        f"test_noise_{image_noisy_file_path[0].split('/')[-1]}"
                                                        )
            self.image_processor.save_image(image=np.array(_fake_noisy).squeeze(), output_file_path=_output_file_path_fake)
            self.image_processor.save_image(image=np.array(image_noisy).squeeze(), output_file_path=_output_file_path_noisy)
            print(f'Save evaluation image: {_output_file_path_fake}')

    def _save_models(self, model_output_path: str):
        """
        Save cycle-gan models

        :param model_output_path: str
            Complete file path of the model output
        """
        self.model.save(filepath=os.path.join(model_output_path, 'document_denoising_auto_encoder.h5'))

    def train(self,
              model_output_path: str,
              n_epoch: int = 30,
              early_stopping_patience: int = 0
              ):
        """
        Train auto-encoder model

        :param model_output_path: str
            Complete file path of the model output

        :param n_epoch: int
            Number of epochs to train

        :param early_stopping_patience: int
            Number of unchanged gradient epoch intervals for early stopping
        """
        _images_clean, _images_noisy = self.image_processor.load_all_images()
        _x_train, _x_valid, _y_train, _y_valid = train_test_split(_images_noisy,
                                                                  _images_clean,
                                                                  test_size=0.2,
                                                                  train_size=0.8,
                                                                  random_state=1234,
                                                                  shuffle=True,
                                                                  stratify=None
                                                                  )
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
        self.model.fit(x=_x_train,
                       y=_y_train,
                       batch_size=self.batch_size,
                       epochs=n_epoch,
                       verbose='auto',
                       callbacks=_callbacks,
                       validation_split=0.0,
                       validation_data=(_x_valid, _y_valid),
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
