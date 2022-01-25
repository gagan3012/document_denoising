"""
Build & train cycle-gan model (Generative Adversarial Network)
"""

from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.initializers.initializers_v2 import (
    Constant, HeNormal, HeUniform, GlorotNormal, GlorotUniform, LecunNormal, LecunUniform, Ones, Orthogonal, RandomNormal, RandomUniform, Zeros
)
from keras.layers import Concatenate, Dense, Dropout, Flatten, Input, InputSpec, Layer, Reshape, ReLU
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv2DTranspose
from keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam

import datetime
import numpy as np
import keras
import sys
import os
import tensorflow as tf


class CycleGANException(Exception):
    """
    Class for handling exceptions of class CycleGAN
    """
    pass


class CycleGAN:
    """
    Class for building and training cycle-gan model
    """
    def __init__(self,
                 file_path_train_clean_data: str,
                 file_path_train_noisy_data: str,
                 file_path_test_clean_data: str = None,
                 file_path_test_noisy_data: str = None,
                 n_channels: int = 1,
                 image_height: int = 256,
                 image_width: int = 256,
                 n_epoch: int = 300,
                 learning_rate: float = 0.0002,
                 batch_size: int = 1,
                 initializer: str = 'xavier',
                 start_n_filters_discriminator: int = 64,
                 max_n_filters_discriminator: int = 512,
                 n_conv_layers_discriminator: int = 4,
                 dropout_rate_discriminator: float = 0.0,
                 start_n_filters_generator: int = 32,
                 max_n_filters_generator: int = 512,
                 generator_type: str = 'u',
                 n_conv_layers_generator_u_net: int = 4,
                 n_conv_layers_generator_res_net: int = 6,
                 dropout_rate_generator_down_sampling: float = 0.0,
                 dropout_rate_generator_up_sampling: float = 0.0,
                 include_moe_layers: bool = False,
                 n_conv_layers_moe_embedder: int = 7,
                 dropout_rate_moe_embedder: float = 0.0,
                 n_hidden_layers_moe_fc_gated_net: int = 1,
                 n_hidden_layers_moe_fc_classifier: int = 1,
                 dropout_rate_moe_fc_gated_net: float = 0.0,
                 n_noise_types_moe_fc_classifier: int = 4,
                 dropout_rate_moe_fc_classifier: float = 0.0,
                 ):
        """
        :param file_path_train_clean_data:
        :param file_path_train_noisy_data:
        :param file_path_test_clean_data:
        :param file_path_test_noisy_data:
        :param n_channels:
        :param image_height:
        :param image_width:
        :param n_epoch:
        :param learning_rate:
        :param batch_size:
        :param initializer:
        :param start_n_filters_discriminator:
        :param max_n_filters_discriminator:
        :param n_conv_layers_discriminator:
        :param dropout_rate_discriminator:
        :param start_n_filters_generator:
        :param max_n_filters_generator:
        :param generator_type:
        :param n_conv_layers_generator_u_net:
        :param n_conv_layers_generator_res_net:
        :param dropout_rate_generator_down_sampling:
        :param dropout_rate_generator_up_sampling:
        :param include_moe_layers:
        :param n_conv_layers_moe_embedder:
        :param dropout_rate_moe_embedder:
        :param n_hidden_layers_moe_fc_gated_net:
        :param n_hidden_layers_moe_fc_classifier:
        :param dropout_rate_moe_fc_gated_net:
        :param n_noise_types_moe_fc_classifier:
        :param dropout_rate_moe_fc_classifier:
        """
        if len(file_path_train_clean_data) == 0:
            raise CycleGANException('File path for clean training document images is empty')
        if len(file_path_train_noisy_data) == 0:
            raise CycleGANException('File path for noisy training document images is empty')
        self.file_path_train_clean_data: str = file_path_train_clean_data
        self.file_path_train_noisy_data: str = file_path_train_noisy_data
        self.n_channels: int = n_channels
        self.image_height: int = image_height
        self.image_width: int = image_width
        self.image_shape: Input = Input(shape=(self.image_height, self.image_width, self.n_channels))
        self.n_epoch: int = n_epoch
        self.learning_rate: float = learning_rate
        self.optimizer: Adam = Adam(learning_rate=learning_rate,
                                    beta_1=0.5,
                                    beta_2=0.999,
                                    epsilon=1e-7,
                                    amsgrad=False
                                    )
        if initializer == 'constant':
            self.initializer: keras.initializers.initializers_v2 = Constant(value=0)
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
        elif initializer == 'zeros':
            self.initializer: keras.initializers.initializers_v2 = Zeros()
        self.batch_size: int = batch_size
        self.start_n_filters_discriminator: int = start_n_filters_discriminator
        self.max_n_filters_discriminator: int = max_n_filters_discriminator
        self.n_conv_layers_discriminator: int = n_conv_layers_discriminator
        self.dropout_rate_discriminator: float = dropout_rate_discriminator if dropout_rate_discriminator >= 0 else 0.0
        self.start_n_filters_generator: int = start_n_filters_generator
        self.max_n_filters_generator: int = max_n_filters_generator
        if generator_type in ['u', 'res']:
            self.generator_type: str = generator_type
        else:
            self.generator_type: str = 'u'
        self.n_conv_layers_generator_u_net: int = n_conv_layers_generator_u_net
        self.n_conv_layers_generator_res_net: int = n_conv_layers_generator_res_net
        self.dropout_rate_generator_down_sampling: float = dropout_rate_generator_down_sampling if dropout_rate_generator_down_sampling >= 0 else 0.0
        self.dropout_rate_generator_up_sampling: float = dropout_rate_generator_up_sampling if dropout_rate_generator_up_sampling >= 0 else 0.0
        self.include_moe_layers: bool = include_moe_layers
        self.n_conv_layers_moe_embedder: int = n_conv_layers_moe_embedder
        self.dropout_rate_moe_embedder: float = dropout_rate_moe_embedder
        self.n_hidden_layers_moe_fc_gated_net: int = n_hidden_layers_moe_fc_gated_net
        self.n_hidden_layers_moe_fc_classifier: int = n_hidden_layers_moe_fc_classifier
        self.dropout_rate_moe_fc_gated_net: float = dropout_rate_moe_fc_gated_net
        self.n_noise_types_moe_fc_classifier: int = n_noise_types_moe_fc_classifier
        self.dropout_rate_moe_fc_classifier: float = dropout_rate_moe_fc_classifier

    def _build_discriminator(self) -> Model:
        """
        Build discriminator

        :return: Model
            Compiled discriminator model
        """
        d = Conv2D(filters=64, kernel_size=(4, 4), strides=(2, 2), padding='same', kernel_initializer=self.initializer)(self.image_shape)
        d = LeakyReLU(alpha=0.2)(d)
        d = Conv2D(filters=128, kernel_size=(4, 4), strides=(2, 2), padding='same', kernel_initializer=self.initializer)(d)
        d = InstanceNormalization(axis=-1)(d)
        d = LeakyReLU(alpha=0.2)(d)
        d = Conv2D(filters=256, kernel_size=(4, 4), strides=(2, 2), padding='same', kernel_initializer=self.initializer)(d)
        d = InstanceNormalization(axis=-1)(d)
        d = LeakyReLU(alpha=0.2)(d)
        d = Conv2D(filters=512, kernel_size=(4, 4), strides=(2, 2), padding='same', kernel_initializer=self.initializer)(d)
        d = InstanceNormalization(axis=-1)(d)
        d = LeakyReLU(alpha=0.2)(d)
        d = Conv2D(filters=512, kernel_size=(4, 4), padding='same', kernel_initializer=self.initializer)(d)
        d = InstanceNormalization(axis=-1)(d)
        d = LeakyReLU(alpha=0.2)(d)
        patch_out = Conv2D(filters=1, kernel_size=(4, 4), padding='same', kernel_initializer=self.initializer)(d)
        model = Model(inputs=self.image_shape, outputs=patch_out)
        model.compile(loss='mse', optimizer=self.optimizer, loss_weights=[0.5])
        return model
