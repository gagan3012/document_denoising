"""
Build & train cycle-gan model (Generative Adversarial Network)
"""

from .utils import ImageProcessor, ReflectionPadding2D
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.engine import keras_tensor
from keras.initializers.initializers_v2 import (
    Constant, HeNormal, HeUniform, GlorotNormal, GlorotUniform, LecunNormal, LecunUniform, Ones, Orthogonal, RandomNormal, RandomUniform, TruncatedNormal, Zeros
)
from keras.layers import Activation, BatchNormalization, Concatenate, Dense, Dropout, Input, ReLU, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv2DTranspose
from keras.models import load_model, Model
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from typing import List

import datetime
import json
import keras
import keras.backend as K
import numpy as np
import os


class CycleGANException(Exception):
    """
    Class for handling exceptions for class CycleGAN
    """
    pass


class CycleGAN:
    """
    Class for building and training cycle-gan model
    """
    def __init__(self,
                 file_path_train_clean_images: str,
                 file_path_train_noisy_images: str,
                 n_channels: int = 1,
                 image_height: int = 256,
                 image_width: int = 256,
                 learning_rate: float = 0.0002,
                 optimizer: str = 'adam',
                 initializer: str = 'glorot_normal',
                 batch_size: int = 1,
                 start_n_filters_discriminator: int = 64,
                 max_n_filters_discriminator: int = 512,
                 n_conv_layers_discriminator: int = 3,
                 dropout_rate_discriminator: float = 0.0,
                 start_n_filters_generator: int = 32,
                 max_n_filters_generator: int = 512,
                 generator_type: str = 'res',
                 n_resnet_blocks: int = 9,
                 n_conv_layers_generator_res_net: int = 0,
                 n_conv_layers_generator_u_net: int = 3,
                 dropout_rate_generator_down_sampling: float = 0.0,
                 dropout_rate_generator_up_sampling: float = 0.0,
                 include_moe_layers: bool = True,
                 n_conv_layers_moe_embedder: int = 6,
                 dropout_rate_moe_embedder: float = 0.0,
                 n_hidden_layers_moe_fc_gated_net: int = 1,
                 n_hidden_layers_moe_fc_classifier: int = 1,
                 dropout_rate_moe_fc_gated_net: float = 0.0,
                 n_noise_types_moe_fc_classifier: int = 4,
                 dropout_rate_moe_fc_classifier: float = 0.0,
                 ignore_moe_fc_classifier: bool = True,
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

        :param start_n_filters_discriminator: int
            Number of filters used in first convolutional layer in discriminator network

        :param max_n_filters_discriminator: int
            Maximum number of filter used in all convolutional layers in discriminator network

        :param n_conv_layers_discriminator: int
            Number of convolutional layers in discriminator network

        :param dropout_rate_discriminator: float
            Dropout rate used after each convolutional layer in discriminator network

        :param start_n_filters_generator: int
            Number of filters used in first convolutional layer in generator network

        :param max_n_filters_generator: int
            Maximum number of filter used in all convolutional layers in generator network

        :param generator_type: str
            Abbreviated name of the type of the generator
                -> u: U-Network architecture
                -> resnet: Residual network architecture

        :param n_resnet_blocks: int
            Number of residual network blocks to use
                Common: -> 6, 9

        :param n_conv_layers_generator_res_net: int
            Number of convolutional layers used for down and up sampling

        :param n_conv_layers_generator_u_net: int
            Number of convolutional layers in generator network with u-net architecture

        :param dropout_rate_generator_down_sampling: float
            Dropout rate used after each convolutional layer in generator down-sampling network

        :param dropout_rate_generator_up_sampling: float
            Dropout rate used after each convolutional layer in generator up-sampling network

        :param include_moe_layers: bool
            Whether to use mixture of experts layers in residual network architecture

        :param n_conv_layers_moe_embedder: int
            Number of convolutional layers in discriminator network

        :param dropout_rate_moe_embedder: float
            Dropout rate used after each convolutional layer in mixture of experts embedder network

        :param n_hidden_layers_moe_fc_gated_net: int
            Number of hidden layers of the fully connected gated network used to process mixture of experts embedding output

        :param n_hidden_layers_moe_fc_classifier: int
            Number of hidden layers of the fully connected gated network used to classify mixture of experts embedding output (noise type)

        :param dropout_rate_moe_fc_gated_net: float
            Dropout rate used after each convolutional layer in mixture of experts fully connected gated network

        :param n_noise_types_moe_fc_classifier: int
            Number of classes (noise types) to classify

        :param dropout_rate_moe_fc_classifier: float
            Dropout rate used after each convolutional layer in mixture of experts fully connected classification network

        :param ignore_moe_fc_classifier: bool
            Whether to ignore mixture of experts fully connected classifier or not (because classifier is used for evaluation purposes only)

        :param print_model_architecture: bool
            Whether to print architecture of cycle-gan model components (discriminators & generators) or not
        """
        if len(file_path_train_clean_images) == 0:
            raise CycleGANException('File path for clean training document images is empty')
        if len(file_path_train_noisy_images) == 0:
            raise CycleGANException('File path for noisy training document images is empty')
        self.file_path_train_clean_data: str = file_path_train_clean_images
        self.file_path_train_noisy_data: str = file_path_train_noisy_images
        if 1 < n_channels < 4:
            self.n_channels: int = 3
        else:
            self.n_channels: int = 1
        self.image_height: int = image_height
        self.image_width: int = image_width
        self.image_shape: Input = Input(shape=(self.image_height, self.image_width, self.n_channels))
        self.learning_rate: float = learning_rate if learning_rate > 0 else 0.0002
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
        self.n_resnet_blocks: int = n_resnet_blocks if n_resnet_blocks > 0 else 9
        self.dropout_rate_generator_down_sampling: float = dropout_rate_generator_down_sampling if dropout_rate_generator_down_sampling >= 0 else 0.0
        self.dropout_rate_generator_up_sampling: float = dropout_rate_generator_up_sampling if dropout_rate_generator_up_sampling >= 0 else 0.0
        self.include_moe_layers: bool = include_moe_layers
        self.ignore_moe_fc_classifier: bool = ignore_moe_fc_classifier
        self.n_conv_layers_moe_embedder: int = n_conv_layers_moe_embedder
        self.dropout_rate_moe_embedder: float = dropout_rate_moe_embedder
        self.n_hidden_layers_moe_fc_gated_net: int = n_hidden_layers_moe_fc_gated_net
        self.n_hidden_layers_moe_fc_classifier: int = n_hidden_layers_moe_fc_classifier
        self.dropout_rate_moe_fc_gated_net: float = dropout_rate_moe_fc_gated_net
        self.n_noise_types_moe_fc_classifier: int = n_noise_types_moe_fc_classifier
        self.dropout_rate_moe_fc_classifier: float = dropout_rate_moe_fc_classifier
        self.print_model_architecture: bool = print_model_architecture
        self.discriminator_patch: tuple = None
        self.discriminator_A: Model = None
        self.discriminator_B: Model = None
        self.generator_A: Model = None
        self.generator_B: Model = None
        self.combined_model: Model = None
        self.model_name: str = None
        self.training_time: str = None
        self.elapsed_time: List[str] = None
        self.epoch: List[int] = []
        self.batch: List[int] = []
        self.discriminator_loss: List[float] = []
        self.discriminator_accuracy: List[float] = []
        self.generator_loss: List[float] = []
        self.adversarial_loss: List[float] = []
        self.reconstruction_loss: List[float] = []
        self.identy_loss: List[float] = []
        # Cycle-consistency loss:
        self.lambda_cycle: float = 10.0
        # Identity loss:
        self.lambda_id: float = 0.1 * self.lambda_cycle
        # Initialize ImageProcessor for loading and preprocessing image data (clean & noisy) in training:
        self.image_processor: ImageProcessor = ImageProcessor(file_path_clean_images=self.file_path_train_clean_data,
                                                              file_path_noisy_images=self.file_path_train_noisy_data,
                                                              n_channels=self.n_channels,
                                                              batch_size=self.batch_size,
                                                              image_resolution=(256, 256),
                                                              flip=True,
                                                              crop=None
                                                              )
        # Build Cycle-GAN Network:
        self._build_cycle_gan_network()

    def _build_discriminator(self) -> Model:
        """
        Build discriminator network

        :return: Model
            Discriminator network model
        """
        _n_filters: int = self.start_n_filters_discriminator
        _d = Conv2D(filters=_n_filters,
                    kernel_size=(4, 4),
                    strides=(2, 2),
                    padding='same',
                    kernel_initializer=self.initializer
                    )(self.image_shape)
        _d = self.normalizer()(_d)
        _d = LeakyReLU(alpha=0.2)(_d)
        for _ in range(0, self.n_conv_layers_discriminator, 1):
            if _n_filters < self.max_n_filters_discriminator:
                _n_filters *= 2
            _d = self._convolutional_layer_discriminator(input_layer=_d, n_filters=_n_filters)
        _d = ZeroPadding2D(padding=(1, 1))(_d)
        _d = Conv2D(filters=_n_filters,
                    kernel_size=(4, 4),
                    strides=(1, 1),
                    padding='valid',
                    kernel_initializer=self.initializer
                    )(_d)
        _d = self.normalizer()(_d)
        _d = LeakyReLU(alpha=0.2)(_d)
        _d = ZeroPadding2D(padding=(1, 1))(_d)
        _patch_out = Conv2D(filters=1,
                            kernel_size=(4, 4),
                            strides=(1, 1),
                            padding='valid',
                            kernel_initializer=self.initializer,
                            activation='sigmoid'
                            )(_d)
        self.discriminator_patch = (K.int_shape(_patch_out[0])[0], K.int_shape(_patch_out[0])[1], K.int_shape(_patch_out[0])[2])
        return Model(inputs=self.image_shape, outputs=_patch_out, name=self.model_name)

    def _build_cycle_gan_network(self):
        """
        Build complete cycle-gan network
        """
        # Build and compile the discriminators:
        self.model_name = 'discriminator_A'
        self.discriminator_A = self._build_discriminator()
        self.model_name = 'discriminator_B'
        self.discriminator_B = self._build_discriminator()
        if self.print_model_architecture:
            self.discriminator_A.summary()
        self.discriminator_A.compile(loss='mse',
                                     optimizer=self.optimizer,
                                     metrics=['accuracy'],
                                     loss_weights=[0.5]
                                     )
        self.discriminator_B.compile(loss='mse',
                                     optimizer=self.optimizer,
                                     metrics=['accuracy'],
                                     loss_weights=[0.5]
                                     )
        # Build the generators:
        self.model_name = 'generator_A'
        self.generator_A = self._build_generator()
        self.model_name = 'generator_B'
        self.generator_B = self._build_generator()
        if self.print_model_architecture:
            self.generator_A.summary()
        # Input images from both domains:
        _image_A: Input = Input(shape=(self.image_height, self.image_width, self.n_channels))
        _image_B: Input = Input(shape=(self.image_height, self.image_width, self.n_channels))
        # Translate images to the other domain:
        _fake_B = self.generator_A(_image_A)
        _fake_A = self.generator_B(_image_B)
        # Translate images back to original domain:
        _reconstruction_A = self.generator_B(_fake_B)
        _reconstruction_B = self.generator_A(_fake_A)
        # Identity mapping of images:
        _image_id_A = self.generator_B(_image_A)
        _image_id_B = self.generator_A(_image_B)
        # For the combined model we will only train the generators:
        self.discriminator_A.trainable = False
        self.discriminator_B.trainable = False
        # Discriminators determine validity of translated images:
        _valid_A = self.discriminator_A(_fake_A)
        _valid_B = self.discriminator_B(_fake_B)
        # Combined model to train generators to fool discriminators:
        self.combined_model = Model(inputs=[_image_A,
                                            _image_B
                                            ],
                                    outputs=[_valid_A,
                                             _valid_B,
                                             _reconstruction_A,
                                             _reconstruction_B,
                                             _image_id_A,
                                             _image_id_B
                                             ]
                                    )
        self.combined_model.compile(loss=['mse',
                                          'mse',
                                          'mae',
                                          'mae',
                                          'mae',
                                          'mae'
                                          ],
                                    loss_weights=[1,
                                                  1,
                                                  self.lambda_cycle,
                                                  self.lambda_cycle,
                                                  self.lambda_id,
                                                  self.lambda_id
                                                  ],
                                    optimizer=self.optimizer
                                    )

    def _build_generator(self) -> Model:
        """
        Build generator network

        :return: Model
            Generator network model
        """
        if self.generator_type == 'u':
            return self._u_net()
        elif self.generator_type == 'res':
            # Mixture of experts layers (MoE):
            if self.include_moe_layers:
                # Noise Type Embedding:
                _embedder = self._embedder()
                # Gated network:
                _gate = self._gated_network(input_layer=_embedder, units=64)
                # Noise Type Classifier:
                if not self.ignore_moe_fc_classifier:
                    _clf = self._moe_classifier(embedding_layer=_embedder)
            else:
                _embedder = None
                _gate = None
                _clf = None
            if self.include_moe_layers and self.n_conv_layers_generator_res_net == 0:
                _n_filters: int = self.start_n_filters_generator // 2
                _strides: tuple = (int(self.image_height / K.int_shape(_embedder[0])[0]), int(self.image_width / K.int_shape(_embedder[0])[1]))
            else:
                _n_filters: int = self.start_n_filters_generator
                _strides: tuple = (1, 1)
            #_g = ReflectionPadding2D(padding=(1, 1))(self.image_shape)
            _g = Conv2D(filters=self.start_n_filters_generator,
                        kernel_size=(7, 7),
                        strides=_strides,
                        padding='same',
                        kernel_initializer=self.initializer
                        )(self.image_shape)
            _g = self.normalizer()(_g)
            _g = ReLU(max_value=None, negative_slope=0, threshold=0)(_g)
            # Down-Sampling:
            for _ in range(0, self.n_conv_layers_generator_res_net, 1):
                if _n_filters < self.max_n_filters_generator:
                    _n_filters *= 2
                _g = self._convolutional_layer_generator_down_sampling(input_layer=_g, n_filters=_n_filters)
            # Residual Network Blocks:
            for _ in range(0, self.n_resnet_blocks, 1):
                _g = self._resnet_block(input_layer=_g, n_filters=_n_filters)
                if self.include_moe_layers:
                    _g = Concatenate()([_g, _gate])
            # Up-Sampling:
            for _ in range(0, self.n_conv_layers_generator_res_net, 1):
                if _n_filters > self.start_n_filters_generator:
                    _n_filters //= 2
                _g = self._convolutional_layer_generator_up_sampling(input_layer=_g, skip_layer=None, n_filters=_n_filters)
            if self.include_moe_layers and self.n_conv_layers_generator_res_net == 0:
                _g = UpSampling2D(size=int(self.image_height / K.int_shape(_g[0])[0]))(_g)
            #_g = ReflectionPadding2D(padding=(1, 1))(_g)
            _g = Conv2D(filters=1,
                        kernel_size=(7, 7),
                        strides=(1, 1),
                        padding='same',
                        kernel_initializer=self.initializer
                        )(_g)
            _g = self.normalizer()(_g)
            _fake_image = Activation('tanh')(_g)
            return Model(inputs=self.image_shape, outputs=_fake_image, name=self.model_name)

    def _convolutional_layer_discriminator(self, input_layer, n_filters: int) -> keras_tensor.KerasTensor:
        """
        Convolutional layer for discriminator

        :param input_layer:
            Network layer to process in the first convolutional layer

        :param n_filters: int
            Number of filters in the convolutional layer

        :return: keras_tensor.KerasTensor
            Processed keras tensor
        """
        _d = Conv2D(filters=n_filters,
                    kernel_size=(4, 4),
                    strides=(2, 2),
                    padding='same',
                    kernel_initializer=self.initializer
                    )(input_layer)
        _d = self.normalizer()(_d)
        return LeakyReLU(alpha=0.2)(_d)

    def _convolutional_layer_generator_down_sampling(self, input_layer, n_filters: int) -> keras_tensor.KerasTensor:
        """
        Convolutional layer for down sampling (u-net)

        :param input_layer:
            Network layer to process in the first convolutional layer

        :param n_filters: int
            Number of filters in the convolutional layer

        :return: keras_tensor.KerasTensor
            Processed keras tensor
        """
        _d = Conv2D(filters=n_filters,
                    kernel_size=(3, 3),
                    strides=(2, 2),
                    padding='same',
                    kernel_initializer=self.initializer
                    )(input_layer)
        _d = self.normalizer()(_d)
        return ReLU(max_value=None, negative_slope=0, threshold=0)(_d)

    def _convolutional_layer_generator_embedder(self, input_layer, n_filters: int) -> keras_tensor.KerasTensor:
        """
        Convolutional layer for embedding layer (mixture of experts)

        :param input_layer:
            Network layer to process in the first convolutional layer

        :param n_filters: int
            Number of filters in the convolutional layer

        :return: keras_tensor.KerasTensor
            Processed keras tensor
        """
        _e = Conv2D(filters=n_filters,
                    kernel_size=(3, 3),
                    strides=(1, 1),
                    padding='same',
                    kernel_initializer=self.initializer
                    )(input_layer)
        _e = self.normalizer()(_e)
        return ReLU(max_value=None, negative_slope=0, threshold=0)(_e)

    def _convolutional_layer_generator_up_sampling(self, input_layer, skip_layer, n_filters: int) -> keras_tensor.KerasTensor:
        """
        Convolutional layer for up sampling

        :param input_layer:
            Network layer to process in the first convolutional layer

        :param skip_layer:
            Network layer to concatenate with up-sampling output

        :param n_filters: int
            Number of filters in the (transposed) convolutional layer

        :return: keras_tensor.KerasTensor
            Processed keras tensor
        """
        if self.generator_type == 'u':
            _kernel_size: tuple = (4, 4)
        else:
            _kernel_size: tuple = (3, 3)
        _u = UpSampling2D(size=2)(input_layer)
        _u = Conv2DTranspose(filters=n_filters,
                             kernel_size=_kernel_size,
                             strides=(1, 1),
                             padding='same',
                             kernel_initializer=self.initializer
                             )(_u)
        if self.dropout_rate_generator_up_sampling > 0:
            _u = Dropout(rate=self.dropout_rate_generator_up_sampling, seed=1234)(_u)
        _u = self.normalizer()(_u)
        _u = ReLU(max_value=None, negative_slope=0, threshold=0)(_u)
        if self.generator_type == 'u':
            return Concatenate()([_u, skip_layer])
        else:
            return _u

    def _embedder(self) -> keras_tensor.KerasTensor:
        """
        Embedding (first part of the mixture of experts architecture)

        :return: keras_tensor.KerasTensor
            Processed keras tensor
        """
        _n_filters: int = self.image_height
        _e = Conv2D(filters=_n_filters,
                    kernel_size=(3, 3),
                    strides=(4, 4),
                    padding='same',
                    kernel_initializer=self.initializer
                    )(self.image_shape)
        _e = self.normalizer()(_e)
        _e = ReLU(max_value=None, negative_slope=0, threshold=0)(_e)
        for _ in range(0, self.n_conv_layers_moe_embedder, 1):
            if _n_filters > self.n_noise_types_moe_fc_classifier:
                _n_filters //= 2
            _e = self._convolutional_layer_generator_embedder(input_layer=_e, n_filters=_n_filters)
        return _e

    def _gated_network(self, input_layer, units: int = 64) -> keras_tensor.KerasTensor:
        """
        Fully connected gated network layer (part of the mixture of experts architecture)

        :param input_layer: keras_tensor.KerasTensor
            Mixture of experts embedding

        :param units: int
            Number of neurons

        :return: keras_tensor.KerasTensor
            Processed keras tensor
        """
        _fc = Dense(units=units, input_dim=self.image_height, activation='relu')(input_layer)
        if self.dropout_rate_moe_fc_gated_net > 0:
            _fc = Dropout(rate=self.dropout_rate_moe_fc_gated_net)(_fc)
        return Concatenate()([input_layer, _fc])

    def _moe_classifier(self, embedding_layer) -> keras_tensor.KerasTensor:
        """
        Classification layer of the mixture of experts embedding

        :param embedding_layer:
            Embedding input

        :return: keras_tensor.KerasTensor
            Probability tensor
        """
        _clf_model = Model(embedding_layer)
        _clf_model.append(Dense(units=64, input_dim=256, activation='relu'))
        _clf_model.append(Dense(units=32, activation='relu'))
        _clf_model.append(Dense(units=16, activation='relu'))
        _clf_model.append(Dense(units=self.n_noise_types_moe_fc_classifier, activation='softmax'))
        _clf_model.compile(optimizer='adam', metrics=['cross_entropy'])
        return _clf_model.predict(x=embedding_layer, batch_size=self.batch_size)

    def _resnet_block(self, input_layer, n_filters: int) -> keras_tensor.KerasTensor:
        """
        Residual network block

        :param input_layer:
            Network layer to process in the first convolutional layer

        :param n_filters: int
            Number of filters in the convolutional layer

        :return: keras_tensor.KerasTensor
            Processed keras tensor
        """
        #_r = ReflectionPadding2D(padding=(1, 1))(input_layer)
        _r = Conv2D(filters=n_filters,
                    kernel_size=(3, 3),
                    strides=(1, 1),
                    padding='same',
                    kernel_initializer=self.initializer
                    )(input_layer)
        _r = self.normalizer()(_r)
        _r = ReLU(max_value=None, negative_slope=0, threshold=0)(_r)
        #_r = ReflectionPadding2D(padding=(1, 1))(_r)
        _r = Conv2D(filters=n_filters,
                    kernel_size=(3, 3),
                    strides=(1, 1),
                    padding='same',
                    kernel_initializer=self.initializer
                    )(_r)
        _r = self.normalizer()(_r)
        return Concatenate()([_r, input_layer])

    def _save_training_report(self, report_output_path: str):
        """
        Save cycle-gan training report

        :param report_output_path: str
            Complete file path of the training report output
        """
        _cycle_gan_architecture: str = 'Discriminator: PatchGAN | '
        if self.generator_type == 'u':
            _cycle_gan_architecture = f'{_cycle_gan_architecture}Generator: U-Network'
        else:
            _cycle_gan_architecture = f'{_cycle_gan_architecture}Generator: Residual Network {self.n_resnet_blocks} Blocks'
            if self.include_moe_layers:
                _cycle_gan_architecture = f'{_cycle_gan_architecture} + Mixture of Experts Layers (MoE)'
        _cycle_gan_training_report: dict = dict(cycle_gan_architecture=_cycle_gan_architecture,
                                                training_time=self.training_time,
                                                elapsed_time=self.elapsed_time,
                                                learning_rate=self.learning_rate,
                                                epoch=self.epoch,
                                                batch=self.batch,
                                                batch_size=self.batch_size,
                                                batches=self.image_processor.n_batches,
                                                image_samples=self.batch_size * self.image_processor.n_batches,
                                                discriminator_loss=self.discriminator_loss,
                                                discriminator_accuracy=self.discriminator_accuracy,
                                                generator_loss=self.generator_loss,
                                                adversarial_loss=self.adversarial_loss,
                                                reconstruction_loss=self.reconstruction_loss,
                                                identity_loss=self.identy_loss
                                                )
        with open(os.path.join(metric_output_path, 'cycle_gan_training_report.json'), 'w', encoding='utf-8') as file:
            json.dump(obj=_cycle_gan_training_report, fp=file, ensure_ascii=False)

    def _save_models(self, model_output_path: str):
        """
        Save cycle-gan models

        :param model_output_path: str
            Complete file path of the model output
        """
        self.generator_A.save(filepath=os.path.join(model_output_path, 'generator_A.h5'))
        self.generator_B.save(filepath=os.path.join(model_output_path, 'generator_B.h5'))
        self.discriminator_A.save(filepath=os.path.join(model_output_path, 'discriminator_A.h5'))
        self.discriminator_B.save(filepath=os.path.join(model_output_path, 'discriminator_B.h5'))
        self.combined_model.save(filepath=os.path.join(model_output_path, 'combined_generator_model.h5'))

    def _u_net(self) -> Model:
        """
        U-network

        :return: Model
            U-network model
        """
        _n_filters: int = self.start_n_filters_generator
        _g = Conv2D(filters=_n_filters,
                    kernel_size=(3, 3),
                    strides=(2, 2),
                    padding='same',
                    kernel_initializer=self.initializer
                    )(self.image_shape)
        _g = self.normalizer()(_g)
        _g = ReLU(max_value=None, negative_slope=0, threshold=0)(_g)
        _u_net_layers: list = [_g]
        # Down-Sampling:
        for _ in range(0, self.n_conv_layers_generator_u_net, 1):
            if _n_filters < self.max_n_filters_generator:
                _n_filters *= 2
            _u_net_layers.append(self._convolutional_layer_generator_down_sampling(input_layer=_u_net_layers[-1],
                                                                                   n_filters=_n_filters
                                                                                   )
                                 )
        # Up-Sampling:
        for i in range(0, self.n_conv_layers_generator_u_net, 1):
            if _n_filters > self.start_n_filters_generator:
                _n_filters //= 2
            _i: int = -2 - (i * 2)
            _u_net_layers.append(self._convolutional_layer_generator_up_sampling(input_layer=_u_net_layers[-1],
                                                                                 skip_layer=_u_net_layers[_i],
                                                                                 n_filters=_n_filters
                                                                                 )
                                 )
        _g = UpSampling2D(size=2)(_u_net_layers[-1])
        _fake_image = Conv2D(filters=self.n_channels,
                             kernel_size=(7, 7),
                             strides=(1, 1),
                             padding='same',
                             kernel_initializer=self.initializer,
                             activation='tanh'
                             )(_g)
        return Model(inputs=self.image_shape, outputs=_fake_image, name=self.model_name)

    def inference(self,
                  file_path_generator: str,
                  file_path_noisy_images: str,
                  file_path_cleaned_images: str,
                  file_suffix: str = 'cleaned'
                  ):
        """
        Clean noisy document images based on training

        :param file_path_generator: str
            Complete file path of trained generator A

        :param file_path_noisy_images: str
            Complete file path of noisy images to clean

        :param file_path_cleaned_images: str
            Complete file path of the output (cleaned / denoised images)

        :param file_suffix: str
            Suffix of the output file name
        """
        _image_processor: ImageProcessor = ImageProcessor(file_path_clean_images='',
                                                          file_path_noisy_images=file_path_noisy_images,
                                                          n_channels=self.n_channels,
                                                          batch_size=self.batch_size,
                                                          image_resolution=(256, 256),
                                                          flip=False,
                                                          crop=None
                                                          )
        _generator_A: Model = load_model(filepath=file_path_generator,
                                         custom_objects=None,
                                         compile=True,
                                         options=None
                                         )
        for image_file_name, image in _image_processor.load_images():
            if file_suffix is None or len(file_suffix) == 0:
                _output_file_path: str = os.path.join(file_path_cleaned_images, image_file_name.split('/')[-1])
            else:
                _output_file_path: str = os.path.join(file_path_cleaned_images,
                                                      f"{file_suffix}_{image_file_name.split('/')[-1]}"
                                                      )
            _image_processor.save_image(image=_generator_A.predict(x=image), output_file_path=_output_file_path)

    def train(self,
              model_output_path: str,
              n_epoch: int = 300,
              checkpoint_epoch_interval: int = 5
              ):
        """
        Train cycle-gan models

        :param model_output_path: str
            Complete file path of the model output

        :param n_epoch: int
            Number of epochs to train

        :param checkpoint_epoch_interval: int
            Number of epoch intervals for saving model checkpoint
        """
        _t0: datetime = datetime.datetime.now()
        # Adversarial loss ground truths:
        _valid: np.array = np.ones((self.batch_size, ) + self.discriminator_patch)
        _fake: np.array = np.zeros((self.batch_size, ) + self.discriminator_patch)
        for epoch in range(n_epoch):
            for batch_i, (images_A, images_B) in enumerate(self.image_processor.load_batch()):
                # Translate images to opposite domain:
                _fake_B = self.generator_A.predict(images_A)
                _fake_A = self.generator_B.predict(images_B)
                # Train the discriminators (original images = real / translated = Fake):
                _discriminator_loss_real_A = self.discriminator_A.train_on_batch(images_A, _valid)
                _discriminator_loss_fake_A = self.discriminator_A.train_on_batch(_fake_A, _fake)
                _discriminator_loss_A = 0.5 * np.add(_discriminator_loss_real_A, _discriminator_loss_fake_A)
                _discriminator_loss_real_B = self.discriminator_B.train_on_batch(images_B, _valid)
                _discriminator_loss_fake_B = self.discriminator_B.train_on_batch(_fake_B, _fake)
                _discriminator_loss_B = 0.5 * np.add(_discriminator_loss_real_B, _discriminator_loss_fake_B)
                # Total discriminator loss:
                _discriminator_loss = 0.5 * np.add(_discriminator_loss_A, _discriminator_loss_B)
                # Train the generators:
                _generator_loss = self.combined_model.train_on_batch([images_A,
                                                                      images_B
                                                                      ],
                                                                     [_valid,
                                                                      _valid,
                                                                      images_A,
                                                                      images_B,
                                                                      images_A,
                                                                      images_B
                                                                      ]
                                                                     )
                _elapsed_time: datetime = datetime.datetime.now() - _t0
                self.elapsed_time.append(str(_elapsed_time))
                self.epoch.append(epoch)
                self.batch.append(batch_i)
                self.discriminator_loss.append(round(_discriminator_loss[0], 8))
                self.discriminator_accuracy.append(round(100 * _discriminator_loss[1], 4))
                self.generator_loss.append(round(_generator_loss[0], 8))
                self.adversarial_loss.append(round(np.mean(_generator_loss[1:3]), 8))
                self.reconstruction_loss.append(round(np.mean(_generator_loss[3:5]), 8))
                self.identy_loss.append(round(np.mean(_generator_loss[5:6]), 8))
                # Print training progress:
                _print_epoch_status: str = f'[Epoch: {epoch}/{n_epoch}]'
                _print_batch_status: str = f'[Batch: {batch_i}/{self.image_processor.n_batches}]'
                _print_discriminator_loss_status: str = f'[D loss: {self.discriminator_loss[-1]}, acc: {self.discriminator_accuracy[-1]}]'
                _print_generator_loss_status: str = f'[G loss: {self.generator_loss[-1]}, adv: {self.adversarial_loss[-1]}, recon: {self.reconstruction_loss[-1]}, id: {self.identy_loss[-1]}]'
                print(_print_epoch_status, _print_batch_status, _print_discriminator_loss_status, _print_generator_loss_status, f'time: {_elapsed_time}')
            # Save checkpoint:
            if (epoch % checkpoint_epoch_interval == 0) and (epoch > 0):
                self._save_models(model_output_path=model_output_path)
        # Save fully trained models:
        self._save_models(model_output_path=model_output_path)
        # Save training report:
        self.training_time = self.elapsed_time[-1]
        self._save_training_report(report_output_path=model_output_path)
