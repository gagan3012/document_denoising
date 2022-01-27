from document_denoising.cycle_gan import CycleGAN
from glob import glob
from typing import List

import unittest

FILE_PATH_CLEAN_IMAGES: str = './data/trainB'
FILE_PATH_NOISY_IMAGES: str = './data/trainA'


class TestCycleGAN(unittest.TestCase):
    """
    Test methods of class CycleGAN
    """
    def test_train_u_net(self):
        """
        Test cycle-gan model training (u-network as generator)
        """
        _n_epoch: int = 25
        _cycle_gan: CycleGAN = CycleGAN(file_path_train_clean_images=FILE_PATH_CLEAN_IMAGES,
                                        file_path_train_noisy_images=FILE_PATH_NOISY_IMAGES,
                                        n_channels=1,
                                        generator_type='u'
                                        )
        _cycle_gan.train(model_output_path='./data/results_u_net', n_epoch=_n_epoch, checkpoint_epoch_interval=5)
        self.assertTrue(expr=round(len(_cycle_gan.generator_loss) / _cycle_gan.image_processor.n_batches, 0) == _n_epoch)

    def test_train_res_net6(self):
        """
        Test cycle-gan model training (residual network 6 blocks as generator)
        """
        _n_epoch: int = 25
        _cycle_gan: CycleGAN = CycleGAN(file_path_train_clean_images=FILE_PATH_CLEAN_IMAGES,
                                        file_path_train_noisy_images=FILE_PATH_NOISY_IMAGES,
                                        n_channels=1,
                                        generator_type='res',
                                        n_resnet_blocks=6,
                                        include_moe_layers=False
                                        )
        _cycle_gan.train(model_output_path='./data/results_res_net6', n_epoch=_n_epoch, checkpoint_epoch_interval=5)
        self.assertTrue(expr=round(len(_cycle_gan.generator_loss) / _cycle_gan.image_processor.n_batches, 0) == _n_epoch)

    def test_train_res_net9(self):
        """
        Test cycle-gan model training (residual network 9 blocks as generator)
        """
        _n_epoch: int = 25
        _cycle_gan: CycleGAN = CycleGAN(file_path_train_clean_images=FILE_PATH_CLEAN_IMAGES,
                                        file_path_train_noisy_images=FILE_PATH_NOISY_IMAGES,
                                        n_channels=1,
                                        generator_type='res',
                                        n_resnet_blocks=9,
                                        include_moe_layers=False
                                        )
        _cycle_gan.train(model_output_path='./data/results_res_net9', n_epoch=_n_epoch, checkpoint_epoch_interval=5)
        self.assertTrue(expr=round(len(_cycle_gan.generator_loss) / _cycle_gan.image_processor.n_batches, 0) == _n_epoch)

    def test_train_res_net6_moe(self):
        """
        Test cycle-gan model training (residual network 6 blocks and mixture of experts layer as generator)
        """
        _n_epoch: int = 25
        _cycle_gan: CycleGAN = CycleGAN(file_path_train_clean_images=FILE_PATH_CLEAN_IMAGES,
                                        file_path_train_noisy_images=FILE_PATH_NOISY_IMAGES,
                                        n_channels=1,
                                        generator_type='res',
                                        n_resnet_blocks=6,
                                        include_moe_layers=True
                                        )
        _cycle_gan.train(model_output_path='./data/results_res_net6_moe', n_epoch=_n_epoch, checkpoint_epoch_interval=5)
        self.assertTrue(expr=round(len(_cycle_gan.generator_loss) / _cycle_gan.image_processor.n_batches, 0) == _n_epoch)

    def test_train_res_net9_moe(self):
        """
        Test cycle-gan model training (residual network 9 blocks and mixture of experts layer as generator)
        """
        _n_epoch: int = 25
        _cycle_gan: CycleGAN = CycleGAN(file_path_train_clean_images=FILE_PATH_CLEAN_IMAGES,
                                        file_path_train_noisy_images=FILE_PATH_NOISY_IMAGES,
                                        n_channels=1,
                                        generator_type='res',
                                        n_resnet_blocks=9,
                                        include_moe_layers=True
                                        )
        _cycle_gan.train(model_output_path='./data/results_res_net9_moe', n_epoch=_n_epoch, checkpoint_epoch_interval=5)
        self.assertTrue(expr=round(len(_cycle_gan.generator_loss) / _cycle_gan.image_processor.n_batches, 0) == _n_epoch)

    def test_inference(self):
        """
        Test inference of trained cycle-gan model
        """
        _noisy_test_images: List[str] = glob(f'./data/testA/*')
        _cycle_gan: CycleGAN = CycleGAN(file_path_train_clean_images=FILE_PATH_CLEAN_IMAGES,
                                        file_path_train_noisy_images=FILE_PATH_NOISY_IMAGES,
                                        batch_size=1,
                                        n_channels=1,
                                        print_model_architecture=False
                                        )
        _cycle_gan.inference(file_path_generator='./data/results_u_net/generator_A.h5',
                             file_path_noisy_images='./data/testA',
                             file_path_cleaned_images='./data/results_u_net/cleaned_images',
                             file_suffix='cleaned'
                             )
        _cleaned_test_images: List[str] = glob(f'./data/results_u_net/cleaned_images/*')
        self.assertTrue(expr=len(_noisy_test_images) == len(_cleaned_test_images))


if __name__ == '__main__':
    unittest.main()
