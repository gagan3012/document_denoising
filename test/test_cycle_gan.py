import unittest

from document_denoising.cycle_gan import CycleGAN

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
                                        generator_type='u'
                                        )
        _cycle_gan.train(model_output_path='./data/results_u_net', n_epoch=_n_epoch, checkpoint_epoch_interval=5)
        self.assertTrue(expr=(len(_cycle_gan.generator_loss) / _cycle_gan.image_processor.n_batches) == _n_epoch)

    def test_train_res_net6(self):
        """
        Test cycle-gan model training (residual network 6 blocks as generator)
        """
        _n_epoch: int = 25
        _cycle_gan: CycleGAN = CycleGAN(file_path_train_clean_images=FILE_PATH_CLEAN_IMAGES,
                                        file_path_train_noisy_images=FILE_PATH_NOISY_IMAGES,
                                        generator_type='res',
                                        n_resnet_blocks=6,
                                        include_moe_layers=False
                                        )
        _cycle_gan.train(model_output_path='./data/results_res_net6', n_epoch=_n_epoch, checkpoint_epoch_interval=5)
        self.assertTrue(expr=(len(_cycle_gan.generator_loss) / _cycle_gan.image_processor.n_batches) == _n_epoch)

    def test_train_res_net9(self):
        """
        Test cycle-gan model training (residual network 9 blocks as generator)
        """
        _n_epoch: int = 25
        _cycle_gan: CycleGAN = CycleGAN(file_path_train_clean_images=FILE_PATH_CLEAN_IMAGES,
                                        file_path_train_noisy_images=FILE_PATH_NOISY_IMAGES,
                                        generator_type='res',
                                        n_resnet_blocks=9,
                                        include_moe_layers=False
                                        )
        _cycle_gan.train(model_output_path='./data/results_res_net9', n_epoch=_n_epoch, checkpoint_epoch_interval=5)
        self.assertTrue(expr=(len(_cycle_gan.generator_loss) / _cycle_gan.image_processor.n_batches) == _n_epoch)

    def test_train_res_net6_moe(self):
        """
        Test cycle-gan model training (residual network 6 blocks and mixture of experts layer as generator)
        """
        _n_epoch: int = 25
        _cycle_gan: CycleGAN = CycleGAN(file_path_train_clean_images=FILE_PATH_CLEAN_IMAGES,
                                        file_path_train_noisy_images=FILE_PATH_NOISY_IMAGES,
                                        generator_type='res',
                                        n_resnet_blocks=6,
                                        include_moe_layers=True
                                        )
        _cycle_gan.train(model_output_path='./data/results_res_net6_moe', n_epoch=_n_epoch, checkpoint_epoch_interval=5)
        self.assertTrue(expr=(len(_cycle_gan.generator_loss) / _cycle_gan.image_processor.n_batches) == _n_epoch)

    def test_train_res_net9_moe(self):
        """
        Test cycle-gan model training (residual network 9 blocks and mixture of experts layer as generator)
        """
        _n_epoch: int = 25
        _cycle_gan: CycleGAN = CycleGAN(file_path_train_clean_images=FILE_PATH_CLEAN_IMAGES,
                                        file_path_train_noisy_images=FILE_PATH_NOISY_IMAGES,
                                        generator_type='res',
                                        n_resnet_blocks=9,
                                        include_moe_layers=True
                                        )
        _cycle_gan.train(model_output_path='./data/results_res_net9_moe', n_epoch=_n_epoch, checkpoint_epoch_interval=5)
        self.assertTrue(expr=(len(_cycle_gan.generator_loss) / _cycle_gan.image_processor.n_batches) == _n_epoch)

    def test_inference(self):
        """
        Test inference of trained cycle-gan model
        """
        pass


if __name__ == '__main__':
    unittest.main()
