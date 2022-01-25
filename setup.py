import setuptools
import subprocess
import sys

python_version: str = '3' if sys.platform.find('win') != 0 else ''

# Install jupyter notebook extensions:
subprocess.run([f'python{python_version} -m pip install jupyter_contrib_nbextensions && jupyter contrib nbextension install'], shell=True)

# Install keras-contrib:
subprocess.run([f'python{python_version} -m pip install git+https://www.github.com/keras-team/keras-contrib.git'], shell=True)

with open('README.md', 'r') as _read_me:
    long_description = _read_me.read()

with open('requirements.txt', 'r') as _requirements:
    requires = _requirements.read()

requires = [r.strip() for r in requires.split('\n') if ((r.strip()[0] != "#") and (len(r.strip()) > 3) and "-e git://" not in r)]

setuptools.setup(
    name='document_denoising',
    version='0.0.1',
    author='Gianni Francesco Balistreri',
    author_email='gbalistreri@gmx.de',
    description='Machine learning algorithms for denoising document images',
    long_description=long_description,
    long_description_content_type='text/markdown',
    keywords='computer vision machine learning gan autoencoder deep learning',
    license='GNU',
    url='https://github.com/GianniBalistreri/document_denoising',
    include_package_data=True,
    packages=setuptools.find_packages(),
    package_data={'document_denoising': ['LICENSE',
                                         'README.md',
                                         'requirements.txt',
                                         'setup.py'
                                         ]
                  },
    data_file=[('test', [
                         ]
                )],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    install_requires=requires
)
