from distutils.core import setup
import setuptools

setup(name='ivy-demo-utils',
      version='0.0.0',
      description='Ivy Demo Utils provides a set of utilities for creating visual demos for Ivy libraries',
      author='Ivy Team',
      author_email='ivydl.team@gmail.com',
      packages=setuptools.find_packages(),
      package_data={'': ['*.npy', '*.ttt']},
      install_requires=['open3d'],
      classifiers=['License :: OSI Approved :: Apache Software License'],
      license='Apache 2.0'
      )
