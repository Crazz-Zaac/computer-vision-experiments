from setuptools import setup, find_packages

setup(
    name='computer_vision_experiment',
    version='0.1',
    description='Computer Vision Experiment to explore different techniques',
    long_description=open('README.md').read(),
    author='Ramkrishna Acharya and Rabin BK',
    author_email='',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'matplotlib',
        'opencv-python',
        'scikit-image',
        'scikit-learn',
        'scipy',
        'torch',
        'torchvision',
        'tqdm',
    ],
    # entry_points={
    #     'console_scripts': [
    #         'run = computer_vision_experiment.run:main'
    #     ]
    # },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.7',
    ],
    include_package_data=True,
    # Indicates whether the package can be installed as a .egg file
    zip_safe=False,
    python_requires='>=3.7',
)