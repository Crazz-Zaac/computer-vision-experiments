from setuptools import setup

setup(
    name="cv_expt",
    version="0.1",
    description="Computer Vision Experiment to explore different techniques",
    long_description=open("README.md").read(),
    author="Ramkrishna Acharya and Rabin BK",
    author_email="",
    packages=["cv_expt"],
    install_requires=[
        "numpy",
        "matplotlib",
        "scikit-image",
        "scikit-learn",
        "scipy",
        "torch==2.2.0",
        "tqdm",
    ],
    # entry_points={
    #     'console_scripts': [
    #         'run = computer_vision_experiment.run:main'
    #     ]
    # },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.7",
    ],
    include_package_data=True,
    # Indicates whether the package can be installed as a .egg file
    zip_safe=False,
    python_requires=">=3.7",
)
