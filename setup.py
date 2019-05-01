import setuptools

setuptools.setup(
    author="Allen Goodman",
    author_email="allen.goodman@icloud.com",
    extras_require={
        "benchmark": [
            "click",
            "sklearn"
        ],
        "test": [
            "pytest"
        ]
    },
    install_requires=[
        "keras>=2.2.4"
    ],
    license="MIT",
    name="keras-resnet",
    package_data={
        "keras-resnet": [
            "data/checkpoints/*/*.hdf5",
            "data/logs/*/*.csv"
        ]
    },
    packages=setuptools.find_packages(
        exclude=[
            "tests"
        ]
    ),
    url="https://github.com/broadinstitute/keras-resnet",
    version="0.2.0"
)
