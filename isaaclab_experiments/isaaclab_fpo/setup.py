from setuptools import find_packages, setup

setup(
    name="isaaclab_fpo",
    version="0.2.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchvision",
        "numpy",
        # "GitPython",
        "onnx",
        "viser",
    ],
)
