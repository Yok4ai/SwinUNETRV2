from setuptools import setup, find_packages

setup(
    name="swinunetrv2",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "monai-weekly[nibabel, tqdm, einops]",
        "matplotlib>=3.3.0",
        "einops>=0.3.0",
        "torch>=1.9.0",
        "torchvision>=0.10.0",
        "numpy>=1.19.2",
        "pandas>=1.2.0",
        "albumentations>=1.0.0",
        "opencv-python>=4.5.0",
        "tqdm>=4.50.0",
    ],
    python_requires=">=3.7",
    author="Imroz R",
    author_email="your.email@example.com",
    description="SwinUNETR V2 for medical image segmentation",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Yok4ai/SwinUNETRV2.git",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
) 