from setuptools import setup, find_packages

setup(
    name="openset_reid_ps1",
    version="1.0.0",
    author="Zed Digital Assessment Team",
    description="Open-Set Person Re-Identification under Low Inter-Class Variance",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.22.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.2.0",
        "opencv-python>=4.7.0",
        "onnx>=1.14.0",
        "onnxruntime>=1.15.0",
        "pyyaml>=6.0",
        "matplotlib>=3.7.0",
        "tabulate>=0.9.0"
    ]
)
