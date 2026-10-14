from setuptools import setup, find_packages

setup(
    name="oar_tracker_ps2",
    version="1.0.0",
    author="Zed Digital Assessment Team",
    description="Identity-Preserving Multi-Object Tracking through Dense Occlusion",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.22.0",
        "scipy>=1.10.0",
        "opencv-python>=4.7.0",
        "matplotlib>=3.7.0",
        "pyyaml>=6.0",
        "tabulate>=0.9.0"
    ]
)
