from setuptools import setup, find_packages

setup(
    name="vizcom",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "transformers>=4.30.0",
        "pillow>=9.5.0",
        "numpy>=1.24.0",
        "fastapi>=0.100.0",
        "uvicorn>=0.22.0",
        "python-multipart>=0.0.6",
        "python-dotenv>=1.0.0",
        "requests>=2.31.0"
    ],
    python_requires=">=3.8",
) 