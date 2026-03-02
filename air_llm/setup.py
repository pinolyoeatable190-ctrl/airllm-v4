"""AirLLM v4 — Setup. Removed scipy & optimum dependencies."""

import sys, subprocess
import setuptools
from setuptools.command.install import install


class PostInstall(install):
    def run(self):
        install.run(self)
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "transformers"])
        except subprocess.CalledProcessError:
            pass


with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setuptools.setup(
    name="airllm",
    version="4.0.0",
    author="Gavin Li",
    author_email="gavinli@animaai.cloud",
    description="AirLLM v4: optimized layer-streaming inference. 70B on 4GB GPU, 405B on 8GB VRAM.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lyogavin/airllm",
    packages=setuptools.find_packages(),
    install_requires=[
        'tqdm',
        'torch',
        'transformers',
        'accelerate',
        'safetensors',
        'huggingface-hub',
        # Removed: optimum (BetterTransformer deprecated, now using SDPA)
        # Removed: scipy (never used)
    ],
    cmdclass={'install': PostInstall},
    python_requires='>=3.8',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
