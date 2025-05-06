"""
Script to set up an environment for my Medical Image Analysis Final
Project using the 2019 Kidney and Kidney Tumor Segmentation Challenge
Dataset.
"""

# Copyright (c) 2025 Aurod Ounsinegad.
#
# This is free, open software released under the MIT License.  See
# `LICENSE` or https://choosealicense.com/licenses/mit/ for details.

# Import necessary packages
import os
import sys
import subprocess

def setup_env():
    """Set up the environment for the KiTS19 dataset analysis."""

    # Required packages
    packages = [
        'numpy',
        'pandas',
        'matplotlib',
        'scikit-learn',
        'scikit-image',
        'nibabel',
        'torch',
        'opencv-python',
        'tqdm'
    ]

    # Install required packages if needed
    for package in packages:
        try:
            __import__(package)
        except ImportError:
            print(f"Installing {package}")
            subprocess.call([sys.executable, "-m", "pip", "install", package])

    # Clone KiTS19 repository if not already present
    if not os.path.exists('kits19'):
        print("Cloning KiTS19 GitHub Repository")
        subprocess.call(["git", "clone", "https://github.com/neheller/kits19"])

        # Install requirements from the repository
        subprocess.call([sys.executable, "-m", "pip", "install", "-r", "kits19/requirements.txt"])
    else:
        print("KiTS19 GitHub Repository already exists")

    # Check if imaging data has been downloaded already
    if not os.path.exists('kits19/data/case_00000/imaging.nii.gz'):
        print("Downloading imaging data (this takes about 20 minutes)")
        os.chdir('kits19')
        subprocess.call([sys.executable, "-m", "starter_code.get_imaging"])
        os.chdir('..')
        print("Imaging data download completed")
    else:
        print("Imaging data already downloaded")

    # Add kits19 to the Python path
    if 'kits19' not in sys.path:
        sys.path.append('kits19')
