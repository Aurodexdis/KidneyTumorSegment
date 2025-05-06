# Kidney Tumor Segmentation Project

This repository contains code for automated kidney and tumor segmentation in CT images using the KiTS19 dataset, comparing adaptive thresholding and U-Net approaches.

## Project Overview

This project implements and evaluates two distinct approaches for kidney and tumor segmentation from abdominal CT images:
1. An anatomically-constrained adaptive thresholding method
2. A deep learning solution using the U-Net architecture

The implementation extracts 2D axial slices at the maximum tumor cross-sectional area from each 3D volume, preprocesses them using windowing and normalization, and evaluates segmentation performance using metrics such as Dice coefficient, precision, and recall.

## Repository Structure

### Python Scripts

- **setup.py**: Sets up the environment for the KiTS19 dataset analysis, installs required packages, clones the KiTS19 repository, and downloads imaging data.

- **eda.py**: Performs exploratory data analysis on the KiTS19 dataset, analyzing dataset structure, case dimensions, tumor distribution, and sample visualizations.

- **slice_extraction.py**: Extracts 2D slices from 3D CT volumes at the maximum tumor cross-sectional area, generating a dataset of representative slices for each case.

- **preprocess.py**: Preprocesses the extracted 2D slices using windowing to enhance kidney/tumor contrast and normalization to standardize image intensity distributions.

- **data_split.py**: Splits the preprocessed data into training, validation, and test sets using a stratified approach based on tumor size.

- **threshold.py**: Implements an adaptive thresholding method for kidney and tumor segmentation using intensity characteristics and anatomical constraints.

- **unet.py**: Implements a U-Net architecture for semantic segmentation of kidney and tumor regions, including model definition, training loop, and evaluation functions.

- **test_eval.py**: Evaluates both segmentation methods on the test dataset, calculating performance metrics and generating visualizations for comparison.

- **feat_extract_analysis.py**: Extracts and analyzes morphological features from kidney and tumor segmentation masks, including area, circularity, and tumor-kidney interface measurements.

### Jupyter Notebook

- **final_proj_copy.ipynb**: Main notebook that orchestrates the entire workflow, calling the individual Python scripts sequentially and displaying results at each stage of the pipeline.

## Getting Started

1. Clone this repository
2. Ensure you have Python 3.8+ installed
3. Run `python setup.py` to set up the environment and download the KiTS19 dataset
4. Run the Jupyter notebook `final_proj_copy.ipynb` to execute the full pipeline

## Results

The U-Net deep learning approach significantly outperforms the adaptive thresholding method, achieving 13-fold and 8-fold improvements in kidney and tumor segmentation Dice scores, respectively. Detailed results and visualizations are available in the notebook and output directories.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- KiTS19 Challenge organizers for providing the dataset and baseline code
- Heller et al. for creating and maintaining the KiTS19 dataset
