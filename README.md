
# Signal Hump Analysis Toolkit

This project provides a pipeline for analyzing vibration signals from industrial devices and detecting abnormal humps in the frequency domain. It includes preprocessing, normalization, peak analysis, hump region detection, and sorting based on metadata timestamps.

## Features

- Load and parse signals from .txt files using a `TS` class
- Normalize and clip signal spectra
- Detect hump regions and analyze weighted energy
- Compare signals over time to detect hump shifting
- Sort multiple signals chronologically using metadata
- Print and debug outputs for each stage

## Usage

Provide an array of `TS` objects representing signals, or file paths to signal files. The pipeline will preprocess and analyze each signal, sort them by timestamp, and print relevant diagnostic outputs.

## Output

The script prints analysis results including detected hump ranges, their energy levels, and whether there's a downward trend over time.

## Requirements

- Python 3.x
- NumPy
- Matplotlib
- Scipy

