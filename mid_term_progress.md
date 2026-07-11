# Real-Time School Bus Spotting in Smart Home Surveillance Video Using Image Processing and a Custom Convolutional Neural Network

**Course:** CS 898BA – Image Analysis and Computer Vision

**Name:** Ijeoma Nwosu

**Project Milestone:** Midterm Progress Report & Presentation

## Project Overview

Residential smart home surveillance cameras provide continuous monitoring of outdoor environments but lack the ability to recognize highly specific objects of interest to homeowners.

The objective of this project is to design and implement a computer vision system capable of automatically detecting school buses within residential surveillance imagery. Unlike general-purpose vehicle detection systems, this project focuses exclusively on binary classification:

- School Bus
- Background (everything else)

The original project proposal included a YOLO-based object detector. During implementation, the design was revised to comply with the course requirement prohibiting pretrained neural network models. The final implementation will therefore use a custom Convolutional Neural Network (CNN) built and trained entirely from scratch while preserving the proposed image-processing pipeline.

## Problem Statement

Parents often wait for school buses by manually observing security camera feeds or looking outside. Existing surveillance systems detect motion but cannot determine whether the approaching vehicle is actually the school bus.

The goal of this project is to automatically classify incoming images as either:

- School Bus
- Background

This provides the foundation for future real-time notification systems.

## Project Objectives

The objectives of this project are:

- Develop an end-to-end computer vision pipeline.
- Apply traditional image processing before deep learning.
- Build a CNN from scratch without pretrained weights.
- Train the model using a custom dataset.
- Evaluate model performance using standard classification metrics.

## Midterm Progress Summary

At the completion of the midterm milestone, the following tasks have been successfully completed.

## Completed

✔ Project proposal approved

✔ Development environment configured

✔ Python virtual environment created

✔ Required Python libraries installed

✔ Dataset collected

✔ Dataset validated

✔ Image loading implemented

✔ Image resizing implemented

✔ CLAHE enhancement implemented

✔ HSV color segmentation implemented

✔ Morphological filtering implemented

✔ Initial preprocessing pipeline validated

## Dataset

The project currently uses two image classes.

## Positive Class

School Bus Images

Current Count:

101 images collected

During validation:

- 17 unreadable images detected
- 84 valid images currently used

## Negative Class

Background Images

Current Count:

150 images

Includes:

- Cars
- SUVs
- Pickup trucks
- Empty streets
- Delivery trucks
- Other vehicles

## Project Directory Structure

```text
IjeomaNwosu-CS898BA-Project/

│
├── dataset/
│   ├── bus/
│   └── background/
│
├── processing/
│   ├── preprocess_images.py
│   └── yellow_mask.py
│
├── models/
│
├── outputs/
│
├── count_images.py
│
├── hello.
│
├── AI_log.md
│
├── mid_term_progress.md
│
├── README.md
│
└── venv/
##Development Environment

Operating System

Windows 10

Editor

Visual Studio Code

Programming Language

Python 3.10

Libraries

- OpenCV
- NumPy
- Matplotlib
- TensorFlow
- Pillow
- Scikit-learn

## Image Processing Pipeline

This project intentionally performs significant image processing before model training.

The pipeline currently consists of:

Input Image

↓

Image Validation

↓

Resize (128 × 128)

↓

CLAHE Contrast Enhancement

↓

HSV Color Conversion

↓

Yellow Color Segmentation

↓

Morphological Opening

↓

Morphological Closing

↓

Processed Image

↓

CNN (Next Phase)

## Image Validation

Before preprocessing, every image is checked using OpenCV.

Unreadable or corrupted files are automatically skipped.

Current results:

Bus Images Collected:

101

Successfully Loaded:

84

Skipped:

17

This validation prevents corrupted files from entering the training pipeline.

## CLAHE Enhancement

Contrast Limited Adaptive Histogram Equalization (CLAHE) is applied to improve local image contrast.

Purpose:

- Improve visibility
- Reduce illumination variation
- Enhance low-light images

## HSV Color Segmentation

Images are converted from RGB/BGR to HSV.

Yellow pixels associated with school buses are isolated using predefined HSV thresholds.

Benefits:

- Reduces background information
- Highlights the target object
- Improves feature extraction

## Morphological Filtering

After HSV masking:

Opening is used to remove isolated noise.

Closing reconnects fragmented regions of the bus.

Result:

Cleaner segmentation masks with reduced background artifacts.

## Current Results

Successfully implemented:

- Image loading
- Image validation
- Image visualization
- CLAHE enhancement
- HSV masking
- Morphological filtering

These preprocessing steps successfully isolate important visual characteristics of school buses while reducing background noise.

## Challenges Encountered

## Challenge 1

Unreadable Dataset Images

Problem

17 downloaded images could not be read by OpenCV.

Solution

Implemented image validation to automatically exclude corrupted files.

## Challenge 2

Course Restriction

Problem

Pretrained neural networks were not permitted.

Solution

Project architecture revised from YOLO-based detection to a custom CNN trained entirely from scratch.

## Challenge 3

Background Noise

Problem

HSV masking introduced isolated noisy regions.

Solution

Morphological opening and closing were added.

## Next Steps

The remaining work includes:

## Phase 1

Process every image automatically.

## Phase 2

Create processed dataset folders.

## Phase 3

Split data into:

- Training
- Validation
- Testing

## Phase 4

Design a custom CNN architecture.

## Phase 5

Train the CNN.

## Phase 6

Evaluate performance using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

## Expected Final Pipeline

Input Image

↓

Resize

↓

CLAHE

↓

HSV Conversion

↓

Yellow Mask

↓

Morphology

↓

Normalize

↓

Custom CNN

↓

Binary Classification

School Bus

or

Background

## Current Repository Status

| Component | Status |
| ----------- | -------- |
| Proposal | Complete |
| Dataset Collection | Complete |
| Dataset Validation | Complete |
| Image Loading | Complete |
| Image Resizing | Complete |
| CLAHE | Complete |
| HSV Segmentation | Complete |
| Morphological Filtering | Complete |
| Dataset Preparation | In Progress |
| CNN Development | Pending |
| CNN Training | Pending |
| Evaluation | Pending |
