---
license: other
pretty_name: RGBT-3M
language:
- en
task_categories:
- object-detection
tags:
- rgbt
- thermal
- yolo
- smoke-detection
- fire-detection
private: true
---

# RGBT-3M

RGB-T paired object detection dataset prepared for the local YOLOv12 dual-input workflow.

## Contents

- `RGBT-3M-full-20260604.tar`: full local dataset directory archive.
- `RGBT-3M-full-20260604.manifest.json`: packaging manifest with source path, file count, byte size, and SHA256 checksum.

## Classes

- `smoke`
- `fire`
- `person`

## Local Training Configuration

This archive corresponds to the local YOLOv12 dataset config:

`ultralytics/cfg/datasets/RGBT-3M.yaml`

The dataset is private pending license and source review.
