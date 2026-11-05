---
task_categories:
- object-detection
language:
- en
- zh
tags:
- fire-detection
- hard-negative
- flame-like
- yolo
pretty_name: FireDetectionDataset - Indoor Flame-like Negatives
---

# Indoor Flame-like Hard-Negative Collection Target

Taxonomy path: `FireDetectionDataset/negative/indoor/flame_like`.

## Status

**Collection target only. No accepted training samples are present.**

This repository is reserved for visually confusing scenes without real flame: warm lamps,
orange reflections, unlit candles and burners, LED decorations, screens, and welding arcs.

## Reviewed upstream candidate

[Kaggle: Yolo Fire and smoke dataset with red negatives](https://www.kaggle.com/datasets/ya5h10/yolo-fire-and-smoke-dataset-with-red-negatives) is Apache-2.0 and contains 23,926 images with 12,237 empty YOLO label files. A visual sample found mixed normal landscapes, fog, real smoke, and warm orange reflections. Empty labels are therefore not automatically safe flame negatives. The 3.08 GB source is linked for subset review and has not been mirrored here.

## Admission requirements

- Every accepted image must be verified to contain no real visible flame.
- YOLO exports use an empty label file for class `0=flame`.
- Preserve negative subtype and scene group in the manifest.
- Record source, license, revision, duplicate hash, and human review result.

