---
license: cc-by-nc-sa-4.0
task_categories:
- object-detection
language:
- en
- zh
tags:
- fire-detection
- flame
- candle-flame
- lighter-flame
- match-flame
- yolo
pretty_name: Characteristic Flame - Candle Lighter Match
---

# Characteristic Flame Source: Candle, Lighter and Match

Taxonomy paths:

- `FireDetectionDataset/flame/indoor/candle_flame`
- `FireDetectionDataset/flame/indoor/lighter_flame`
- `FireDetectionDataset/flame/indoor/match_flame`

## Source

- Kaggle: [Flame Dataset (Candle, Lighter, Match Stick Flames)](https://www.kaggle.com/datasets/sreemantabarman/flame-dataset-candlelightermatch-stick-flames)
- Publisher: Sreemanta Barman
- Source version: 1
- Retrieved: 2026-08-20
- Kaggle license: CC BY-NC-SA 4.0
- Embedded Roboflow project: `flame-bd9xu/fire-detection-gvj54`, version 5

The embedded `data.yaml` says `CC BY 4.0`, while the Kaggle distribution page says
`CC BY-NC-SA 4.0`. This mirror conservatively applies the more restrictive Kaggle license.
Commercial training is not permitted without separate permission.

## Structure audit

- 8,452 JPEG images
- 8,322 paired YOLO label files
- 130 images without label files, treated as negative candidates
- One source class: `0=Fire`; project mapping: `0=flame`
- Splits: 7,907 train, 444 validation, 101 test
- ZIP path-safety check: passed

The archive is preserved without re-splitting because filenames do not reliably identify
candle, lighter, or match scenes. Scene subtype assignment, duplicate removal, visual label
review, and train/validation/test leakage review are required before training use.

## Files

- `archive.zip`: unchanged Kaggle version 1 distribution
- `data.yaml`: YOLO dataset definition extracted from the archive for inspection
- `source_manifest.yaml`: provenance, license, counts, and admission state

