---
license: other
language:
- en
- zh
tags:
- fire-detection
- object-detection
- yolo
- dataset-catalog
pretty_name: FireDetectionDataset Catalog
---

# FireDetectionDataset Catalog

This repository is a metadata index. It does not contain training images.

Hugging Face dataset repository IDs support only `namespace/repository-name`, so the logical hierarchy is encoded with the naming rule:

```text
FireDetectionDataset-<category>-<subcategory>-<source>
```

## Logical taxonomy

```text
FireDetectionDataset/
|-- flame/
|   |-- indoor/
|   |   |-- candle_flame
|   |   |-- lighter_flame
|   |   |-- match_flame
|   |   |-- gas_burner
|   |-- outdoor/small_flame
|   |-- forest
|   `-- vehicle
|-- spark/
|   |-- welding
|   |-- grinding
|   `-- electrical
`-- negative/
    |-- indoor/flame_like
    |-- lighting
    |-- reflection
    |-- sunset
    |-- welding_arc
    |-- fog_smoke
    `-- exhaust
```

## Current datasets

| Taxonomy path | Managed dataset | Status |
|---|---|---|
| `flame/forest` | [FireDetectionDataset-flame-forest-flameye-wildfire](https://huggingface.co/datasets/baizhanquan/FireDetectionDataset-flame-forest-flameye-wildfire) | Candidate accepted for flame label review; CC BY 4.0 |
| `flame/outdoor/small_flame` | [FireDetectionDataset-flame-outdoor-small-flame-vn](https://huggingface.co/datasets/baizhanquan/FireDetectionDataset-flame-outdoor-small-flame-vn) | Small-object candidate; license review required |
| `flame/indoor/lighter_flame` | [FireDetectionDataset-flame-indoor-lighter-flame](https://huggingface.co/datasets/baizhanquan/FireDetectionDataset-flame-indoor-lighter-flame) | One boxed candidate; license review required |
| `flame/indoor/candle_flame` | [FireDetectionDataset-flame-indoor-candle-flame](https://huggingface.co/datasets/baizhanquan/FireDetectionDataset-flame-indoor-candle-flame) | One boxed candidate; license review required |
| `flame/indoor/match_flame` | [FireDetectionDataset-flame-indoor-match-flame](https://huggingface.co/datasets/baizhanquan/FireDetectionDataset-flame-indoor-match-flame) | Three independent boxed candidates; license review required |
| `flame/indoor/gas_burner` | [FireDetectionDataset-flame-indoor-gas-burner](https://huggingface.co/datasets/baizhanquan/FireDetectionDataset-flame-indoor-gas-burner) | Collection target; no qualified source found |
| `negative/indoor/flame_like` | [FireDetectionDataset-negative-indoor-flame-like](https://huggingface.co/datasets/baizhanquan/FireDetectionDataset-negative-indoor-flame-like) | Collection target; no accepted samples yet |

| 
lame/indoor/{candle,lighter,match} | [FireDetectionDataset-flame-indoor-characteristic-flame-barman](https://huggingface.co/datasets/baizhanquan/FireDetectionDataset-flame-indoor-characteristic-flame-barman) | 8,452-image YOLO source mirrored; CC BY-NC-SA 4.0; curation pending |

## Admission rule

A repository being listed here does not mean its samples are ready for training. Sources must pass license verification, label normalization, exact and perceptual deduplication, scene-group split review, and human label sampling before they enter `flame_clean_v1`.
