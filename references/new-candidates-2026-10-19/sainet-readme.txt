---
license: cc-by-4.0
task_categories:
  - object-detection
tags:
  - wildfire
  - smoke
  - fire-detection
  - yolo
  - computer-vision
  - environmental
  - early-warning
language:
  - en
  - es
pretty_name: SAINetset - Wildfire Smoke Detection Dataset
size_categories:
  - 1K<n<10K
---

# SAINetset - Wildfire Smoke Detection Dataset

Dataset of real-world images captured by SAI (Sistema de Alerta de Incendios / Fire Alert System) surveillance nodes for wildfire smoke detection in Cordoba, Argentina.

**Current version:** v8.0 (January 2026)

## About SAI

The SAI (Fire Alert System) is an open-source early wildfire detection platform developed by [AlterMundi](https://sainet.info/), a civil association in Argentina. The system uses distributed camera nodes with [YOLO](https://docs.ultralytics.com/)-based AI (powered by [Ultralytics](https://ultralytics.com/)) to detect smoke in real-time and alert local firefighters via Telegram.

In its first month of operation (September 2025), SAI successfully detected **6 real wildfires** (including one at 32.8 km distance) with zero missed detections within the monitored area. Since then, the system has continued detecting fires regularly.

## Dataset Description

This dataset contains **only images from SAI nodes**. It is released separately from third-party datasets for licensing clarity.

### What's Included

| Split | Images | Positives | Negatives | Pos:Neg Ratio |
|-------|--------|-----------|-----------|---------------|
| Train | 4,462 | 200 | 4,262 | 1:21 |
| Val | 1,134 | 45 | 1,089 | 1:24 |
| **Total** | **5,596** | **245** | **5,351** | **1:22** |

### Content Types
- **True Positives (VP):** Real smoke/fire detections from operating nodes
- **False Positives (FP):** Hard negatives (urban lights, lens flare, atmospheric conditions)
- **Synthetic:** AI-generated smoke added to real backgrounds (photorealistic)
- **Donated:** Community-contributed images from the region

### Classes
| ID | Class |
|----|-------|
| 0 | smoke |

### Label Format
Standard YOLO format (`.txt` files):
```
<class_id> <x_center> <y_center> <width> <height>
```
All coordinates are normalized (0-1). Empty `.txt` files indicate negative samples (background).

## Full Training Dataset

To reproduce SAINet training, combine this dataset with:

1. **D-Fire Dataset** - [GitHub](https://github.com/gaiasd/DFireDataset)
   - ~21,000 images of fire and smoke
   - License: Check repository

2. **Pyro-SDIS Dataset** - [Hugging Face](https://huggingface.co/datasets/pyronear/pyro-sdis)
   - ~33,000 images from French fire services
   - License: Check repository

## Usage

### With Ultralytics YOLO

```python
from ultralytics import YOLO

# Train with this dataset
model = YOLO("yolo12m.pt")
model.train(data="path/to/sainetset/data.yaml", epochs=100)

# Or validate
model.val(data="path/to/sainetset/data.yaml")
```

### With Hugging Face Datasets

```python
from datasets import load_dataset

# Load the dataset
ds = load_dataset("SAINetset/SAINetset_v8.0")

# Access samples
sample = ds["train"][0]
print(sample)
```

### Direct Download

```bash
# Clone with Git LFS
git lfs install
git clone https://huggingface.co/datasets/SAINetset/SAINetset_v8.0
```

## Structure

```
sainetset/
├── README.md
├── data.yaml
└── data/
    ├── train/
    │   ├── images/    # 4,462 images
    │   └── labels/    # 4,462 label files
    └── val/
        ├── images/    # 1,134 images
        └── labels/    # 1,134 label files
```

## Related Resources

- **SAINet Model:** [GitHub - sai-sainet-latest](https://github.com/AlterMundi/sai-sainet-latest)
- **SAI Project:** [Website](https://sainet.info/)

## Citation

If you use this dataset, please cite:

```bibtex
@misc{sainetset2026,
  title={SAINetset: Wildfire Smoke Detection Dataset from SAI Nodes},
  author={AlterMundi},
  year={2026},
  publisher={Hugging Face},
  url={https://huggingface.co/datasets/SAINetset/SAINetset_v8.0}
}
```

## License

This dataset is released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

You are free to:
- **Share** - copy and redistribute the material
- **Adapt** - remix, transform, and build upon the material

Under the following terms:
- **Attribution** - You must give appropriate credit to AlterMundi

## Contact

- **Organization:** [AlterMundi](https://sainet.info/)
- **Project:** SAI - Sistema de Alerta de Incendios
