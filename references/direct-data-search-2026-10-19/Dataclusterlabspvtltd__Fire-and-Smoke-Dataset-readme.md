---
license: other
license_name: datacluster-commercial-sample
license_link: LICENSE
tags:
- fire-detection
- smoke-detection
- object-detection
- computer-vision
- surveillance
- safety
- sample-dataset
pretty_name: Fire and Smoke Detection Dataset (Sample)
size_categories:
- n<1K
task_categories:
- object-detection
- image-classification
---

# Fire and Smoke Detection Dataset — Sample

> ⚠️ **This is a free sample subset for evaluation purposes only.**  
> The full dataset (~10,000+ annotated images, multiple formats) is available for commercial licensing.  
> **Contact:** [sales@datacluster.ai](mailto:sales@datacluster.ai) · [datacluster.ai](https://datacluster.ai)

---

## Dataset Summary

This dataset contains real-world images of fire and smoke scenarios, captured on mobile phones under diverse conditions. It is well-suited for training early fire and smoke detection models used in smart cameras, fire alarm systems, surveillance, and safety applications.

Scenes include typical domestic situations such as garbage burning, paper and plastic burning, field crop burning, and domestic cooking — covering both indoor and outdoor environments under varied lighting and weather.

## Classes

- `fire`
- `smoke`

## Sample vs. Full Dataset

| | Sample (this repo) | Full Dataset |
|---|---|---|
| Images | ~100–200 (subset) | ~10,000+ |
| Annotation formats | Pascal VOC (XML) | COCO, YOLO, Pascal VOC, TF-Record |
| Scene diversity | Representative subset | Full range (indoor, outdoor, day, night, close, far) |
| Commercial use | ❌ Not permitted | ✅ With license |
| Redistribution | ❌ Not permitted | Per license terms |
| Updates | One-time | Ongoing |

**To license the full dataset:** [sales@datacluster.ai](mailto:sales@datacluster.ai)

## Dataset Structure

```
fire-and-smoke-sample/
├── images/                  # JPG images
│   ├── image_0001.jpg
│   └── ...
└── annotations/
    └── voc/                 # Pascal VOC XML annotations (one per image)
        ├── image_0001.xml
        └── ...
```

Each XML file contains bounding-box annotations in the Pascal VOC format, with filenames matching their corresponding image.

## Data Collection

- **Source:** Real-world mobile phone captures
- **Conditions:** Indoor and outdoor scenes; varied lighting, weather, distances, and vantage points
- **Use cases:** Early fire/smoke detection, anomaly detection, fire alarm systems, smart surveillance

## How to Use

### Download

```bash
# Using the Hugging Face CLI
huggingface-cli download datacluster-labs/fire-and-smoke-sample --repo-type dataset --local-dir ./fire-and-smoke-sample
```

Or clone directly:

```bash
git lfs install
git clone https://huggingface.co/datasets/Dataclusterlabspvtltd/fire-and-smoke-sample
```

### Convert VOC to YOLO or COCO

The sample ships in Pascal VOC format. Convert easily with `pylabel`:

```python
from pylabel import importer

# VOC → YOLO
dataset = importer.ImportVOC(path="annotations/voc")
dataset.export.ExportToYoloV5(output_path="annotations/yolo")

# VOC → COCO
dataset.export.ExportToCoco(output_path="annotations/coco/annotations.json")
```

## License

This sample dataset is released under a **restricted commercial-sample license** — see the `LICENSE` file in this repository.

Key points:
- ✅ Free to download and evaluate
- ✅ Free for academic research with attribution
- ❌ No commercial use without a license from DataCluster Labs
- ❌ No redistribution or re-upload
- ❌ No use in training commercial ML models

For commercial licensing of the full dataset, contact **sales@datacluster.ai**.

## Citation

If you use this dataset in academic work, please cite:

```bibtex
@misc{datacluster_fire_smoke_sample,
  title        = {Fire and Smoke Detection Dataset (Sample)},
  author       = {DataCluster Labs},
  year         = {2026},
  howpublished = {\url{https://huggingface.co/datasets/Dataclusterlabspvtltd/fire-and-smoke-sample}},
  note         = {Sample subset. Full dataset available for commercial licensing at sales@datacluster.ai}
}
```

## About DataCluster Labs

DataCluster Labs specializes in managed crowd-sourced data collection and annotation — images, videos, audio, text, and surveys — through our Dailydata platform. We deliver custom datasets for computer vision, NLP, and ML use cases.

📧 **Sales / Full Dataset Access:** sales@datacluster.ai  
🌐 **Website:** [datacluster.ai](https://datacluster.ai)