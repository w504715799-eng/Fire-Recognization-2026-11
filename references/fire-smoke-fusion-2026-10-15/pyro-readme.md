---
license: apache-2.0
dataset_info:
  features:
  - name: image
    dtype: image
  - name: annotations
    dtype: string
  - name: image_name
    dtype: string
  - name: partner
    dtype: string
  - name: camera
    dtype: string
  - name: date
    dtype: string
  splits:
  - name: train
    num_bytes: 1276375414.202
    num_examples: 11249
  - name: val
    num_bytes: 121624567.902
    num_examples: 1129
  download_size: 1318526032
  dataset_size: 1397999982.1039999
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: val
    path: data/val-*
tags:
- wildfire
- smoke
- yolo
- pyronear
- ultralytics
size_categories:
- 10K<n<100K
---

# Pyro-SDIS Dataset

![Pyronear Logo](https://huggingface.co/datasets/pyronear/pyro-sdis/resolve/main/logo.png)


---

## About the Dataset

Pyro-SDIS is a dataset designed for wildfire smoke detection using AI models. It is developed in collaboration with the Fire and Rescue Services (SDIS) in France and the dedicated volunteers of the Pyronear association.

The images in this dataset come from Pyronear cameras installed with the support of our SDIS partners. These images have been carefully annotated by Pyronear volunteers, whose tireless efforts we deeply appreciate. 

We extend our heartfelt thanks to all Pyronear volunteers and our SDIS partners for their trust and support:

- **Force 06**
- **SDIS 07**
- **SDIS 12**
- **SDIS 77**

Additionally, we express our gratitude to the DINUM for their financial and strategic support through the AIC, Etalab, and the Legal Service. Special thanks also go to the Mission Stratégie Prospective (MSP) for their guidance and collaboration.

Pyro-SDIS contains **12,378 images** (train 11,249, val 1,129), including:

- **10,744 images with smoke**
- **12,137 smoke instances**
- **1,634 images without smoke**, false positives of the Pyronear system

This dataset is formatted to be compatible with the Ultralytics YOLO framework, enabling efficient training of object detection models.

---

## Versions

`main` always holds the latest release. Previous releases stay available as tags:

| Version | Year | Images | Notes |
|---------|------|--------|-------|
| v2.0 | 2026 | 12,378 | Images from 2023 to 2025, false positives added |
| v1.0 | 2024 | 33,636 | Initial release |

To load a specific version:
```python
from datasets import load_dataset

dataset = load_dataset("pyronear/pyro-sdis", revision="v1.0")
```

Changes in v2.0:
- At most 10 images per smoke sequence, so consecutive near duplicate frames are no longer all kept. v2.0 therefore has fewer images than v1.0.
- 1,634 false positive images added, with empty annotations.
- New partners and cameras, images from 2023 to 2025.
- Smoke boxes use class `0` (v1.0 used `1`), matching `nc: 1` in `data.yaml`.


## Dataset Overview

### Contents
Pyro-SDIS contains images and annotations for wildfire smoke detection. The dataset is structured with the following metadata for each image:

- **Image Path**: File path to the image.
- **Annotations**: YOLO-format bounding box annotations for smoke detection:
  - `class_id`: always `0` (smoke). Images without smoke have empty annotations.
  - `x_center`, `y_center`: Normalized center coordinates of the bounding box.
  - `width`, `height`: Normalized width and height of the bounding box.
- **Metadata**:
  - `partner`: Partner organization responsible for the camera (e.g., SDIS 07, Force 06).
  - `camera`: Camera identifier.
  - `date`: Date of image capture (formatted as `YYYY-MM-DDTHH-MM-SS`).
  - `image_name`: Original file name of the image.
- **Split**: Indicates whether the image belongs to the training or validation set (`train` or `val`).

### Example Record
Each record in the dataset contains the following structure:
```json
{
  "image": "./images/train/partner_camera_date.jpg",
  "annotations": "0 0.5 0.5 0.1 0.2",
  "split": "train",
  "image_name": "partner_camera_date.jpg",
  "partner": "partner",
  "camera": "camera",
  "date": "YYYY-MM-DDTHH-MM-SS"
}
```

### Splits
The dataset is divided into:
- **Training split**: Used for training the model.
- **Validation split**: Used to evaluate model performance.



## Exporting the Dataset for Ultralytics Training

To train a YOLO model using the Ultralytics framework, the dataset must be structured as follows:
- **Images**: Stored in `images/train/` and `images/val/` directories.
- **Annotations**: Stored in YOLO-compatible format in `labels/train/` and `labels/val/` directories.

### Steps to Export the Dataset

1. **Install Required Libraries**:
   ```bash
   pip install datasets ultralytics
   ```

2. **Define Paths**:
   Set up the directory structure for the Ultralytics dataset:
   ```python
   import os
   from datasets import load_dataset

   # Define paths
   REPO_ID = "pyronear/pyro-sdis"
   OUTPUT_DIR = "./pyro-sdis"
   IMAGE_DIR = os.path.join(OUTPUT_DIR, "images")
   LABEL_DIR = IMAGE_DIR.replace("images", "labels")

   # Create the directory structure
   for split in ["train", "val"]:
       os.makedirs(os.path.join(IMAGE_DIR, split), exist_ok=True)
       os.makedirs(os.path.join(LABEL_DIR, split), exist_ok=True)

   # Load the dataset from the Hugging Face Hub
   dataset = load_dataset(REPO_ID)
   ```

3. **Export Dataset**:
   Use the following function to save the dataset in Ultralytics format:
   ```python
   def save_ultralytics_format(dataset_split, split):
       """
       Save a dataset split into the Ultralytics format.
       Args:
           dataset_split: The dataset split (e.g., dataset["train"])
           split: "train" or "val"
       """
       for example in dataset_split:
           # Save the image to the appropriate folder
           image = example["image"]  # PIL.Image.Image
           image_name = example["image_name"]  # Original file name
           output_image_path = os.path.join(IMAGE_DIR, split, image_name)

           # Save the image object to disk
           image.save(output_image_path)

           # Save label
           annotations = example["annotations"]
           label_name = image_name.replace(".jpg", ".txt").replace(".png", ".txt")
           output_label_path = os.path.join(LABEL_DIR, split, label_name)
           
           with open(output_label_path, "w") as label_file:
               label_file.write(annotations)

   # Save train and validation splits
   save_ultralytics_format(dataset["train"], "train")
   save_ultralytics_format(dataset["val"], "val")

   print("Dataset exported to Ultralytics format.")
   ```

4. **Directory Structure**:
   After running the script, the dataset will have the following structure:
   ```
   pyro-sdis/
   ├── images/
   │   ├── train/
   │   ├── val/
   ├── labels/
   │   ├── train/
   │   ├── val/
   ```

---

### Training with Ultralytics YOLO

1. **Download the `data.yaml` File**:
   Use the following code to download the configuration file:
   ```python
    from huggingface_hub import hf_hub_download
    
    # Correctly set repo_id and repo_type
    repo_id = "pyronear/pyro-sdis"
    filename = "data.yaml"
    
    # Download data.yaml to the current directory
    yaml_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset", local_dir=".")
    print(f"data.yaml downloaded to: {yaml_path}")
   ```

2. **Train the Model**:
   Install the Ultralytics YOLO framework and train the model:
   ```bash
   pip install ultralytics
   yolo task=detect mode=train data=data.yaml model=yolov8n.pt epochs=50 imgsz=640 single_cls=True
   ```


## License

The dataset is released under the [Apache-2.0 License](https://www.apache.org/licenses/LICENSE-2.0).

## Citation

If you use this dataset, please cite:
```
@dataset{pyro-sdis,
  author = {Pyronear Team},
  title = {Pyro-SDIS Dataset},
  year = {2026},
  version = {2.0},
  publisher = {Hugging Face},
  url = {https://huggingface.co/datasets/pyronear/pyro-sdis}
}
```