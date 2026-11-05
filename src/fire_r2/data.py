from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile

from .common import ProjectError, child, file_lock, output_path, read_json, sha256, write_json

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def parse_yolo(text, class_map):
    boxes = []
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        values = line.split()
        if len(values) != 5 or values[0] not in class_map:
            raise ProjectError(f"Invalid YOLO class/fields on line {number}")
        try:
            xywh = [float(v) for v in values[1:]]
        except ValueError as exc:
            raise ProjectError(f"Non-numeric YOLO box on line {number}") from exc
        if not all(math.isfinite(x) for x in xywh):
            raise ProjectError(f"Non-finite YOLO box on line {number}")
        boxes.append({"class_id": class_map[values[0]], "bbox": xywh,
                      "coordinates": "normalized_xywh", "original_class_id": int(values[0])})
    return boxes


def assign_roles(rows, spec, salt):
    policy = spec["split_policy"]
    if policy == "external_only":
        for row in rows:
            row["role"] = "external"
        return
    source = "train" if policy == "train_80_10_10" else "val"
    selected = [r for r in rows if r["official_split"] == source]
    def key(row):
        payload = f'{salt}|{spec["split_key"]}|{source}|{row["file"]}'
        return hashlib.sha256(payload.encode()).hexdigest(), row["file"]
    selected.sort(key=key)
    n = len(selected)
    if policy == "train_80_10_10":
        n_fit, n_dev = n * 8 // 10, n // 10
        for i, row in enumerate(selected):
            row["role"] = "fit" if i < n_fit else "dev" if i < n_fit+n_dev else "cal"
    elif policy == "val_half":
        for i, row in enumerate(selected):
            row["role"] = "dev" if i < n//2 else "cal"
    else:
        raise ProjectError(f"Unknown split policy {policy}")
    for row in rows:
        if row["official_split"] != source:
            row["role"] = "fit" if row["official_split"] == "train" else "test"


def _image_path(root, relative):
    p = child(root, relative)
    if p.is_file():
        return p
    for ext in IMAGE_EXTENSIONS:
        candidate = p.with_suffix(ext)
        if candidate.is_file():
            return candidate
    return p


def load_dataset(project_root, config_path, dataset, dataset_root=None):
    config = read_json(config_path)
    if dataset not in config["datasets"]:
        raise ProjectError(f"Unknown dataset: {dataset}")
    spec = config["datasets"][dataset]
    shared = Path(dataset_root or os.environ.get("FIRE_DATASETS_ROOT") or Path(project_root)/"datasets").resolve()
    labels = child(shared, spec["annotation_root"])
    images = child(shared, spec["image_root"])
    if not labels.exists():
        raise ProjectError(f"Annotations not available: {labels}; no download attempted")
    rows, sources = [], {}
    if spec["format"] == "yolo":
        for split in spec["official_splits"]:
            folder = labels/split/"labels"
            files = sorted(folder.glob("*.txt"))
            if not files:
                raise ProjectError(f"No original labels under {folder}")
            for p in files:
                raw = p.read_bytes()
                sources[str(p.relative_to(labels))] = hashlib.sha256(raw).hexdigest()
                image_name = f"{split}/images/{p.stem}.jpg"
                rows.append({"file": image_name, "official_split": split,
                             "label_file": str(p.relative_to(labels)),
                             "boxes": parse_yolo(raw.decode("utf-8-sig"), spec["class_map"])})
    elif spec["format"] == "coco":
        for split in spec["official_splits"]:
            p = labels/f"{split}.json"
            if not p.is_file():
                raise ProjectError(f"COCO annotation file missing: {p}")
            sources[p.name] = sha256(p)
            obj = read_json(p)
            categories = {str(c["id"]): c["name"] for c in obj["categories"]}
            expected = {k: config["names"][v] for k, v in spec["class_map"].items()}
            if categories != expected:
                raise ProjectError(f"COCO category mismatch: {categories} vs {expected}")
            indexed = {im["id"]: im for im in obj["images"]}
            if len(indexed) != len(obj["images"]):
                raise ProjectError("Duplicate COCO image IDs within an official split")
            anns = defaultdict(list)
            for a in obj["annotations"]:
                if a["image_id"] not in indexed or str(a["category_id"]) not in spec["class_map"]:
                    raise ProjectError("COCO annotation has an unknown image/category")
                box = a["bbox"]
                if len(box) != 4 or not all(math.isfinite(v) for v in box):
                    raise ProjectError("Invalid COCO geometry")
                anns[a["image_id"]].append({"class_id": spec["class_map"][str(a["category_id"])],
                    "bbox": box, "coordinates": "pixel_xywh", "annotation_id": a["id"],
                    "original_class_id": a["category_id"], "area": a.get("area"),
                    "iscrowd": a.get("iscrowd", 0)})
            for im in obj["images"]:
                if im["width"] <= 0 or im["height"] <= 0:
                    raise ProjectError("Invalid COCO image dimensions")
                child(images, im["file_name"])
                rows.append({"file": im["file_name"], "official_split": split,
                             "image_id": im["id"], "width": im["width"], "height": im["height"],
                             "boxes": anns[im["id"]]})
    else:
        raise ProjectError(f"Unsupported annotation format: {spec['format']}")
    assign_roles(rows, spec, config["split_salt"])
    for row in rows:
        row["sample_id"] = f'{dataset}/{row["official_split"]}/{row["file"]}'
        counts = Counter(b["class_id"] for b in row["boxes"])
        row["state"] = "both" if counts[0] and counts[1] else "fire_only" if counts[0] else "smoke_only" if counts[1] else "empty"
        row["image_path"] = str(_image_path(images, row["file"]))
    if len({r["sample_id"] for r in rows}) != len(rows):
        raise ProjectError("Duplicate record identity; no image deduplication performed")
    summary = {
        "dataset": dataset, "version": spec["version"], "images": len(rows),
        "official_splits": dict(Counter(r["official_split"] for r in rows)),
        "roles": dict(Counter(r["role"] for r in rows)),
        "states": dict(Counter(r["state"] for r in rows)),
        "boxes": dict(Counter(b["class_id"] for r in rows for b in r["boxes"])),
        "nonpositive_boxes": sum(b["bbox"][2] <= 0 or b["bbox"][3] <= 0 for r in rows for b in r["boxes"]),
        "expected_counts_match": dict(Counter(r["official_split"] for r in rows)) == spec["expected_counts"],
        "source_digest": hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest(),
        "config_sha256": sha256(config_path), "annotation_root": str(labels),
        "image_root": str(images), "image_root_present": images.is_dir(),
        "images_checked": False, "training_ready": False,
        "original_labels_modified": False, "cross_dataset_deduplication": False,
    }
    return rows, summary


def verify_existing_manifest(rows, path):
    expected = {(x["official_split"], x["file"]): x["role"]
                for x in (json.loads(line) for line in Path(path).read_text().splitlines() if line)}
    actual = {(x["official_split"], x["file"]): x["role"] for x in rows}
    if actual != expected:
        raise ProjectError("Generated roles differ from the frozen existing manifest")
    return {"matched": True, "records": len(actual), "manifest_sha256": sha256(path)}


def inspect_images(rows, summary):
    missing = [r["sample_id"] for r in rows if not Path(r["image_path"]).is_file()]
    summary.update(images_checked=True, missing_image_count=len(missing), missing_examples=missing[:10],
                   training_ready=not missing and summary["expected_counts_match"])
    return summary


def prepare_view(project_root, rows, summary, destination):
    inspect_images(rows, summary)
    if not summary["training_ready"]:
        raise ProjectError(f"Images/counts not ready; missing={summary['missing_image_count']}. No training view written.")
    target = output_path(project_root, destination)
    if target.exists():
        raise ProjectError(f"Output already exists; use a new view directory: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    lock = Path(project_root)/"artifacts/.locks"/(summary["dataset"]+"-prepare.lock")
    with file_lock(lock):
        if target.exists():
            raise ProjectError(f"Output appeared during preparation: {target}")
        staging = Path(tempfile.mkdtemp(prefix=".prepare-", dir=target.parent))
        try:
            by_role = defaultdict(list)
            with (staging/"manifest.jsonl").open("w") as manifest:
                for row in rows:
                    from PIL import Image, ImageOps
                    with Image.open(row["image_path"]) as image:
                        image.load()
                        orientation=image.getexif().get(274,1)
                        displayed=ImageOps.exif_transpose(image)
                        row['decoded_width'],row['decoded_height']=displayed.size
                        row['exif_orientation']=orientation
                        mismatch="width" in row and displayed.size != (row["width"], row["height"])
                        row['dimension_mismatch_without_boxes']=bool(mismatch and not row['boxes'])
                        if mismatch and row['boxes']:
                            raise ProjectError(f"COCO/image dimensions differ for labelled image {row['sample_id']}")
                    key = hashlib.sha256(row["sample_id"].encode()).hexdigest()[:24]
                    ext = Path(row["image_path"]).suffix.lower()
                    image_rel = Path("images")/row["role"]/(key+ext)
                    label_rel = Path("labels")/row["role"]/(key+".txt")
                    (staging/image_rel).parent.mkdir(parents=True, exist_ok=True)
                    (staging/label_rel).parent.mkdir(parents=True, exist_ok=True)
                    (staging/image_rel).symlink_to(Path(row["image_path"]))
                    lines = []
                    for b in row["boxes"]:
                        x,y,w,h = b["bbox"]
                        if b["coordinates"] == "pixel_xywh":
                            x,y,w,h = (x+w/2)/row["width"],(y+h/2)/row["height"],w/row["width"],h/row["height"]
                        lines.append(str(b["class_id"])+" "+" ".join(format(v,'.17g') for v in (x,y,w,h)))
                    (staging/label_rel).write_text("\n".join(lines)+("\n" if lines else ""))
                    by_role[row["role"]].append(str(target/image_rel))
                    manifest.write(json.dumps({**row,"view_image":str(image_rel),"view_label":str(label_rel),
                        "view_label_sha256":sha256(staging/label_rel)},ensure_ascii=False)+"\n")
            for role, paths in by_role.items():
                (staging/f"{role}.txt").write_text("\n".join(paths)+"\n")
            if "fit" in by_role and "dev" in by_role:
                # Deliberately exclude cal/test from the training configuration.
                (staging/"train.yaml").write_text(f'path: {json.dumps(str(target))}\ntrain: fit.txt\nval: dev.txt\nnames:\n  0: fire\n  1: smoke\n')
            summary = {**summary, "view_root":str(target), "view_manifest_sha256":sha256(staging/"manifest.jsonl"),
                       "all_images_decoded":True,
                       "role_list_sha256":{role:sha256(staging/f"{role}.txt") for role in by_role},
                       "training_yaml_sha256":sha256(staging/"train.yaml") if (staging/"train.yaml").exists() else None,
                       "export_policy":"Derived YOLO view; no clipping, no dropping; original metadata retained in manifest"}
            write_json(staging/"view.json",summary)
            os.rename(staging,target)
        except BaseException:
            shutil.rmtree(staging)
            raise
    return summary
