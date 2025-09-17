# Sketch Primitive Detector Model

### Overview

This repository provides a dataset and tooling for detecting **annotation primitives** in photographed hand-drawn sketches.
The goal is to transform raw sketch images into structured representations containing **digits, arrows, dimensions, radii, and geometric primitives**.

Inspired by [*From 2D CAD Drawings to 3D Parametric Models: A Vision-Language Approach*](https://arxiv.org/html/2412.11892v1), we extend the idea of **raster-to-program modeling** to the domain of noisy, real-world sketches enriched with engineering annotations.

---

### Motivation

Traditional CAD digitization pipelines assume clean vector data and separated geometry/annotation layers.
In practice, engineers and designers often work with **photos of freehand sketches**, where geometry and annotations are intertwined.

This dataset provides:

- Photographic sketches annotated with **support elements** (numbers, arrows, dimension marks).
- JSON-based structured labels describing both **what** was drawn and **where** it appears.
- A foundation for training models that **interpret sketches as programs of primitives**.

---

### Dataset Structure

```
dataset/
  raw_images/            # Photographed sketches
  annotations/           # JSON labels per image
  splits/                # train/val/test lists
preprocessing/
  preprocess_images.py   # Normalization & augmentation
  convert_annotations.py # Convert raw labels into schema
models/
  vision_encoder/        # ViT/DETR backbone configs
  primitive_detector/    # Training scripts
evaluation/
  metrics.py             # IoU / precision / recall
  evaluate.py            # Benchmark pipeline
```

---

### Annotation Schema

Each sketch image is paired with a JSON file describing detected primitives:

```json
{
  "image": "sketch_0123.jpg",
  "primitives": [
    {"type":"digit","bbox":[120,80,30,40],"value":"5"},
    {"type":"dimension","bbox":[200,150,100,20],"value":"50mm"},
    {"type":"arrow","bbox":[220,130,10,30]},
    {"type":"radius","bbox":[310,200,20,20],"value":"10"}
  ]
}
```

- **type**: `"digit"`, `"arrow"`, `"dimension"`, `"radius"`, etc.
- **bbox**: `[x, y, width, height]` in pixels.
- **value**: Optional, e.g. recognized number or dimension string.
- **confidence**: Optional, model prediction score.

---

### Evaluation

We provide metrics to benchmark detection and structured prediction:

- **Per-primitive detection**: IoU-based precision, recall, F1.
- **Script-level accuracy**: End-to-end correctness of primitive sets (Hungarian matching, similar to CAD2Program evaluation).

---

### Example Workflow

```bash
# Preprocess images and annotations
python preprocessing/preprocess_images.py --input dataset/raw_images --output dataset/processed

# Train a primitive detection model
python models/primitive_detector/train.py --config configs/detector.yaml

# Evaluate on test set
python evaluation/evaluate.py --split test
```

---

### Use Cases

- **Vision-Language Modeling**: Convert sketches into symbolic representations.
- **CAD Reconstruction**: Feed detected primitives into tools like `sketch-tool` or `sketchvis`.
- **Synthetic Data Augmentation**: Generate large-scale labeled sketches for training.

---

### Roadmap

- [ ] Expand primitive types (e.g., hatch, symbols, text notes).
- [ ] Release pretrained ViT/DETR baselines.
- [ ] Integrate with CAD program generation pipelines.
- [ ] Publish benchmark leaderboard.

---

### Citation

If you use this dataset or code, please cite:

```bibtex
@article{cad2program2024,
  title={From 2D CAD Drawings to 3D Parametric Models: A Vision-Language Approach},
  author={...},
  journal={arXiv preprint arXiv:2412.11892},
  year={2024}
}
```

---

### License

MIT License. See [LICENSE](LICENSE) for details.

---

### Contributions

We welcome PRs for:

- New annotation types
- Improved preprocessing pipelines
- Additional evaluation metrics
