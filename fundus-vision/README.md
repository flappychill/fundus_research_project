# Fundus Vision

A complete retinal fundus classification project for medical image recognition. It includes training with multiple heads on top of modern backbones, exploratory data analysis, an inference API, and a simple web app.

## Dataset

Folder layout
```
eye/
  train/
    Cataract/
    Central Serous Chor/
    Diabetic Retinopathy/
    Disc Edema/
    Glaucoma/
    Healthy/
    Macular Scar/
    Myopia/
    Pterygium/
    Retinal Detachment/
    Retinitis Pigmentosa/
  valid/...
  test/...
```
Place the `eye` directory under `data/` or set `root` in the config.

## Install

```
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

## EDA

```
python -m src.data.eda --root ./data/eye --out ./outputs/eda
```

Artifacts:
- `files.csv` lists every image with size and split
- `counts.csv` class counts per split
- `sizes.csv` image size stats
- `summary.json` quick totals

## Train

Config file: `configs/default.yaml`

Key fields:
- `backbone`: any timm features-only model, e.g. `convnext_base`, `resnet50`, `efficientnet_b3`
- `head`: `plain`, `strong`, `fpn`, `revpyr`, `rpattn`
- CB Focal and class-aware sampling are used automatically

Run
```
python -m src.train.train --cfg configs/default.yaml
```

Outputs per seed:
- `best.pt` weights
- `classes.json`
- `VAL_report.txt`, `TEST_report.txt`
- confusion matrices in `.npy`
- `results_per_run.csv` and `summary.csv` in the save dir

## Inference API

Set environment variables or keep defaults and place weights and classes here:
```
models/checkpoints/best.pt
models/checkpoints/classes.json
```

Start the server
```
export MODEL_PATH=models/checkpoints/best.pt
export CLASSES_PATH=models/checkpoints/classes.json
export BACKBONE=convnext_base
export HEAD=rpattn
export IMG_SIZE=224
gunicorn -w 2 -b 0.0.0.0:8000 src.api.app:app
```

Open `http://localhost:8000` and upload an image.

## Docker

```
docker build -t fundus-vision .
docker run -p 8000:8000 -e MODEL_PATH=/app/models/checkpoints/best.pt -e CLASSES_PATH=/app/models/checkpoints/classes.json fundus-vision
```

## Export TorchScript

```
python scripts/export_torchscript.py --ckpt models/checkpoints/best.pt --classes models/checkpoints/classes.json --backbone convnext_base --head rpattn --out models/checkpoints/model.ts
```

## Tips

- Train with different heads by changing `head` in config.
- Change `backbone` to compare timm models.
- Use `outputs/eda` to understand class balance and image sizes.
- The web app returns top-5 predictions with scores.

## License

MIT
