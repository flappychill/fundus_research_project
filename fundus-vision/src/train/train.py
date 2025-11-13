import os, yaml, json, numpy as np, torch, pandas as pd
from pathlib import Path
from collections import Counter
from .utils import set_seed, ensure_dir
from .sampler import class_balanced_alpha, make_class_aware_sampler
from ..data.transforms import make_train_transforms, make_eval_transforms
from ..data.dataset import build_datasets, make_loader
from ..models.registry import build_model
from .metrics import run_eval

def main(cfg_path):
    cfg = yaml.safe_load(open(cfg_path))
    set_seed(123)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ensure_dir(cfg["save_dir"])
    tr_tfms = make_train_transforms(cfg["img_size"])
    ev_tfms = make_eval_transforms(cfg["img_size"])
    train_ds, valid_ds, test_ds = build_datasets(cfg["root"], tr_tfms, ev_tfms)
    classes = train_ds.classes
    counts = [Counter([y for _,y in train_ds.samples])[i] for i in range(len(classes))]
    alpha = class_balanced_alpha(counts)
    sampler = make_class_aware_sampler(train_ds)
    bs = cfg["batch_size"]; nw = cfg["num_workers"]
    train_dl = make_loader(train_ds, bs, nw, sampler=sampler)
    valid_dl = make_loader(valid_ds, bs, nw, shuffle=False)
    test_dl  = make_loader(test_ds, bs, nw, shuffle=False)
    rows = []
    for seed in cfg["seed_list"]:
        set_seed(seed)
        model = build_model(num_classes=len(classes), backbone=cfg["backbone"], head=cfg["head"], pooling=cfg["pooling"], feat_ch=cfg["feat_ch"], use_se=cfg["use_se"], dilations=tuple(cfg["dilations"]), num_heads=cfg["num_heads"], attn_pool_stride=cfg["attn_pool_stride"])
        model = model.to(device)
        from .engine import train_loop
        best = train_loop(model, train_dl, valid_dl, device, cfg["epochs"], cfg["lr"], cfg["weight_decay"], alpha=alpha)
        model.load_state_dict(best["state"])
        val_m = run_eval(model, valid_dl, classes, device)
        test_m = run_eval(model, test_dl, classes, device)
        ckpt_dir = Path(cfg["save_dir"]) / f"{cfg['head']}_{seed}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), ckpt_dir/"best.pt")
        (ckpt_dir/"classes.json").write_text(json.dumps(classes))
        (ckpt_dir/"VAL_report.txt").write_text(val_m["report"])
        (ckpt_dir/"TEST_report.txt").write_text(test_m["report"])
        np.save(ckpt_dir/"VAL_confusion.npy", val_m["cm"])
        np.save(ckpt_dir/"TEST_confusion.npy", test_m["cm"])
        rows.append({
            "seed": seed,
            "val_acc": val_m["acc"],
            "val_f1": val_m["f1_macro"],
            "val_auc": val_m["auc_macro"],
            "val_balacc": val_m["bal_acc"],
            "val_kappa": val_m["kappa"],
            "test_acc": test_m["acc"],
            "test_f1": test_m["f1_macro"],
            "test_auc": test_m["auc_macro"],
            "test_balacc": test_m["bal_acc"],
            "test_kappa": test_m["kappa"],
            "ckpt": str(ckpt_dir/"best.pt")
        })
    df = pd.DataFrame(rows)
    df.to_csv(Path(cfg["save_dir"])/"results_per_run.csv", index=False)
    g = df.agg({"val_acc":["mean","std"],"val_f1":["mean","std"],"val_auc":["mean","std"],"test_acc":["mean","std"],"test_f1":["mean","std"],"test_auc":["mean","std"]})
    g.to_csv(Path(cfg["save_dir"])/"summary.csv")
    return 0

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", default="configs/default.yaml")
    a = p.parse_args()
    main(a.cfg)
