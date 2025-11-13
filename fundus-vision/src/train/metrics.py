import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, balanced_accuracy_score, cohen_kappa_score, classification_report, confusion_matrix

def one_hot(y, num_classes):
    y = np.asarray(y, dtype=int)
    oh = np.zeros((y.shape[0], num_classes), dtype=np.float32)
    oh[np.arange(y.shape[0]), y] = 1.0
    return oh

def compute_metrics(y_true, logits, class_names):
    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
    preds = probs.argmax(1)
    y_true = np.asarray(y_true)
    acc = accuracy_score(y_true, preds)
    f1m = f1_score(y_true, preds, average="macro", zero_division=0)
    bal = balanced_accuracy_score(y_true, preds)
    kappa = cohen_kappa_score(y_true, preds)
    y_oh = one_hot(y_true, probs.shape[1])
    try:
        auc_macro = roc_auc_score(y_oh, probs, multi_class="ovr", average="macro")
    except Exception:
        aucs = []
        for c in range(probs.shape[1]):
            bt = (y_true == c).astype(int)
            if len(np.unique(bt)) == 2:
                try:
                    aucs.append(roc_auc_score(bt, probs[:, c]))
                except Exception:
                    pass
        auc_macro = float(np.mean(aucs)) if len(aucs) else float("nan")
    rep = classification_report(y_true, preds, target_names=class_names, digits=4)
    cm = confusion_matrix(y_true, preds)
    mse = float(np.mean((probs - y_oh) ** 2))
    return {"acc":acc,"f1_macro":f1m,"auc_macro":auc_macro,"bal_acc":bal,"kappa":kappa,"mse":mse,"report":rep,"cm":cm}

@torch.no_grad()
def run_eval(model, loader, class_names, device):
    model.eval()
    logits_all, y_true = [], []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True); yb = yb.to(device, non_blocking=True)
        out = model(xb)
        logits_all.append(out.detach().cpu()); y_true.append(yb.detach().cpu())
    logits_all = torch.cat(logits_all, 0).numpy()
    y_true = torch.cat(y_true, 0).numpy()
    return compute_metrics(y_true, logits_all, class_names)
