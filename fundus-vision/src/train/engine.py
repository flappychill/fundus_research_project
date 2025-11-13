import torch
from .metrics import run_eval
from .losses import FocalCE

def train_loop(model, train_dl, valid_dl, device, epochs, lr, weight_decay, alpha=None):
    criterion = FocalCE(alpha=alpha.to(device) if alpha is not None else None, gamma=1.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type=='cuda'))
    best = {"auc": -1.0, "state": None}
    for _ in range(epochs):
        model.train()
        for xb, yb in train_dl:
            xb = xb.to(device, non_blocking=True); yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type=='cuda')):
                logits = model(xb); loss = criterion(logits, yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update()
        scheduler.step()
        val_m = run_eval(model, valid_dl, valid_dl.dataset.classes, device)
        if val_m["auc_macro"] > best["auc"]:
            best["auc"] = val_m["auc_macro"]
            best["state"] = {k: v.cpu() for k,v in model.state_dict().items()}
    return best
