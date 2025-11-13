import json, torch, numpy as np, albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from io import BytesIO
from ..models.registry import build_model

def make_eval_transforms(size):
    return A.Compose([A.Resize(height=size, width=size), A.Normalize(), ToTensorV2()])

class Predictor:
    def __init__(self, ckpt_path, classes_path, backbone, head, img_size=224, device=None):
        self.classes = json.loads(open(classes_path).read())
        self.model = build_model(num_classes=len(self.classes), backbone=backbone, head=head, pooling="gem", feat_ch=256, use_se=True, dilations=(2,3), num_heads=4, attn_pool_stride=2)
        state = torch.load(ckpt_path, map_location="cpu")
        self.model.load_state_dict(state)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()
        self.tfms = make_eval_transforms(img_size)
    def predict_bytes(self, b):
        img = Image.open(BytesIO(b)).convert("RGB")
        arr = np.array(img)
        import numpy as np
        x = self.tfms(image=arr)["image"].unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.model(x)
            prob = torch.softmax(out, 1)[0].cpu().numpy()
        idx = int(prob.argmax())
        score = float(prob[idx])
        topk = np.argsort(-prob)[:5]
        items = [{"label": self.classes[i], "score": float(prob[i])} for i in topk]
        return {"label": self.classes[idx], "score": score, "topk": items}
