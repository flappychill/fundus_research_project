import json, torch, argparse
from src.models.registry import build_model

p = argparse.ArgumentParser()
p.add_argument("--ckpt", required=True)
p.add_argument("--classes", required=True)
p.add_argument("--backbone", default="convnext_base")
p.add_argument("--head", default="rpattn")
p.add_argument("--out", default="models/checkpoints/model.ts")
args = p.parse_args()

classes = json.loads(open(args.classes).read())
m = build_model(num_classes=len(classes), backbone=args.backbone, head=args.head, pooling="gem", feat_ch=256, use_se=True, dilations=(2,3), num_heads=4, attn_pool_stride=2)
state = torch.load(args.ckpt, map_location="cpu")
m.load_state_dict(state)
m.eval()
ex = torch.randn(1,3,224,224)
ts = torch.jit.trace(m, ex)
ts.save(args.out)
print(args.out)
