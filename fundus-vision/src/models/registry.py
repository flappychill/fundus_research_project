from .heads import PlainHead, StrongHead, FPNHead, RevPyrHead, RPAttnHead

REGISTRY = {
    "plain": PlainHead,
    "strong": StrongHead,
    "fpn": FPNHead,
    "revpyr": RevPyrHead,
    "rpattn": RPAttnHead
}

def build_model(num_classes, backbone, head, **kwargs):
    cls = REGISTRY[head]
    return cls(num_classes=num_classes, backbone_name=backbone, **kwargs)
