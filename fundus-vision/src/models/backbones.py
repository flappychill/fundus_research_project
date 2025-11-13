import timm

def build_backbone(name):
    return timm.create_model(name, pretrained=True, features_only=True)

def feature_channels(fi, take=3):
    idxs = list(range(len(fi)))[-take:]
    chs = [fi[i]['num_chs'] for i in idxs]
    return idxs, chs
