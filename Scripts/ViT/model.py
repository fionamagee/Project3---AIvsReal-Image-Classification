import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights

def get_vit_model(num_classes=2, freeze_backbone=False):
    model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.heads.head.in_features
    model.heads.head = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_classes)
    )
    return model