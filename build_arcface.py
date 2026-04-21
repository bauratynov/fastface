"""
Build ResNet-50 + ArcFace-style head from pure torch.nn (no torchvision).
Export to ONNX for benchmarking against the FastFace Rust reimplementation.

ArcFace R50 reference: ResNet-50 backbone → avgpool → flatten → Linear(2048→512) → L2-norm
Input: [1, 3, 112, 112] (standard ArcFace size). Output: [1, 512] embedding.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 4

    def __init__(self, in_ch, out_ch, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.conv3 = nn.Conv2d(out_ch, out_ch * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_ch * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class ResNet50ArcFace(nn.Module):
    """ResNet-50 → avgpool → flatten → Linear(2048→512). L2-norm done outside model for ONNX."""

    def __init__(self, embedding_dim=512):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        # ResNet-50 layout: [3, 4, 6, 3] blocks per stage
        self.layer1 = self._make_layer(64, 64, 3, stride=1)
        self.layer2 = self._make_layer(256, 128, 4, stride=2)
        self.layer3 = self._make_layer(512, 256, 6, stride=2)
        self.layer4 = self._make_layer(1024, 512, 3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, embedding_dim)

    def _make_layer(self, in_ch, out_ch, blocks, stride):
        downsample = None
        if stride != 1 or in_ch != out_ch * BasicBlock.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch * BasicBlock.expansion, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch * BasicBlock.expansion),
            )
        layers = [BasicBlock(in_ch, out_ch, stride, downsample)]
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_ch * BasicBlock.expansion, out_ch))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


if __name__ == "__main__":
    import sys
    model = ResNet50ArcFace(embedding_dim=512).eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"ResNet-50 ArcFace: {n_params:.1f}M params")

    dummy = torch.randn(1, 3, 112, 112)
    with torch.no_grad():
        out = model(dummy)
    print(f"Output shape: {out.shape}")

    # Export to ONNX
    torch.onnx.export(
        model, dummy, "arcface_r50.onnx",
        input_names=["input"], output_names=["embedding"],
        dynamic_axes={"input": {0: "batch"}, "embedding": {0: "batch"}},
        opset_version=14,
    )
    import os
    sz_mb = os.path.getsize("arcface_r50.onnx") / 1024 / 1024
    print(f"Exported arcface_r50.onnx ({sz_mb:.1f} MB)")
