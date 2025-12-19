import torch
import torch.nn as nn
from torchvision import models


class ImageEncoder(nn.Module):
    """
    Image encoder using a pretrained ResNet backbone.

    Produces a fixed-size embedding (default 2048) by removing the
    ResNet classification head.
    """
    def __init__(self, model_name='resnet50', pretrained=True, freeze_base=False, embedding_dim=2048):
        super(ImageEncoder, self).__init__()

        self.model_name = model_name
        self.embedding_dim = embedding_dim

        # Build backbone
        if 'resnet' in model_name:
            try:
                # Preferred API for newer torchvision
                backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
            except Exception:
                # Fallback for older torchvision versions
                backbone = models.resnet50(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported image model: {model_name}")

        # Remove classification head
        backbone.fc = nn.Identity()

        self.backbone = backbone

        if freeze_base:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, images):
        """
        Args:
            images: Tensor [batch_size, 3, H, W]

        Returns:
            embeddings: Tensor [batch_size, embedding_dim]
        """
        features = self.backbone(images)
        if features.dim() == 4:
            features = torch.flatten(features, 1)
        return features

    def get_embedding_dim(self):
        return self.embedding_dim


class ImageClassifier(nn.Module):
    """
    Simple image-only classifier that wraps `ImageEncoder` and adds
    a lightweight classification head.
    """
    def __init__(self, num_classes, model_name='resnet50', pretrained=True, freeze_base=False, dropout=0.3, hidden_dim=512):
        super(ImageClassifier, self).__init__()

        self.encoder = ImageEncoder(model_name=model_name, pretrained=pretrained, freeze_base=freeze_base)
        emb_dim = self.encoder.get_embedding_dim()

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, images):
        emb = self.encoder(images)
        logits = self.classifier(emb)
        return logits

    def get_embedding_dim(self):
        return self.encoder.get_embedding_dim()


if __name__ == '__main__':
    # Quick smoke test (uses non-pretrained backbone to avoid downloads)
    import torch

    device = torch.device('cpu')
    enc = ImageEncoder(pretrained=False).to(device)
    cls = ImageClassifier(num_classes=10, pretrained=False).to(device)

    x = torch.randn(2, 3, 224, 224).to(device)
    emb = enc(x)
    logits = cls(x)
    print('Embedding shape:', emb.shape)
    print('Logits shape:', logits.shape)
