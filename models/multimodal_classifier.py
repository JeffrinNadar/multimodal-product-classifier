import torch
import torch.nn as nn
from .text_encoder import TextEncoder
from .image_encoder import ImageEncoder

class MultimodalClassifier(nn.Module):
    """
    Multimodal classifier combining text and image encoders
    Concatenates embeddings and passes through MLP
    """
    def __init__(self, num_classes, 
                 text_model='distilbert-base-uncased',
                 image_model='resnet50',
                 dropout=0.3,
                 freeze_text=False,
                 freeze_image=False,
                 fusion_hidden_dim=1024):
        """
        Args:
            num_classes: Number of output classes
            text_model: Text encoder model name
            image_model: Image encoder model name
            dropout: Dropout probability
            freeze_text: Whether to freeze text encoder
            freeze_image: Whether to freeze image encoder
            fusion_hidden_dim: Hidden dimension for fusion layer
        """
        super(MultimodalClassifier, self).__init__()
        
        # Encoders
        self.text_encoder = TextEncoder(
            model_name=text_model,
            freeze_base=freeze_text
        )
        self.image_encoder = ImageEncoder(
            model_name=image_model,
            freeze_base=freeze_image
        )
        
        # Get embedding dimensions
        text_dim = self.text_encoder.get_embedding_dim()
        image_dim = self.image_encoder.get_embedding_dim()
        combined_dim = text_dim + image_dim
        
        print(f"Text embedding dim: {text_dim}")
        print(f"Image embedding dim: {image_dim}")
        print(f"Combined dim: {combined_dim}")
        
        # Fusion network (MLP)
        self.fusion = nn.Sequential(
            nn.Linear(combined_dim, fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, fusion_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim // 2, num_classes)
        )
    
    def forward(self, input_ids, attention_mask, images):
        """
        Forward pass
        
        Args:
            input_ids: Text token IDs [batch_size, seq_len]
            attention_mask: Text attention mask [batch_size, seq_len]
            images: Image tensor [batch_size, 3, H, W]
            
        Returns:
            Logits [batch_size, num_classes]
        """
        # Get embeddings from both modalities
        text_embedding = self.text_encoder(input_ids, attention_mask)
        image_embedding = self.image_encoder(images)
        
        # Concatenate embeddings
        combined = torch.cat([text_embedding, image_embedding], dim=1)
        
        # Pass through fusion network
        logits = self.fusion(combined)
        
        return logits
    
    def get_embeddings(self, input_ids, attention_mask, images):
        """
        Get multimodal embeddings before classification layer
        Useful for visualization or active learning
        
        Returns:
            Combined embeddings [batch_size, text_dim + image_dim]
        """
        with torch.no_grad():
            text_embedding = self.text_encoder(input_ids, attention_mask)
            image_embedding = self.image_encoder(images)
            combined = torch.cat([text_embedding, image_embedding], dim=1)
        
        return combined


class EarlyFusionClassifier(nn.Module):
    """
    Alternative fusion strategy: process features earlier
    """
    def __init__(self, num_classes, 
                 text_model='distilbert-base-uncased',
                 image_model='resnet50',
                 dropout=0.3):
        """
        Early fusion: project both modalities to same dimension before fusion
        """
        super(EarlyFusionClassifier, self).__init__()
        
        self.text_encoder = TextEncoder(model_name=text_model)
        self.image_encoder = ImageEncoder(model_name=image_model)
        
        text_dim = self.text_encoder.get_embedding_dim()
        image_dim = self.image_encoder.get_embedding_dim()
        
        # Project to common dimension
        common_dim = 512
        self.text_projection = nn.Linear(text_dim, common_dim)
        self.image_projection = nn.Linear(image_dim, common_dim)
        
        # Fusion with element-wise operations
        self.fusion = nn.Sequential(
            nn.Linear(common_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, input_ids, attention_mask, images):
        text_emb = self.text_encoder(input_ids, attention_mask)
        image_emb = self.image_encoder(images)
        
        # Project to common space
        text_proj = self.text_projection(text_emb)
        image_proj = self.image_projection(image_emb)
        
        # Element-wise addition (could also use multiplication or concat)
        fused = text_proj + image_proj
        
        logits = self.fusion(fused)
        return logits


# Example usage and testing
if __name__ == "__main__":
    # Test multimodal classifier
    batch_size = 4
    seq_len = 128
    num_classes = 100
    
    # Create dummy inputs
    input_ids = torch.randint(0, 30522, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    images = torch.randn(batch_size, 3, 224, 224)
    
    # Test late fusion (concatenation)
    print("Testing MultimodalClassifier (Late Fusion)...")
    model = MultimodalClassifier(num_classes=num_classes)
    logits = model(input_ids, attention_mask, images)
    print(f"Output shape: {logits.shape}")
    print(f"Expected: [{batch_size}, {num_classes}]")
    
    # Get embeddings
    embeddings = model.get_embeddings(input_ids, attention_mask, images)
    print(f"Embedding shape: {embeddings.shape}")
    
    # Test early fusion
    print("\nTesting EarlyFusionClassifier...")
    model2 = EarlyFusionClassifier(num_classes=num_classes)
    logits2 = model2(input_ids, attention_mask, images)
    print(f"Output shape: {logits2.shape}")
    
    # Check trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")