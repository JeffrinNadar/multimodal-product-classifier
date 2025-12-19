import torch
import torch.nn as nn
from transformers import AutoModel

class TextEncoder(nn.Module):
    """
    Text encoder using pretrained transformer (DistilBERT)
    Outputs 768-dimensional embeddings
    """
    def __init__(self, model_name='distilbert-base-uncased', 
                 embedding_dim=768, freeze_base=False):
        """
        Args:
            model_name: HuggingFace model name
            embedding_dim: Output embedding dimension
            freeze_base: Whether to freeze transformer weights
        """
        super(TextEncoder, self).__init__()
        
        self.transformer = AutoModel.from_pretrained(model_name)
        self.embedding_dim = embedding_dim
        
        # Optionally freeze transformer weights
        if freeze_base:
            for param in self.transformer.parameters():
                param.requires_grad = False
        
    def forward(self, input_ids, attention_mask):
        """
        Forward pass
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Text embeddings [batch_size, embedding_dim]
        """
        # Get transformer outputs
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token embedding (first token)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        
        return cls_embedding
    
    def get_embedding_dim(self):
        """Return output embedding dimension"""
        return self.embedding_dim


class TextClassifier(nn.Module):
    """
    Complete text-only classifier with transformer encoder
    """
    def __init__(self, num_classes, model_name='distilbert-base-uncased',
                 dropout=0.3, freeze_base=False):
        """
        Args:
            num_classes: Number of output classes
            model_name: HuggingFace model name
            dropout: Dropout probability
            freeze_base: Whether to freeze transformer
        """
        super(TextClassifier, self).__init__()
        
        self.encoder = TextEncoder(model_name, freeze_base=freeze_base)
        embedding_dim = self.encoder.get_embedding_dim()
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, input_ids, attention_mask):
        """
        Forward pass
        
        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
            
        Returns:
            Logits [batch_size, num_classes]
        """
        # Get text embeddings
        text_embedding = self.encoder(input_ids, attention_mask)
        
        # Classify
        logits = self.classifier(text_embedding)
        
        return logits


# Example usage and testing
if __name__ == "__main__":
    # Test text encoder
    batch_size = 4
    seq_len = 128
    num_classes = 100
    
    # Create dummy input
    input_ids = torch.randint(0, 30522, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    
    # Test encoder
    print("Testing TextEncoder...")
    encoder = TextEncoder()
    embeddings = encoder(input_ids, attention_mask)
    print(f"Input shape: {input_ids.shape}")
    print(f"Output shape: {embeddings.shape}")
    print(f"Expected: [{batch_size}, 768]")
    
    # Test classifier
    print("\nTesting TextClassifier...")
    classifier = TextClassifier(num_classes=num_classes)
    logits = classifier(input_ids, attention_mask)
    print(f"Logits shape: {logits.shape}")
    print(f"Expected: [{batch_size}, {num_classes}]")
    
    # Check trainable parameters
    total_params = sum(p.numel() for p in classifier.parameters())
    trainable_params = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")