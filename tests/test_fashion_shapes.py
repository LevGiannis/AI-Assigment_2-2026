import torch
from src.fashion.model import FashionCNN

def test_fashion_forward_shape():
    model = FashionCNN()
    x = torch.randn(4, 1, 28, 28)
    logits = model(x)
    assert logits.shape == (4, 10)
