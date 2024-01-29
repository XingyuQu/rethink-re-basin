"""AlexNet in PyTorch"""
# import torch
from torchvision import models

__all__ = ["alexnet"]


def alexnet():
    model = models.alexnet()
    return model
