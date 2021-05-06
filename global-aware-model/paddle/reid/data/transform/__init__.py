from . import operators

from .operators import *
import pdb

def build_transform(cfg):
    train_transform = [DecodeImage(),
                       RandomResizeTwiceImage(0.3),
                       ResizeRandomCrop(big_height=cfg.big_height, big_width=cfg.big_width, 
                                target_height=cfg.target_height, target_width=cfg.target_width),
                       RandomHorizontalFlip(),
                       ImageNetPolicy(),
                       NormalizeImage(),
                       RandomErasing(cfg.re_prob),
                       Permute()]
    test_transform = [DecodeImage(),
                      ResizeImage(height=cfg.target_height, width=cfg.target_width), 
                      NormalizeImage(),
                      Permute()]
    return train_transform, test_transform


def build_transform_flip(cfg):
    train_transform = [DecodeImage(),
                       ResizeRandomCrop(big_height=cfg.big_height, big_width=cfg.big_width, 
                                target_height=cfg.target_height, target_width=cfg.target_width),
                       RandomHorizontalFlip(),
                       NormalizeImage(),
                       RandomErasing(cfg.re_prob),
                       Permute()]
    test_transform = [DecodeImage(),
                      ResizeImage(height=cfg.target_height, width=cfg.target_width), 
                      HorizontalFlip(),
                      NormalizeImage(),
                      Permute()]
    return train_transform, test_transform

def build_transform_mixup(cfg):
    train_transform = [DecodeImage(with_mixup=True),
                       MixupImage(), 
                       ResizeRandomCrop(big_height=cfg.big_height, big_width=cfg.big_width, 
                                target_height=cfg.target_height, target_width=cfg.target_width),
                       RandomHorizontalFlip(),
                       NormalizeImage(),
                       RandomErasing(cfg.re_prob),
                       Permute()]
    test_transform = [DecodeImage(),
                      ResizeImage(height=cfg.target_height, width=cfg.target_width), 
                      NormalizeImage(),
                      Permute()]
    return train_transform, test_transform
