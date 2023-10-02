from base.base_dataloader import BaseDataLoader
from torchvision import datasets, transforms
from skimage import io
import cv2
import numpy as np

class YCBDataLoader(BaseDataLoader):
    def __init__(self):
        raise NotImplementedError