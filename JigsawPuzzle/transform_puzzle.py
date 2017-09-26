from __future__ import division
import torch
import math
import random
from PIL import Image, ImageOps
try:
    import accimage
except ImportError:
    accimage = None
import numpy as np
import numbers
import types
import collections

class IntoPatch(object):
    """Creates PIL.Image patches from given PIL.Image.
    Args:
        size (sequence or int): Desired output size of the patch. If size is an
            int instead of sequence like (w, h), a square crop (size, size) is
            made.
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
            
        self.patches = []

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be patched.
        Returns:
            PIL.Image: PIL.Image with patches concatenated in one.
        """

        w, h = img.size
        tw, th = self.size
        newimg = Image.new('RGB', (tw*9, th))
        self.patches = []
        
        for w1 in range(0, w, int(w/3)): 
            for h1 in range(0, h, int(h/3)): 
                w2 = w1 + int(np.random.randint(1, int(w/3) - tw)) 
                h2 = h1 + int(np.random.randint(1, int(h/3) - th)) 
                self.patches.append(img.crop((w2, h2, w2 + tw, h2 + tw)))
                
        x_offset = 0
        for im in self.patches:
            newimg.paste(im, (x_offset,0))
            x_offset += im.size[0]
            
        return newimg