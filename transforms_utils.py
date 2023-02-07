import torch
import random
import torchvision.transforms.functional as TF


class Rotate():
    def __init__(self, degrees):
        self.degrees = degrees
        
    def __call__(self, imgs):
        angle = random.choice(range(*self.degrees))
        return [TF.rotate(img, angle) for img in imgs]


class RandomChoice(torch.nn.Module):
    def __init__(self, transforms):
       super().__init__()
       self.transforms = transforms

    def __call__(self, imgs):
        t = random.choice(self.transforms)
        if type(t) == Rotate:
            return t.__call__(imgs)
        else:
            return [t(img) for img in imgs]
