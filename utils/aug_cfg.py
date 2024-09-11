from .data_augmentation import *

# BGR
mean = (85.08, 85.64, 86.12)
std = (52.98, 54.79, 56.09)

def default_aug():
    """ used for train 
    mode = [allocate, order, without_order]
    """
    return ProcessingSequential([
        RandomFlip(p_horizontal=0.5, p_vertical=0.5, mode="allocate"),
        RandomRotate(mode="allocate"),
        Normalize(mean=mean, std=std)
    ])

def test_aug():
    """ used for test"""
    return ProcessingSequential([
        Normalize(mean=mean, std=std)
    ])

 