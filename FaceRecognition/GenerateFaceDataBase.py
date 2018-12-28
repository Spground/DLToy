#!/usr/bin/env python
# coding: utf-8

from inception_block_v2 import *
from fr_utils import *
from numpy as np

FRmodel = faceRecModel(input_shape=(3, 96, 96))
input_image = sys.argv[1]
encoding = img_to_encoding(input_image, FRmodel)
np.save(encoding, "")


