#!/usr/bin/env python

from pathlib import Path
import json
import numpy as np
import sys
import os
import math
import numpy as np
from ipywidgets import widgets,interact,IntProgress
import matplotlib.pyplot as plt
from matplotlib import image
from skimage import morphology
from skimage.morphology import closing, square, reconstruction
from skimage import filters
from sklearn import preprocessing
#from OctCorrection import *
#from ImageProcessing import *
import pickle
from mpl_toolkits.mplot3d import Axes3D
import os
import glob
import re

from PIL import Image
import pandas as pd
#import cv2
from tifffile import imsave, imread
from numba import jit

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import imageio
import shutil 

from struct import unpack

from skimage.metrics import structural_similarity

import time

from io import StringIO # "import StringIO" directly in python2
from PIL import Image
from io import BytesIO

#paths = ['val/*.png', 'train/*.png', 'test/*.png']
paths = ['Train/imgs/*.png', 'Test/imgs/*.png']

#srcpath='/scratch/mfaykus/dissertation/datasets/cityscapes2/leftImg8bit/'
srcpath = "/scratch/aniemcz/CAT2/CAT/mixed/"

flag = 0
i = 0
while i < len(paths):

    l=glob.glob(srcpath+paths[i])
    l.sort()
    
    print(f"paths list is {len(l)} long")

    for name in l:
        print(name)
        im1 = Image.open(name)

        Quality = 0
        while Quality <= 100:
            buffer = BytesIO()
            im1.save(buffer, "JPEG", quality=Quality)
            
            dest = '/scratch/aniemcz/cavsMiniLossyCompressorsV2/jpeg/'
            
            if i ==0:
                dest_2 = '/Train/rgb/'
            elif i == 1:
                dest_2 = '/Test/rgb/'
                      
            path = dest + 'Q' + str(Quality) + dest_2

            # make the result directory
            if not os.path.exists(path):
                os.makedirs(path)
            
            filename = os.path.basename(name).split('/')[-1]
            
            filename = filename.split('.', 1)[0]
            filename = str(filename) + '.jpg'
            location = path + str(filename)
            
            print(location)
            
           # with open(location, "wb") as handle:  # Open in binary mode
            #    handle.write(buffer.getvalue())
                
            #destination_dir = (dest + 'Q' + str(Quality)+'/gtFine')
            #os.makedirs(destination_dir, exist_ok=True)

            #copy_gt = '/scratch/mfaykus/dissertation/datasets/cityscapes2/gtFine'

            #if flag == 5 and Quality > 95:
                #shutil.copytree(copy_gt, destination_dir, dirs_exist_ok=True) 

            Quality += 10
        flag = 1
    i += 1
            
            
            

    





