# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 09:29:37 2023

@author: Laktop
"""
import sys

import numpy as np
import matplotlib.pyplot as pl
from matplotlib.gridspec import GridSpec
import pandas as pd
import glob
from cmcrameri import cm as cmCrameri
import time

# Add direct path here
scriptspath=r'C:\Users\H6\LEMcode'

# use file MyPaths with machine paths
from MyPaths import*  # File with path of script locations. Specific for each computer. As alternative specify path to be add to sys.
scriptspath=LEMscripts_path     # location of LEM scripts

sys.path.append(scriptspath)
sys.path.append(scriptspath+r'\pyLEM')
from dataADC import*


# %%  file info and processing settings

filename='260612_221215'
path= r'D:\FIeldData\Utq2026\260612'

dT_start=-10.1
T_max=0



# %% Load, merge and save data ADC +INS+ Laser

joinDataLEM_onboardSimulator(path,filename,plot=True,dT_start=dT_start,T_max=T_max)




