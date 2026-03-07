# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 09:29:37 2023

@author: Laktop
"""
import sys
import os

import numpy as np
import matplotlib.pyplot as pl
from matplotlib.gridspec import GridSpec
import pandas as pd
import glob
from cmcrameri import cm as cmCrameri
from scipy import signal
import time


from MyPaths import*  # File with path of script locations. Specific for each computer. As alternative specify path to be add to sys.

scriptspath=LEMscripts_path
pcpath=GDrive_path


sys.path.append(scriptspath)


from pyLEM.dataADC import *
from pyLEM.INSLASERdata import *


# sys.path.append(scriptspath+r'\pyLEM')
# from dataADC import*
# from INSLASERdata import *




# %%   Settings

SPS=19200

chunksize=10000
window=960
flowpass=20

distCenter=0.244

# %% read data known file

path=r'C:/dataADC'

filename='000101_144839'


fileLASER=path+r'/INS'+filename+'.csv'
fileADC=path+r'/ADC'+filename+'.csv'

print("File to process is:", fileADC)

# %% Load last file

path=r'C:/dataADC'


# Get a list of all files and directories in the folder
file_list = os.listdir(path)

# get ADC data only
file_list =[k for k in file_list if ('ADC' in k and '.csv' in k)]

# Sort the files by their modification time
sorted_files = sorted(file_list, key=lambda x: os.path.getmtime(os.path.join(path, x)))

# Get the last file in the list, which is the most recently modified file
last_modified_file = sorted_files[-1]

print("The last modified file in the folder is:", last_modified_file)

filename=last_modified_file.lstrip('ADC').rstrip('.csv')

fileLASER=path+r'/INS'+filename+'.csv'
fileADC=path+r'/ADC'+filename+'.csv'





# %% Load data INS

dataINS=INSLASERdata(fileLASER,name='Check IMX and Laser',correct_Laser=True,distCenter=distCenter,roll0=0.0,pitch0=0.0)

# plot_summary2(dataINS)

plot_summary(dataINS,getextent(dataINS),heading=False)

plot_GPSquality(dataINS,title='GPS quality')


# %% Load, merge and save data ADC +IN+ Laser
T_max=0
autoCal=True

datamean,dataINS,params=processDataLEM(path,filename,plot=True,plotINS=False,savefile=True,
                          window=1920,
                          Tx_ch='ch2', Rx_ch=['ch1'],
                          findFreq=True,i_blok=[int(150000),int(250000)],
                          i_cal=[], T_max=T_max,
                          autoCal=autoCal)



# %%  Check calibration

CheckCalibration(dataINS,datamean,params['f'],plot=True)



# %% plot section of raw data

t0=5
t1=8

lookupSectionRaw(fileADC,t0,t1,SPS=19200,units='seconds',title='File:'+filename, channels=[1,2])


# %% Load processed data
[datamean,dataINS,params]=sl.load_pkl(path+r'\LEM'+filename+'.pkl')

