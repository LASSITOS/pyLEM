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

try: 
    from hampel import hampel
except:
    print("coudn't install hampel")

# from MyPaths import*  # File with path of script locations. Specific for each computer. As alternative specify path to be add to sys.
# # if you set up scripth path masterfile
# scriptspath=LEMscripts_path
# pcpath=GDrive_path

#if you use a direct path
scriptspath=r'C:\Users\H6\LEMcode'

sys.path.append(scriptspath)
from pyLEM.dataADC import *
from pyLEM.INSLASERdata import *




# %%  Settings

filename='260206_212311'


path=r'C:\dataADC'
fileLASER=path+r'\INS'+filename+'.csv'
fileADC=path+r'\ADC'+filename+'.csv'
savpath=path

SPS=19200
window=1920
flowpass=15
INSkargs={}


i_autoCal=0
dT_start=0.4

i_blok=[int(50*SPS),int(55*SPS)]

# data margins
start=200
stop=530

# climbs
t_str=[335]
t_stp=[373,]
h_tot=[1.3055]
w_depth=[2000,]


# freeboard
t_Freeboard=[535,370]
cal_freeboard=14.5/100	



# emagpy params
w_cond=2408
d_coils=2.027

T_max=0


# %% plot section of raw data

t0=230 
t1=238

lookupSectionRaw(fileADC,t0,t1,SPS=19200,units='seconds',title='File:'+filename, channels=[1,2])

# %% plot INS data to set 
dataINS=INSLASERdata(fileLASER,correct_Laser=True,
                     roll0=0.0,pitch0=0.0)

plot_summary(dataINS,getextent(dataINS),heading=False)

# %% Load, merge and save data ADC +INS+ Laser

datamean,dataINS,params=processDataLEM(path,filename,plot=True,savefile=True,
                          window=window,flowpass=flowpass,chunksize=38401,
                          Tx_ch='ch2', Rx_ch=['ch1'],
                          findFreq=True,i_blok=i_blok,
                          MultiFreq=False,
                          INSkargs=INSkargs,
                          autoCal=True,i_autoCal=i_autoCal,dT_start=dT_start,
                          T_max=T_max)

# datamean['time']=datamean.TOW-datamean.TOW[0]


# %% Load processed data
[datamean,dataINS,params]=sl.load_pkl(path+r'\LEM'+filename+'.pkl')




# %% cut data 
datamean=trim_data(start,stop,datamean,params)

# %% Hampel filter
# window_size=10
# deviation=5.0

# filter_outliers(datamean,params,window_size=window_size,deviation=deviation,plot=True)
# %%% plots of I,Q and h

plot_QIandH(datamean,params,title='')




# %% Load,H6  GPS .pos data

fileGPSUAV="LEM20370.pos"

dataGPS=addGPSdata(path,fileGPSUAV,datamean, dataINS,plot=True)
getFreeboard(datamean,t_Freeboard,cal_freeboard,UAV=True,plot=True)



# %% -------------------------------------
#............ Empirical Inversion ...........
#----------------------------------------


Fit_climbs_emp(datamean,params,
                   t_str,t_stp, h_tot,  plot=True,h_lim=15)



# %% plot inverted data



fig,ax=pl.subplots(1,1,sharex=True)
ax.plot(datamean.h_Laser,datamean.h_water_empQ,'x',label='from Q')
ax.plot(datamean.h_Laser,datamean.h_water_empI,'x',label='from I')
ax.plot([0,30],[0,30],'--k')
ax.set_ylabel('h EM (m)')
ax.set_xlabel('h Laser/GPS (m)')
ax.legend()
ax.set_xlim(0,25)
ax.set_ylim(0,25)



fig,ax=pl.subplots(1,1,sharex=True)
ax.plot(datamean.h_Laser,datamean.h_water_empQ,'x',label='EM Q')
ax.plot(datamean.h_Laser,datamean.h_water_empI,'x',label='EM I')
ax.plot([0,30],[0,30],'--k')
ax.set_ylabel('h EM (m)')
ax.set_xlabel('h Laser (m)')
ax.legend()
# ax.set_xlim(0,25)
# ax.set_ylim(0,25)


fig,ax=pl.subplots(1,1,sharex=True)
ax.plot(datamean.time,datamean.h_tot_empQ,'x',label='EM Q')
ax.plot(datamean.time,datamean.h_tot_empI,'x',label='EM I')
pl.xlabel('time (s)')
ax.set_ylabel('Total Thickness (ice+snow) (m)')
ax.legend()


plot_xy(datamean,'h_tot_empQ',origin=0,colorlim=[1,2])

plot_xy(datamean,'freeboard',origin=0,colorlim=[-.2,.2])


fig,[ax1,ax2,ax3]=pl.subplots(3,1,sharex=True)
ax1.plot(datamean.time,datamean.h_tot_empQ,'x',label='EM Q')
ax1.plot(datamean.time,datamean.h_tot_empI,'x',label='EM I')
# ax1.set_xlabel('time (s)')
ax1.set_ylabel('Total Thickness (ice+snow) (m)')
ax3.legend()

ax2.plot(datamean.time,datamean.freeboard,'x',label='freeboard')
# ax2.set_xlabel('time (s)')
ax2.set_ylabel('tot. Freeboard (m)')
ax3.legend()
ax3.plot(datamean.time,datamean.roll/np.pi*180,'x',label='roll')
ax3.plot(datamean.time,datamean.pitch/np.pi*180,'x',label='pith')
ax3.set_xlabel('time (s)')
ax3.set_ylabel('Roll/pitch angle (deg)')
ax3.legend()


fig,ax=pl.subplots(1,1,sharex=True)
ax.plot(datamean.pitch,datamean.h_tot_empQ,'x',label='h EM Q')
pl.xlabel('pitch angle (deg)')
ax.set_ylabel('Total Thickness (ice+snow) (m)')
ax.legend()


fig,ax=pl.subplots(1,1,sharex=True)
ax.plot(datamean.pitch,datamean.freeboard,'x',label='freeboard')
pl.xlabel('pitch angle (deg)')
ax.set_ylabel('tot. Freeboard (m)')
ax.legend()

# %% save inverted data
saveDataLEM(datamean, dataINS,params)




# %% -------------------------------------
#............ Emagpy Inversion ...........
#----------------------------------------
# %%% calibration climbs

data_climbs=Fit_climbs(datamean,params,
               t_str,t_stp, h_tot,w_depth=w_depth,shallow=False,
               w_cond=w_cond,d_coils=0,d_BxCoil=params['d_Bx' ],
               plot=True)



# %% invert data


Invert_data(datamean,params,
               w_cond=2408,d_coils=0,
               plot=True, d_BxCoil=params['d_Bx' ],
               dataType=['Q','I','QI'])



# %% plot inverted data
fig,ax=pl.subplots(1,1,sharex=True)
ax.plot(datamean.time,datamean.hw_invQ,'x',label='EM Q')
ax.plot(datamean.time,datamean.hw_invI,'x',label='EM I')
# ax.plot(datamean.time,datamean.hw_invQI,'x',label='EM Q+I')
ax.plot(datamean.time,datamean.h_GPS,'--k',label='GPS')
ax.plot(datamean.time,datamean.h_Laser,'--b',label='Laser')
pl.xlabel('time (s)')
ax.set_ylabel('h (m)')
ax.legend()



fig,ax=pl.subplots(1,1,sharex=True)
ax.plot(datamean.h_Laser,datamean.hw_invQ,'x',label='Laser all')
ax.plot(datamean.h_GPS,datamean.hw_invQ,'x',label='GPS deep water')
ax.plot([0,30],[0,30],'--k')
ax.set_ylabel('h EM (m)')
ax.set_xlabel('h Laser/GPS (m)')
ax.legend()
ax.set_xlim(0,25)
ax.set_ylim(0,25)



fig,ax=pl.subplots(1,1,sharex=True)
ax.plot(datamean.h_Laser,datamean.hw_invQ,'x',label='EM Q')
ax.plot(datamean.h_Laser,datamean.hw_invI,'x',label='EM I')
# ax.plot(datamean.h_Laser,datamean.hw_invQI,'x',label='EM Q+I')
ax.plot([0,30],[0,30],'--k')
ax.set_ylabel('h EM (m)')
ax.set_xlabel('h Laser (m)')
ax.legend()
# ax.set_xlim(0,25)
# ax.set_ylim(0,25)


fig,ax=pl.subplots(1,1,sharex=True)
ax.plot(datamean.time,datamean.h_totQ,'x',label='EM Q')
ax.plot(datamean.time,datamean.h_totI,'x',label='EM I')
# ax.plot(datamean.time,datamean.h_totQI,'x',label='EM Q+I')
pl.xlabel('time (s)')
ax.set_ylabel('Total Thickness (ice+snow) (m)')
ax.legend()


fig=plot_xy(datamean,'h_totQ',origin=0,colorlim=[1,6], z_label='h tot (m)')
fig.savefig(savpath+r'\Runway_xy_hTot_Q.png')

fig=plot_xy(datamean,'freeboard',origin=0,colorlim=[-.2,1],z_label='freeboard(m)')
fig.savefig(savpath+r'\Runway_xy_freeboard.png')


# %% save inverted data

saveDataLEM(datamean, dataINS,params)


#%% compare emphirical vs inversion

fig,[ax,ax2]=pl.subplots(2,1,sharex=True)
ax.plot(datamean.time,datamean.h_tot_empQ,'x',label='emphirical')
ax.plot(datamean.time,datamean.h_totQ,'x',label='inversion')
# pl.xlabel('time (s)')
ax.set_ylabel('Total Thickness (ice+snow) (m)')
ax.legend()
ax2.plot(datamean.time,datamean.h_tot_empQ-datamean.h_totQ,'x',label='Delta h')
ax2.set_xlabel('time (s)')
ax2.set_ylabel('$\Delta$h (m)')

