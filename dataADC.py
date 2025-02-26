# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 12:49:19 2023

@author: Laktop
"""

# %% import modules
import sys

import numpy as np
import matplotlib.pyplot as pl
from matplotlib.gridspec import GridSpec
import pandas as pd
import glob

from scipy import signal
from scipy import optimize
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from scipy.optimize import curve_fit,least_squares
from scipy.stats import linregress

# non anaconda libraries
try:
    from hampel import hampel   # pip install hampel
except ModuleNotFoundError:
    print('No module named "hampel"!!!! Install it otherwhise outlier filter will not work!!!')   
from cmcrameri import cm as cmCrameri


from INSLASERdata import *
import save_load as sl



# sys.path.append(r'C:\Users\Laktop')
from emagpy_seaice import Problem
from emagpy_seaice import invertHelper 


# %% define variables
SPS= 19200
ZoneInfo("UTC")

#%%  function for Postprocessing raw data


def processDataLEM(path,name, Tx_ch='ch2', Rx_ch=['ch1','ch2'],
                   plot=False, plotINS=False,
                   savefile=True,saveCSV=None,savePKL=None,
                     window=1920,freq=0,phase0=0,SPS=19200,flowpass=30,
                     autoCal=True,i_autoCal=0,i_cal=[],
                     INSkargs={},MultiFreq=False,n_freqs=3,iStart=2,dT_start=0,T_max=0,
                     **kwargs):
    
    
    # Create dictionary containing parameters
    params=locals()  # add all function arguments to dictionary
    version='v1.3'
    params['pyLEM_version']=version
    
    
    
    file=path+'/ADC'+name+r'.csv'
    fileLASER=path+'/INS'+name+'.csv'

    
    header=loadDataHeader(fileLASER)
    params.update( header)
    
    if 'distCenter' in INSkargs.keys():
        params['distCenter']=INSkargs['distCenter']
    
    else: 
        if 'distCenter' in params.keys():
            INSkargs['distCenter']=params['distCenter']
        else:
            params['distCenter']=0
        
    
    print('Load INS+Laser: ',file)
    try:
        dataINS=INSLASERdata(fileLASER,name='\\INS'+name+'.csv',**INSkargs)
        noLASERfile=False
    except FileNotFoundError:
        print('Lase file: ',file, ' can not be loaded')
        noLASERfile=True
    
    
    
    print('Reading file: ',file)
    if MultiFreq:
        #----------------------------
        #Multi frequency processing
        #----------------------------
        datamean, i_missing, gap,start_ind,freqs=loadADCraw_multiFreq(file,
                                                               f=freq,phase0=phase0,SPS=SPS,
                                                               flowpass=flowpass,window=window,
                                                               i_Tx=int(Tx_ch[2:]),Rx_ch=Rx_ch,
                                                               freqs=[1079.0228, 3900.1925, 8207.628],
                                                                **kwargs)    # ToDo: read parameters from params
        
        params['start_ind']=start_ind
        params['freqs']=freqs
        params['i_missing']=i_missing
        params['gap']=gap
        
        # To do: add parameters to params
        
        tx=int(Tx_ch[2:])
        columns=[]
        
        # normalize Rx by Tx
        #----------------------------
        NormRxbyTx_multi(datamean,Rx_ch,columns,n_freqs,tx=tx)
        
        
        # sync ADC and INS/Laser
        #----------------------------
        params['TOW_ADC0']= sync_ADC_INS(datamean,dataINS,iStart=iStart,dT_start=dT_start)
        

        
        
        # Calibrate ADC
        #----------------------------
        CalParams=Calibrate_multiFreq(datamean,dataINS,params,Rx_ch,i_cal,n_freqs=n_freqs,autoCal=autoCal,i_autoCal=i_autoCal,plot=plot)
        params['CalParams']=CalParams
        
        

    
    
    else:  
        #----------------------------
        #single frequency processing
        #----------------------------
        datamean, i_missing, gap,f,phase0=loadADCraw_singleFreq(file,
                                                               f=freq,phase0=phase0,SPS=SPS,
                                                               flowpass=flowpass,window=window,keep_HF_data=False,
                                                               i_Tx=int(Tx_ch[2:]),plot=plot,
                                                               T_max=T_max,
                                                               **kwargs)    

        # params={}
        params['f']=f
        params['phase0']=phase0
        params['i_missing']=i_missing
        params['gap']=gap
        
        print(f'Freq: {f:.2f} Hz')
        print(f'Phase lockIn: {phase0:.2f} rad')
        
        tx=int(Tx_ch[2:])
        columns=[]
        
        # normalize Rx by Tx
        #----------------------------
        NormRxbyTx(datamean,Rx_ch,columns,tx=tx)
    
        
        # sync ADC and INS/Laser
        #----------------------------
        params['TOW_ADC0']= sync_ADC_INS(datamean,dataINS,iStart=iStart,dT_start=dT_start)
        
        
        
        # Calibrate ADC
        #----------------------------
        CalParams=Calibrate(datamean,dataINS,params,Rx_ch,i_cal,autoCal=autoCal,i_autoCal=i_autoCal,plot=plot)
        params['CalParams']=CalParams
        
        
    

        

    
    if plotINS: 
        plot_summary(dataINS,getextent(dataINS),heading=False)
    
    if plot:
        try:
            plot_QandI(datamean,params,Rx_ch,MultiFreq)
        except KeyError as e:
            print("Can't find data in datamean to plot:") 
            print(e)
    
    
    
    # save files
    if savefile:
        saveCSV=True
        savePKL=True
    if saveCSV:
        save_LEM_csv(datamean,params,columns=columns)
    if savePKL:   
       fileOutputPKL=params['path']+'/LEM'+params['name']+'.pkl'
       sl.save_pkl([datamean, dataINS,params], fileOutputPKL)
    
       
    return datamean, dataINS,params



#%%  Supporting functions for loading and handling data + Postprocessing


def saveDataLEM(datamean, dataINS,params,saveCSV=True,savePKL=True):
    """
    

    Parameters
    ----------
    datamean : pandas.Dataframe
        Processed LEM data.
    dataINS : TYPE
        DESCRIPTION.
    params : dictonary
        Parameters for processed LEM data.
    saveCSV : TYPE, optional
        DESCRIPTION. The default is True.
    savePKL : TYPE, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    None.

    """
    if saveCSV:
        save_LEM_csv(datamean,params)
    if savePKL:   
       fileOutputPKL=params['path']+'/LEM'+params['name']+'.pkl'
       sl.save_pkl([datamean, dataINS,params], fileOutputPKL)



def loadDataLEM(path,name):
    file=path+'/LEM'+name+'.csv'
    return pd.read_csv(file,header=15)







def NormRxbyTx(datamean,Rx_ch,columns,tx=2):
    for i,ch in enumerate(Rx_ch):
        rx=int(ch[2:])
        j=i+1
        datamean[f'A_Rx{j:d}']=datamean[f'A{rx:d}']/datamean[f'A{tx:d}']
        datamean[f'phase_Rx{j:d}']=datamean[f'phase{rx:d}']-datamean[f'phase{tx:d}']   
        # columns.append(f'A_Rx{j:d}')
        # columns.append(f'phase_Rx{j:d}')
  
def NormRxbyTx_multi(datamean,Rx_ch,columns,n_freqs,tx=2):
    for i,ch in enumerate(Rx_ch):
        j=i+1
        for k in range(1,n_freqs+1):
            
            datamean[f'A_Rx{j:d}_f{k:d}']=datamean[f'A_Rx{j:d}_f{k:d}']/datamean[f'A_Tx_f{k:d}']
            datamean[f'phase_Rx{j:d}_f{k:d}']=datamean[f'phase_Rx{j:d}_f{k:d}']-datamean[f'phase_Tx_f{k:d}']   
            
        #     columns.append(f'A_Rx{j:d}_f{k:d}')
        #     columns.append(f'phase_Rx{j:d}_f{k:d}')      
       
        # columns.append(f'A_Tx_f{k:d}')
        # columns.append(f'phase_Tx_f{k:d}')  
    
   
def Calibrate(datamean,dataINS,params,Rx_ch,i_cal,autoCal=True,i_autoCal=0,plot=False):
    A0=[]
    phase0=[]
    
    CalParams={}
    
    if i_cal!=[]:
        print('calibrating!')
        for i,ch in enumerate(Rx_ch):
            j=i+1
            A0.append(datamean[f'A_Rx{j:d}'][i_cal].mean())
            phase0.append(datamean[f'phase_Rx{j:d}'][i_cal].mean())
        
            datamean[f'I_Rx{j:d}']=datamean[f'A_Rx{j:d}']/A0[i]*np.cos(datamean[f'phase_Rx{j:d}']-phase0[i])-1
            datamean[f'Q_Rx{j:d}']=datamean[f'A_Rx{j:d}']/A0[i]*np.sin(datamean[f'phase_Rx{j:d}']-phase0[i])
    else:
        for i,ch in enumerate(Rx_ch):
            j=i+1
            datamean[f'I_Rx{j:d}']=datamean[f'A_Rx{j:d}']*np.cos(datamean[f'phase_Rx{j:d}'])
            datamean[f'Q_Rx{j:d}']=datamean[f'A_Rx{j:d}']*np.sin(datamean[f'phase_Rx{j:d}'])
    
    
    # derive calibration parameters from automatic calibration using calibration coil
    if autoCal and dataINS.Cal.len>0:
        gs,phis, calQs2,calIs2,calQ0,calI0,start,stop,on,off=CheckCalibration(dataINS,
                                                                              datamean,
                                                                              params,
                                                                              params['f'],
                                                                              Rx_ch=Rx_ch,
                                                                              plot=plot)

    
        # Define transformation function
        def Coordinate_trans(I,Q, g, phi):
            X = Q*1j+I
            Z=g*X*np.exp(phi*1j)
            return np.real(Z), np.imag(Z)
        
        k=i_autoCal # define which  calibration cicle to use
        
        # transform Voltage data to normalized secondary field using derived calibration parameters 
        for i,ch in enumerate(Rx_ch):
            j=i+1
            

            I,Q=Coordinate_trans(datamean[f'I_Rx{j:d}']-calI0[i_autoCal][i],
                      datamean[f'Q_Rx{j:d}']-calQ0[i_autoCal][i], 
                      gs[i_autoCal][i], 
                      phis[i_autoCal][i])
            
            datamean[f'I_Rx{j:d}']=I
            datamean[f'Q_Rx{j:d}']=Q
    
    
        CalParams['g']=np.array(gs)[i_autoCal,:]
        CalParams['phi']=np.array(phis)[i_autoCal,:]
        CalParams['Q0']=np.array(calQ0)[i_autoCal,:]
        CalParams['I0']=np.array(calI0)[i_autoCal,:]
        CalParams['start']=start[i_autoCal]
        CalParams['stop']=stop[i_autoCal]
        CalParams['on']=on[i_autoCal]
        CalParams['off']=off[i_autoCal]
    
    elif dataINS.Cal.len==0:
        print('No calibration stamps found!  Autocalibration not possible!!!')
        params['autoCal']=False
    
    CalParams['A0']=A0
    CalParams['phase0']=phase0

    return CalParams



def Calibrate_multiFreq(datamean,dataINS,params,Rx_ch,i_cal,autoCal=True,i_autoCal=0,n_freqs=3,plot=False):
    A0=[]
    phase0=[]
    
    CalParams={}
    
    if i_cal!=[]:
        print('calibrating!')
        for i,ch in enumerate(Rx_ch):
            j=i+1
            for k in range(1,n_freqs+1):
                A0.append(datamean[f'A_Rx{j:d}_f{k:d}'][i_cal].mean())
                phase0.append(datamean[f'phase_Rx{j:d}_f{k:d}'][i_cal].mean())
            
                datamean[f'I_Rx{j:d}_f{k:d}']=datamean[f'A_Rx{j:d}_f{k:d}']/A0[i]*np.cos(datamean[f'phase_Rx{j:d}_f{k:d}']-phase0[i])-1
                datamean[f'Q_Rx{j:d}_f{k:d}']=datamean[f'A_Rx{j:d}_f{k:d}']/A0[i]*np.sin(datamean[f'phase_Rx{j:d}_f{k:d}']-phase0[i])
    else:
        for i,ch in enumerate(Rx_ch):
            j=i+1
            for k in range(1,n_freqs+1):
                datamean[f'I_Rx{j:d}_f{k:d}']=datamean[f'A_Rx{j:d}_f{k:d}']*np.cos(datamean[f'phase_Rx{j:d}_f{k:d}'])
                datamean[f'Q_Rx{j:d}_f{k:d}']=datamean[f'A_Rx{j:d}_f{k:d}']*np.sin(datamean[f'phase_Rx{j:d}_f{k:d}'])
    
    
    # derive calibration parameters from automatic calibration using calibration coil
    if autoCal and dataINS.Cal.len>0:
        gs,phis, calQs2,calIs2,calQ0,calI0,start,stop,on,off=CheckCalibration_multiFreq(dataINS,datamean,
                                                                                        params,
                                                                                        plot=plot)

    
        # Define coordinates transformation function, adjust phase and transform to ppm
        def Coordinate_trans(I,Q, g, phi):
            X = Q*1j+I
            Z=g*X*np.exp(phi*1j)
            return np.real(Z), np.imag(Z)
        
        k=i_autoCal # define which  calibration cicle to use
        
        # transform Voltage data to normalized secondary field using derived calibration parameters 
        for i,ch in enumerate(Rx_ch):
            j=i+1
            
            for k in range(1,n_freqs+1):
                # print('i:',i)
                # print('j:',j)
                # print('k:',k)
                # print('calI0:',calI0)

                I,Q=Coordinate_trans(datamean[f'I_Rx{j:d}_f{k:d}']-calI0[i_autoCal][i][k-1],
                          datamean[f'Q_Rx{j:d}_f{k:d}']-calQ0[i_autoCal][i][k-1], 
                          gs[i_autoCal][i][k-1], 
                          phis[i_autoCal][i][k-1])
                
                datamean[f'I_Rx{j:d}_f{k:d}']=I
                datamean[f'Q_Rx{j:d}_f{k:d}']=Q
    
    
        CalParams['g']=np.array(gs)[i_autoCal]
        CalParams['phi']=np.array(phis)[i_autoCal]
        CalParams['Q0']=np.array(calQ0)[i_autoCal]
        CalParams['I0']=np.array(calI0)[i_autoCal]
        CalParams['start']=start[i_autoCal]
        CalParams['stop']=stop[i_autoCal]
        CalParams['on']=on[i_autoCal]
        CalParams['off']=off[i_autoCal]
    elif dataINS.Cal.len==0:
        print('No calibration stamps found!  Autocalibration not possible!!!')
        params['autoCal']=False
    
    CalParams['A0']=A0
    CalParams['phase0']=phase0
    
    return CalParams

def save_LEM_csv(datamean,params,columns=[],fileOutput=''):
        
    if len(fileOutput)==0:
        fileOutput=params['path']+'/LEM'+params['name']+'.csv'
        
    if params['MultiFreq']:
        freqs=params['freqs']
        # write file header
        write_file_header_multi(fileOutput,params,
                                  params['i_missing'],params['gap'],
                                  params['Tx_ch'],params['Rx_ch'],
                                  len(freqs))
        
        
        #save datamean to file
        #----------------------------
        # columns to save
        columns+=['t', 'time','TOW', 'lat', 'lon', 'h_GPS',  'h_Laser', 'diff_hGPSLaser','roll', 'pitch','heading', 'velX', 'velY', 'velZ',  
                  'signQ', 'TempLaser' ] #
        
        for i,ch in enumerate(params['Rx_ch']):

            columns.extend([f'I_Rx{i+1:d}_f{k:d}' for k in range(1,len(freqs)+1)])
            columns.extend([f'Q_Rx{i+1:d}_f{k:d}' for k in range(1,len(freqs)+1)])
            columns.extend([f'A_Rx{i+1:d}_f{k:d}' for k in range(1,len(freqs)+1)])
            columns.extend([f'phase_Rx{i+1:d}_f{k:d}' for k in range(1,len(freqs)+1)])
            columns.append(f'A_Tx_f{k:d}')
            columns.append(f'phase_Tx_f{k:d}') 
        
        for i in range(1,2+len(params['Rx_ch'])):    
            columns.extend([f'Q_ch{i:d}_f{k:d}' for k in range(1,len(freqs)+1)])
            columns.extend([f'I_ch{i:d}_f{k:d}' for k in range(1,len(freqs)+1)])

            
    else: # SINGLE FREQUENCY
        # write file header
        write_file_header(fileOutput,params,
                          params['i_missing'],params['gap'],
                          params['Tx_ch'],params['Rx_ch'])
        
        
        #save datamean to file
        #----------------------------
        # columns to save
        columns+=['t', 'time','TOW', 'lat', 'lon', 'h_GPS',  'h_Laser', 'diff_hGPSLaser', 'roll', 'pitch','heading', 'velX', 'velY', 'velZ',  
                  'signQ', 'TempLaser',
                  'Q1', 'I1', 'Q2', 'I2', 'Q3', 'I3','A1', 'phase1','A2', 'phase2', 'A3', 'phase3' ] #
    

        for i,ch in enumerate(params['Rx_ch']):
            columns.extend([f'I_Rx{i+1:d}',f'Q_Rx{i+1:d}',f'A_Rx{i+1:d}',f'phase_Rx{i+1:d}'])
            
        
        columns.extend(['h_water_empQ', 'h_water_empI', 'h_tot_empQ', 'h_tot_empI'])
            
            
    # remove columns to save that are not in datamean
    for c in columns:
        if not (datamean.keys()==c).any():
            columns.remove(c)
            print('Column not in datamean:', c) 
            
    # remove duplicate entries
    res = []
    [res.append(x) for x in columns if x not in res]
    columns=res
            
    try:
        datamean.to_csv(fileOutput,mode='a',index=True,header=True,columns=columns)
    except Exception as e: 
            print("error. Can't save data ")
            print(e)
            print('Columns to write:', columns)
            print('Columns in datamean:', datamean.keys()) 
            
            

def write_file_header_multi(fileOutput,params,i_missing,gap,Tx_ch,Rx_ch,n_freqs):

    file=open(fileOutput,'w')

    file.write("# Signal extracted with LockIn form raw data\n")
    file.write("# Processing date: {:} \tScript verion: {:s}  \n\n".format( datetime.now().strftime("%Y-%m-%d, %H:%M:%S"),params['pyLEM_version']))
    file.write("# Freqeuncies LockIn: {:} Hz\n".format(str(params['freqs'])))
    file.write("# SPS: {:}\n\n".format(SPS))
    file.write("# Frequency low pass: {:}\n".format(params['flowpass']))
    file.write("# TxChannel: {:}\n".format(Tx_ch))
    file.write("# Rx channels: {:}\n".format(str(Rx_ch)))
    
    CalParams=params['CalParams']
    if params['i_cal']!=[]:
        file.write("# Index calibration: [{:},{:}]\n".format(str(i_cal[0]),str(i_cal[1])))
        for i,ch in enumerate(Rx_ch):
            file.write("# A0_Rx{:d}= {:f}, phase0_Rx{:d}= {:f},\n".format(i+1,CalParams['A0'][i],
                                                                          i+1,CalParams['phase0'][i]))
    else:
        file.write("# No index calibration")
    
    if params['autoCal']:
        file.write("# auto calibration: start:{:.4f},  stop:{:.4f} \n".format(CalParams['start'],CalParams['start']))
        for i,ch in enumerate(Rx_ch):
            for k in range(1,n_freqs+1):
                file.write("#\t g_Rx{0:d}_f{1:d}= {2:f}, phi_Rx{0:d}_f{1:d}= {3:f},\n".format(i+1,k,CalParams['g'][i,k-1],CalParams['phi'][i,k-1]))
    else:
        file.write("# No auto calibration")    
        
    file.write("#\n")
    file.write("# Missing index: {:}, Gap sizes: {:}\n".format(str( i_missing), str( gap)))
    file.write("##\n")
    file.close()


def write_file_header(fileOutput,params,i_missing,gap,Tx_ch,Rx_ch):

    file=open(fileOutput,'w')

    file.write("# Signal extracted with LockIn form raw data\n")
    file.write("# Processing date: {:} \tScript version: {:s}  \n\n".format( datetime.now().strftime("%Y-%m-%d, %H:%M:%S"),params['pyLEM_version']))
    file.write("# Freqeuncy LockIn: {:} Hz\n".format(params['f']))
    file.write("# Phase LockIn: {:}\n".format(params['phase0']))
    file.write("# SPS: {:}\n\n".format(params['SPS']))
    file.write("# Frequency low pass: {:}\n".format(params['flowpass']))
    file.write("# Averaging window size: {:}\n".format(params['window']))
    file.write("# TxChannel: {:}\n".format(Tx_ch))
    file.write("# Rx channels: {:}\n".format(str(Rx_ch)))
    
    CalParams=params['CalParams']
    try:
        if params['i_cal']!=[]:
            file.write("# Index calibration: [{:},{:}]\n".format(str(i_cal[0]),str(i_cal[1])))
            for i,ch in enumerate(Rx_ch):
                file.write("# A0_Rx{:d}= {:f}, phase0_Rx{:d}= {:f},\n".format(i+1,CalParams['A0'][i],                                                                i+1,CalParams['phase0'][i]))
        else:
            file.write("# No index calibration")
    except NameError:
        file.write("# No index calibration")
        
    if params['autoCal']:
        file.write("# auto calibration: start:{:.4f},  stop:{:.4f} \n".format(CalParams['start'],CalParams['start']))
        for i,ch in enumerate(Rx_ch):
            file.write("# g_Rx{:d}= {:f}, phi_Rx{:d}= {:f},\n".format(i+1,CalParams['g'][i],i+1,CalParams['phi'][i]))
    else:
        file.write("# No auto calibration")    
        
    file.write("#\n")
    file.write("# Missing index: {:}, Gap sizes: {:}\n".format(str( i_missing), str( gap)))
    file.write("##\n")
    file.close()


def trim_data(t0,t1,datamean,params):
    lims_i=[np.searchsorted(datamean.time.values,t0),np.searchsorted(datamean.time.values,t1)]
    params['trim_times']=[t0,t1]
    return datamean.iloc[lims_i[0]:lims_i[1]].copy()
    



def filter_outliers(datamean,params,window_size=10,deviation=3.0,plot=False):
    """
    Filter outliers over a running window and substitute them with median. Use Hampel function. 
    Datapoints are considered outliers if the deviation excheed the Median Absolute Deviation (MAD)
    multiplied by deviation parameter. 

    Parameters
    ----------
    datamean : pandas.Dataframe
        Processed LEM data.
    params : dictonary
        Parameters for processed LEM data.
    window_size : TYPE, optional
        Number of datapoints in running window. The default is 10.
    deviation : TYPE, optional
        Define the deviation form the MAD for considering datapoints as outliers . The default is 3.0.
    plot : TYPE, optional
        Plot filtered data. The default is False.

    Returns
    -------
    filtQ : TYPE
        DESCRIPTION.
    filtI : TYPE
        DESCRIPTION.

    """

    try: 
        datamean.Q_Rx1=datamean['Q_Rx1_original'].copy()
        datamean.I_Rx1=datamean['I_Rx1_original'].copy()
    except KeyError:
        datamean['Q_Rx1_original']=datamean.Q_Rx1.values
        datamean['I_Rx1_original']=datamean.I_Rx1.values
        
    filtQ = hampel(datamean.Q_Rx1_original, window_size,deviation)
    filtI = hampel(datamean.I_Rx1_original, window_size,deviation)
    
    datamean.Q_Rx1=filtQ.filtered_data.values
    datamean.I_Rx1=filtI.filtered_data.values
    
    ind=filtQ.thresholds>50*filtQ.medians
    filtQ.thresholds[ind]=50*filtQ.medians[ind]
    ind=filtI.thresholds>50*filtI.medians
    filtI.thresholds[ind]=50*filtI.medians[ind]
    
    if plot:
        fig,[ax,ax2]=pl.subplots(2,1,sharex=True)
        ax.plot(datamean.time,datamean['Q_Rx1_original'].values,':',label='Q Rx')
        ax.fill_between(datamean.time, filtQ.medians + filtQ.thresholds,
                             filtQ.medians - filtQ.thresholds, color='gray', 
                             alpha=0.5, label='Median +- Threshold')
        ax.plot(datamean.time,filtQ.filtered_data,'x',label='filtered')
        ax.scatter(datamean.time.values[filtQ.outlier_indices],
                   datamean[f'Q_Rx1_original'].values[filtQ.outlier_indices],
                   marker='o',edgecolor='g',facecolor='None',label='outliers')
        ax.set_title('window={:d}, deviation={:.1f} '.format(window_size,deviation))
        ax.set_ylabel('Q (-)')
        ax.set_xlabel('time (s)')
        ax.legend()
        ax.set_ylim([np.min(filtQ.filtered_data.values),np.max(filtQ.filtered_data.values)])
        
        ax2.plot(datamean.time,datamean['I_Rx1_original'].values,':',label='I Rx')
        ax2.fill_between(datamean.time, filtI.medians + filtI.thresholds,
                             filtI.medians - filtI.thresholds, color='gray', 
                             alpha=0.5, label='Median +- Threshold')
        
        ax2.plot(datamean.time,filtI.filtered_data,'x',label='filtered')
        ax2.scatter(datamean.time.values[filtI.outlier_indices],
                    datamean[f'I_Rx1_original'].values[filtI.outlier_indices],
                   marker='o',edgecolor='g',facecolor='None',label='outliers')
        ax2.set_ylabel('I (-)')
        ax2.set_xlabel('time (s)')
        ax2.legend()
        ax2.set_ylim([np.min(filtI.filtered_data.values),np.max(filtI.filtered_data.values)])

            
    return filtQ,filtI


def filter_median(data,column,window_size=10):
    """
    Filter outliers over a running window and substitute them with median. Use Hampel function. 
    Datapoints are considered outliers if the deviation excheed the Median Absolute Deviation (MAD)
    multiplied by deviation parameter. 

    Parameters
    ----------
    datamean : pandas.Dataframe
        Processed LEM data.
    params : dictonary
        Parameters for processed LEM data.
    window_size : TYPE, optional
        Number of datapoints in running window. The default is 10.
    deviation : TYPE, optional
        Define the deviation form the MAD for considering datapoints as outliers . The default is 3.0.
    plot : TYPE, optional
        Plot filtered data. The default is False.

    Returns
    -------
    filtQ : TYPE
        DESCRIPTION.
    filtI : TYPE
        DESCRIPTION.

    """

    try: 
        datamean.Q_Rx1=datamean['Q_Rx1_original'].copy()
        datamean.I_Rx1=datamean['I_Rx1_original'].copy()
    except KeyError:
        datamean['Q_Rx1_original']=datamean.Q_Rx1.values
        datamean['I_Rx1_original']=datamean.I_Rx1.values
        
    filtQ = hampel(datamean.Q_Rx1_original, window_size,deviation)
    filtI = hampel(datamean.I_Rx1_original, window_size,deviation)
    
    datamean.Q_Rx1=filtQ.filtered_data.values
    datamean.I_Rx1=filtI.filtered_data.values
    
    ind=filtQ.thresholds>50*filtQ.medians
    filtQ.thresholds[ind]=50*filtQ.medians[ind]
    ind=filtI.thresholds>50*filtI.medians
    filtI.thresholds[ind]=50*filtI.medians[ind]
    
    if plot:
        fig,[ax,ax2]=pl.subplots(2,1,sharex=True)
        ax.plot(datamean.time,datamean['Q_Rx1_original'].values,':',label='Q Rx')
        ax.fill_between(datamean.time, filtQ.medians + filtQ.thresholds,
                             filtQ.medians - filtQ.thresholds, color='gray', 
                             alpha=0.5, label='Median +- Threshold')
        ax.plot(datamean.time,filtQ.filtered_data,'x',label='filtered')
        ax.scatter(datamean.time.values[filtQ.outlier_indices],
                   datamean[f'Q_Rx1_original'].values[filtQ.outlier_indices],
                   marker='o',edgecolor='g',facecolor='None',label='outliers')
        ax.set_title('window={:d}, deviation={:.1f} '.format(window_size,deviation))
        ax.set_ylabel('Q (-)')
        ax.set_xlabel('time (s)')
        ax.legend()
        ax.set_ylim([np.min(filtQ.filtered_data.values),np.max(filtQ.filtered_data.values)])
        
        ax2.plot(datamean.time,datamean['I_Rx1_original'].values,':',label='I Rx')
        ax2.fill_between(datamean.time, filtI.medians + filtI.thresholds,
                             filtI.medians - filtI.thresholds, color='gray', 
                             alpha=0.5, label='Median +- Threshold')
        
        ax2.plot(datamean.time,filtI.filtered_data,'x',label='filtered')
        ax2.scatter(datamean.time.values[filtI.outlier_indices],
                    datamean[f'I_Rx1_original'].values[filtI.outlier_indices],
                   marker='o',edgecolor='g',facecolor='None',label='outliers')
        ax2.set_ylabel('I (-)')
        ax2.set_xlabel('time (s)')
        ax2.legend()
        ax2.set_ylim([np.min(filtI.filtered_data.values),np.max(filtI.filtered_data.values)])

            
    return 


# %% load raw data and Lock-In filter

# def LockInADCrawfile(path,name, Tx_ch='ch2', Rx_ch=['ch1'],
#                      plot=False,
#                      window=1920,freq=1063.3985,phase0=0,SPS=19200,flowpass=50,i_cal=[],
#                      **kwargs):
#     '''

#     Parameters
#     ----------
#     path : TYPE
#         DESCRIPTION.
#     name : TYPE
#         DESCRIPTION.
#     Tx_ch : TYPE, optional
#         DESCRIPTION. The default is 'ch3'.
#     Rx_ch : TYPE, optional
#         DESCRIPTION. The default is ['ch1','ch2'].
#     plot : TYPE, optional
#         DESCRIPTION. The default is False.
#     window : TYPE, optional
#         DESCRIPTION. The default is 1920.
#     freq : TYPE, optional
#         DESCRIPTION. The default is 1063.3985.
#     phase0 : TYPE, optional
#         DESCRIPTION. The default is 0.
#     SPS : TYPE, optional
#         DESCRIPTION. The default is 19200.
#     flowpass : TYPE, optional
#         DESCRIPTION. The default is 50.
#     i_cal : TYPE, optional
#         DESCRIPTION. The default is [].
#     **kwargs : TYPE
#         DESCRIPTION.

#     Returns
#     -------
#     datamean : TYPE
#         DESCRIPTION.

#     '''
    
#     file=path+r'\\'+name+r'.csv'
#     savefile=path+'\\'+name+r'LockIn.csv'
#     version='v1.0'
    
    
#     print('Reading file: ',file)
#     datamean, i_missing, gap,f,phase0=loadADCraw_singleFreq(file,
#                                                    f=freq,phase0=phase0,SPS=SPS,
#                                                    flowpass=flowpass,window=window,keep_HF_data=False,
#                                                    i_Tx=int(Tx_ch[2:]),
#                                                    **kwargs)    
    
#     print(f'Freq: {f:.2f} Hz')
#     print(f'Phase: {phase0:.2f} rad')
    
#     tx=int(Tx_ch[2:])
#     columns=[]
    
#     # normalize Rx by Tx
#     for i,ch in enumerate(Rx_ch):
#         rx=int(ch[2:])
#         j=i+1
#         datamean[f'A_Rx{j:d}']=datamean[f'A{rx:d}']/datamean[f'A{tx:d}']
#         datamean[f'phase_Rx{j:d}']=datamean[f'phase{rx:d}']-datamean[f'phase{tx:d}']   
#         columns.append(f'A_Rx{j:d}')
#         columns.append(f'phase_Rx{j:d}')


#     A0,phase0=Calibrate(datamean,Rx_ch,i_cal)
 
    
#     # write file header
#     file=open(savefile,'w')
    
#     file.write("# Signal extracted with LockIn form raw data\n")
#     file.write("# Processing date: {:} \tScript verion: {:s}  \n\n".format( datetime.now().strftime("%Y-%m-%d, %H:%M:%S"),version))
#     file.write("# Freqeuncy LockIn: {:} Hz\n".format(freq))
#     file.write("# Phase LockIn: {:}\n".format(phase0))
#     file.write("# SPS: {:}\n\n".format(SPS))
#     file.write("# Frequency low pass: {:}\n".format(flowpass))
#     file.write("# Averaging window size: {:}\n".format(window))
#     file.write("# TxChannel: {:}\n".format(Tx_ch))
#     file.write("# Rx channels: {:}\n".format(str(Rx_ch)))
#     if i_cal!=[]:
#         file.write("# Index calibration: [{:},{:}]\n".format(str(i_cal[0]),str(i_cal[1])))
#         for i,ch in enumerate(Rx_ch):
#             file.write("# A0_Rx{:d}= {:f}, phase0_Rx{:d}= {:f},\n".format(i+1,A0[i],i+1,phase0[i]))
#     else:
#         file.write("# No calibration")
        
#     file.write("#\n")
#     file.write("# Missing index: {:}, Gap sizes: {:}\n".format(str( i_missing), str( gap)))
#     file.write("##\n")
#     file.close()
    
#     # columns to save
#     columns+=[ 'Q1', 'I1', 'Q2', 'I2', 'Q3', 'I3','A1', 'phase1','A2', 'phase2', 'A3', 'phase3' ] #'t', 'TOW', 'lat', 'lon', 'h_GPS',  'h_Laser', 'roll', 'pitch','heading', 'velX', 'velY', 'velZ',  'signQ', 'TempLaser',
    
#     #save to file
#     try:
#         datamean.to_csv(savefile,mode='a',index=True,header=True,columns=columns)
#     except Exception as e: 
#             print("error. Can't save data ")
#             print(e)
#             print('Columns to write:', columns)
#             print('Columns to in datamean:', datamean.keys())
    
#     if plot:
#         for i,ch in enumerate(Rx_ch):
#             rx=int(ch[2:])
#             j=i+1
#             pl.figure()
#             pl.plot(datamean.index/SPS,datamean[f'Q_Rx{j:d}'],'x',label='Q Rx')
#             pl.plot(datamean.index/SPS,datamean[f'I_Rx{j:d}'],'x',label='I Rx')
#             pl.ylabel('amplitude (-)')
#             pl.xlabel('time (s)')
#             pl.legend()
#             pl.title(ch)
        
#             pl.figure()
#             pl.plot(datamean.index/SPS,datamean[f'A{rx:d}'],'x', label='Amplitude Rx signal')
#             pl.plot(datamean.index/SPS,datamean[f'Q{rx:d}'],'x', label='Quadr Rx')
#             pl.plot(datamean.index/SPS,datamean[f'I{rx:d}'],'x', label='Inph Rx')
#             pl.ylabel('amplitude (-)')
#             pl.xlabel('time (s)')
#             pl.legend()
#             pl.title(ch)

#         pl.figure()
#         pl.plot(datamean.index/SPS,datamean[f'A{tx:d}'],'x', label='Amplitude Tx')
#         pl.plot(datamean.index/SPS,datamean[f'Q{tx:d}'],'x', label='Quadr Tx')
#         pl.plot(datamean.index/SPS,datamean[f'I{tx:d}'],'x', label='Inph Tx')
#         pl.ylabel('amplitude (-)')
#         pl.xlabel('time (s)')
#         pl.legend()
#         pl.title(Tx_ch)
        
        
#         pl.figure()
#         ax=pl.subplot(211)
#         ax2=pl.subplot(212)
#         for i,ch in enumerate(Rx_ch):
#             rx=int(ch[2:])
#             j=i+1
#             ax.plot(datamean.index/SPS,datamean[f'Q_Rx{j:d}'],'x',label=ch)
#             ax2.plot(datamean.index/SPS,datamean[f'I_Rx{j:d}'],'x',label=ch)
#         ax.set_ylabel('Quadrature (-)')
#         ax.set_xlabel('time (s)')
#         ax2.set_ylabel('InPhase (-)')
#         ax2.set_xlabel('time (s)')
#         ax.legend()
#         ax2.legend()
    
#     return datamean

def loadADCraw_singleFreq(file,window=1920,f=0,phase0=0,SPS=19200,
                          flowpass=50,chunksize=19200,keep_HF_data=False,
                          findFreq=True,i_Tx=3,i_blok=[],plot=True,T_max=0,
                          **kwargs):
    """
    Read raw data file in blocks and extract signal over LockIn with lowpass filter. Single frequency. 

    Parameters
    ----------
    file : string
        path to file containing rawdata.
    window : TYPE, optional
        DESCRIPTION. The default is 1920.
    f : TYPE, optional
        DESCRIPTION. The default is 0.
    phase0 : TYPE, optional
        DESCRIPTION. The default is 0.
    SPS : TYPE, optional
        DESCRIPTION. The default is 19200.
    flowpass : TYPE, optional
        DESCRIPTION. The default is 50.
    chunksize : TYPE, optional
        DESCRIPTION. The default is 115200.
    findFreq : TYPE, optional
        DESCRIPTION. The default is True.
    i_Tx : TYPE, optional
        DESCRIPTION. The default is 3.
    i_blok : TYPE, optional
        DESCRIPTION. The default is [].
    T_max : TYPE, optional
        Data after T_mas (s) are not loaded. The default is []
    **kwargs : TYPE
        DESCRIPTION.

    Raises
    ------
    ValueError
        DESCRIPTION.
    
        DESCRIPTION.


     Returns
     -------
     datamean : TYPE
         DESCRIPTION.
     i_missing : TYPE
         DESCRIPTION.
     gap : TYPE
         DESCRIPTION.
     f : TYPE
         DESCRIPTION.
     phase0 : TYPE
         DESCRIPTION.

    """
    #filter the signal using a low pass filter
    flowpass_norm=flowpass/(SPS/2)
    b,a=signal.butter(3,flowpass_norm,'low')
    
    
    i_missing=np.array([])
    gap=np.array([])
    
    f0=f
    if findFreq:   # upload a chunk of data and determine the frequency
        
    

        
        
        threshold=0.5
        win_f0=100
        dt_min=0.2
        detrend=False
        df=0.005
        n_fmaxsearch=200
        drop_startup=15000
        if len(i_blok)==2:
            ia=0
            ib=i_blok[1]-i_blok[0]
        
            data=np.genfromtxt(file, dtype='int32' , delimiter=',', usecols=[0,1,2,3],
                           converters={1:convert,2:convert,3:convert},
                           skip_header=i_blok[0],max_rows=ib+1)
        
        
        elif len(i_blok)==0:  # find starting poit of data 
            data=np.genfromtxt(file, dtype='int32' , delimiter=',', usecols=[0,1,2,3],
                           converters={1:convert,2:convert,3:convert},
                           max_rows=200000)
        
            i_on,i_off=getIndexBlocks(data[:,i_Tx],threshold=threshold,parameter='std',
                                      plot=plot,window=win_f0,detrend=detrend) 
            print(i_on)      
            print(i_off)
            
            if len(i_on)!=len(i_off):
                raise ValueError('I_on and I_off have not same lenght! You need to fix this or handle exception!')
            
            # filter times
            di=dt_min*SPS    
            ii=(i_off- i_on)>di
            i_on=i_on[ii]
            i_off=i_off[ii]
            
            if plot:
                ax=pl.gca()
                for i in i_on:
                    ax.plot([i,i],[-1,1],'g--')
                for i in i_off:
                    ax.plot([i,i],[-1,1],'g.-')
                    pl.show()
            
            ia=i_on[0]+drop_startup
            ib=i_off[0]
            

        else:
            raise ValueError('i_block must be a list with two limits. Frequency can not be determined!')
        # ia=SPS*5
        # ib=SPS*8
        f,A,phase0,Q,I= maxCorr(data[ia:ib,i_Tx],df=df,n=n_fmaxsearch,plot=plot,f0=f0)
        
        if plot:
            plotraw2(data[ia:ib,:])
        
    # load file in chunks as pandas dataframe
    df = pd.read_csv(file, 
                     chunksize=chunksize,
                     converters={1:convert,2:convert,3:convert}, 
                     names=['ind','ch1','ch2','ch3' ],
                     skiprows=1, 
                     index_col=0,
                     comment='#')
    
    
    zQ1=signal.lfilter_zi(b,a)
    zI1=signal.lfilter_zi(b,a)
    zQ2=signal.lfilter_zi(b,a)
    zI2=signal.lfilter_zi(b,a)
    zQ3=signal.lfilter_zi(b,a)
    zI3=signal.lfilter_zi(b,a)
    
    data=pd.DataFrame()
    datamean=pd.DataFrame()
    chunkrest=pd.DataFrame()
    tic = time.perf_counter()
    i=0
    
    
    for chunk in df:
        i+=1
        j,g=find_missing(chunk,pr=False)
        i_missing=np.concatenate((i_missing,j))
        gap=np.concatenate((gap,g))
        
        try:
            chunk2=chunk.reindex(np.arange(chunk.index[0],chunk.index[-1]),fill_value=0)
            chunk2=pd.concat([chunkrest,chunk2])
        except MemoryError:
           print('Chunck number: {:d}, Index A: {:d}  Index B: {:d} '.format(i,chunk.index[0],chunk.index[-1]))
           print('i_missing:')
           print(i_missing)
           print('gap:')
           print(gap)
           raise
            
        
        # LockIn + filter
        try:
            ia= chunk2.index[0]
            ib= chunk2.index[-1]+1
        except IndexError:
            print('Error. Chunck number: {:d}, Index A: {:d}  Index B: {:d} '.format(i,chunk.index[0],chunk.index[-1]))
            raise 
            
        s=np.sin(2*np.pi*f*np.arange(ia,ib)/SPS+phase0)
        c=np.cos(2*np.pi*f*np.arange(ia,ib)/SPS+phase0)
        chunk2['Q1'],zQ1=signal.lfilter(b,a,chunk2.ch1*s,zi=zQ1)
        chunk2['I1'],zI1=signal.lfilter(b,a,chunk2.ch1*c,zi=zI1)
        chunk2['Q2'],zQ2=signal.lfilter(b,a,chunk2.ch2*s,zi=zQ2)
        chunk2['I2'],zI2=signal.lfilter(b,a,chunk2.ch2*c,zi=zI2)
        chunk2['Q3'],zQ3=signal.lfilter(b,a,chunk2.ch3*s,zi=zQ3)
        chunk2['I3'],zI3=signal.lfilter(b,a,chunk2.ch3*c,zi=zI3)


        
        #Get mean values
        datamean=pd.concat([datamean,chunk2.rolling(window,center=True,min_periods=1,step=window).mean()[1:]])
        #datamean=pd.concat([datamean,chunk2.rolling(window,center=True,min_periods=1).mean()[int(window/2)::window]])   # using step argument instead of slicing shulud be more efficient but need a higher version of pandas 
        
        
        
        i_rest=int(chunk2.index[-1]-(chunk2.index[-1]-chunk2.index[0])%window) # get index of last window of running window 
        chunkrest=chunk.loc[i_rest:]
        
        
        
        # to be dropped
        if keep_HF_data:
            chunk2['A1']=np.sqrt(chunk2.I1**2+chunk2.Q1**2)
            chunk2['phase1']=np.arctan2(chunk2.I1,chunk2.Q1)
            chunk2['A2']=np.sqrt(chunk2.I2**2+chunk2.Q2**2)
            chunk2['phase2']=np.arctan2(chunk2.I2,chunk2.Q2)
            chunk2['A3']=np.sqrt(chunk2.I3**2+chunk2.Q3**2)
            chunk2['phase3']=np.arctan2(chunk2.I3,chunk2.Q3)
            data=pd.concat([data,chunk2.loc[:i_rest]])
        
        # stop loading data if ovet T_max
        if T_max>0 and T_max*SPS<=i_rest:
            break
        
    # get phase and amplitude
    datamean['A1']=np.sqrt(datamean.I1**2+datamean.Q1**2)
    datamean['phase1']=np.arctan2(datamean.I1,datamean.Q1)
    datamean['A2']=np.sqrt(datamean.I2**2+datamean.Q2**2)
    datamean['phase2']=np.arctan2(datamean.I2,datamean.Q2)
    datamean['A3']=np.sqrt(datamean.I3**2+datamean.Q3**2)
    datamean['phase3']=np.arctan2(datamean.I3,datamean.Q3)

    
    toc = time.perf_counter()
    print(f"Loaded and processed data in {toc - tic:0.1f} seconds") 
    
    if keep_HF_data:
        return datamean, i_missing, gap,f,phase0, data
    else:
        return datamean, i_missing, gap,f,phase0


def loadADCraw_multiFreq(file,SPS=19200,
                          flowpass=50,chunksize=115200,
                          freqs=[1079.0228, 3900.1925, 8207.628],
                          findFreq=False,
                          i_Tx=3,Rx_ch=['ch1'],
                          dT_spacing=0.3125,
                          dT_lBlock=0.29,
                          dT_quite=0.02,
                          dT_raise=0.01, # time from signal start before signal can be used for lockin. Drop this data at beginning of each block
                          t_start=0.122, # start time of data
                          dt_check=0.02,
                          winGetBlock=20,
                          threshold=0.4,
                          threshold2=0.3,
                          **kwargs):
    """
    Read raw data file in blocks and extract signal over LockIn with lowpass filter. 
    Multifrequency version. The chuncks with different frequencies are detected and Lockin is applied and mean calculated for each chunk.  

    Parameters
    ----------
    file : string
        path to file containing rawdata.
    window : TYPE, optional
        DESCRIPTION. The default is 1920.
    f : TYPE, optional
        DESCRIPTION. The default is 0.
    phase0 : TYPE, optional
        DESCRIPTION. The default is 0.
    SPS : TYPE, optional
        DESCRIPTION. The default is 19200.
    flowpass : TYPE, optional
        DESCRIPTION. The default is 50.
    chunksize : TYPE, optional
        DESCRIPTION. The default is 115200.
    findFreq : TYPE, optional
        DESCRIPTION. The default is True.
    i_Tx : TYPE, optional
        DESCRIPTION. The default is 3.
    **kwargs : TYPE
        DESCRIPTION.

    Raises
    ------
    ValueError
        DESCRIPTION.
    
        DESCRIPTION.


     Returns
     -------
     datamean : TYPE
         DESCRIPTION.
     i_missing : TYPE
         DESCRIPTION.
     gap : TYPE
         DESCRIPTION.
     f : TYPE
         DESCRIPTION.
     phase0 : TYPE
         DESCRIPTION.

    """
    
    
    # SET CONSTANTS AND PROCESS INPUTS

    n_ch=1+len(Rx_ch)
    n_freq=len(freqs)
    
    
    #setting up lowpass filter
    flowpass_norm=flowpass/(SPS/2)
    b,a=signal.butter(3,flowpass_norm,'low')
    
    
    i_missing=np.array([])
    gap=np.array([])
    
        
    
    df = pd.read_csv(file, 
                     chunksize=chunksize,
                     converters={1:convert,2:convert,3:convert}, 
                     names=['ind','ch1','ch2','ch3' ],
                     skiprows=1, 
                     index_col=0,
                     comment='#')
    

    
    datamean=pd.DataFrame()
    chunkrest=pd.DataFrame()
    start_ind=np.array([])
    tic = time.perf_counter()
    time_LockIn=0
    time_indexes=0
    i_chunk=0
   
    
    for chunk in df:
        i_chunk+=1
        print('Processing chunk number: {:d}'.format(i_chunk))
        
        
        
        
        j,g=find_missing(chunk,pr=False)
        i_missing=np.concatenate((i_missing,j))
        gap=np.concatenate((gap,g))
        try:
            chunk2=chunk.reindex(np.arange(chunk.index[0],chunk.index[-1]),fill_value=0)  # fill missing values
            chunk2=pd.concat([chunkrest,chunk2])
        except MemoryError:
           print('Chunck number: {:d}, Index A: {:d}  Index B: {:d} '.format(i_chunk,chunk.index[0],chunk.index[-1]))
           print('i_missing:')
           print(i_missing)
           print('gap:')
           print(gap)
           raise
            
        
        start_time = time.perf_counter()
        # Get frequecy switch indixes (start) 
        i1,i2=getIndexBlocks2(np.array(chunk2[f'ch{i_Tx:d}']),threshold=threshold,window=winGetBlock,detrend=False,threshold2=threshold2)
        
        di=dT_spacing*SPS*0.9   # minimum distance in sample number
        N_drop=1
        while(N_drop>0):
            ii=np.where(np.diff(i1)<(di))[0] #find start times with difference smaller than 90% of dT_spacing
            if len(ii)>0:
                indices_to_remove=ii[np.where(np.diff(ii)>1)[0]]+1 # exclude indexes for consecutives elements  
                indices_to_remove=np.append(indices_to_remove, ii[0]+1) # include first element found
                # print(indices_to_remove)
                i1 = np.delete(i1, indices_to_remove)
                N_drop=len(indices_to_remove)
            else:
                N_drop=0
        time_indexes+= time.perf_counter()-start_time 
        
        if (i1[-1]+int(dT_lBlock*SPS)) >=len(chunk2):  # drop last start index if stop index is not in chunk2
            i1=i1[:-1]
        
        if (len(i1)%n_freq)>0:   # drop freq blocks of last incomplete frequency cycle if it is the case
            i_st=i1[:-(len(i1)%n_freq)]
        else:
            i_st=i1
        
        i_sp=i_st+int(dT_lBlock*SPS)  # stop indices
        N=int(np.ceil(len(i_st)/n_freq))
        
        

        
        # LockIn + filter
        start_time = time.perf_counter() 
        As=np.zeros([N,n_freq,n_ch])
        Is=np.zeros_like(As)
        Qs=np.zeros_like(As)
        phis=np.zeros_like(As)
        maxFreq=np.zeros([N,n_freq])
        indices=np.zeros([N,n_freq],dtype=int)
           
        for   k,[a,b] in enumerate(zip(i_st,i_sp)):
            i=int(k/n_freq)
            j=int(k%n_freq)
            # print(i,j)
            a+=int(dT_raise*SPS)
            # print([a,b])
            # Tx
            k=0
            if findFreq:
                f,A,phase,Q,I= maxCorr_optimize(chunk2[f'ch{i_Tx:d}'].iloc[a:b],df=0.05,n=101,plot=False,flowpass=flowpass,f0=freqs[j])
            else:
                f=freqs[j]
                A,phase,Q,I=getACorr(chunk2[f'ch{i_Tx:d}'].iloc[a:b],f,SPS,phase=0,flowpass=100,lims=[])
                
            As[i,j,k]=A
            Qs[i,j,k]=Q
            Is[i,j,k]=I
            phis[i,j,k]=phase
            maxFreq[i,j]=f
            indices[i,j]=chunk2.index[int((b+a)/2)]
            
            for r,Rx in enumerate(Rx_ch):
                k=r+1
                
                # lock_in one block of data
                A,phi,Q,I=getACorr(chunk2[Rx].iloc[a:b],f,SPS,phase=0,flowpass=100,lims=[])
                As[i,j,k]=A
                Qs[i,j,k]=Q
                Is[i,j,k]=I
                phis[i,j,k]=phi
        time_LockIn+= time.perf_counter()-start_time 
        # print('execution time for computing correlation: {:.5f} s'.format(time_LockIn)) 

        

        

        try: 
            i_rest=chunk2.index[b]
        except Exception as e:
            try:
                print('Processing chunk number: {:d}'.format(i_chunk))
                print('b: {:d}'.format(b))
                print('len(chunk2): {:d}, start:{:d}, stop:{:d} '.format(len(chunk2),chunk2.index[0],chunk2.index[-1]))
                print(i_st)
                print(i_sp)
                i_rest=chunk2.index[-1]
                print('i_rest: {:d} '.format(i_rest))
                print(e)
            except Exception as e:
                print('Processing chunk number: {:d}'.format(i_chunk))
                print('b: {}'.format(b))
                print('len(chunk2): {:d}, start:{:d}, stop:{:d} '.format(len(chunk2),chunk2.index[0],chunk2.index[-1]))
                print(i_st)
                print(i_sp)
                i_rest=chunk2.index[-1]
                print('i_rest: {:d} '.format(i_rest))
                print(e)
            
        chunkrest=chunk2.loc[i_rest:]+int(dT_quite*SPS)
        start_ind=np.append(start_ind, chunk2.index[i_st])
        
        
        
        # add data to datamean
        data=pd.DataFrame(data=As[:,:,0],index=indices[:,0], columns=[f'A_Tx_f{k:d}' for k in range(1,len(freqs)+1)])
        data[[f'Q_ch{i_Tx:d}_f{k:d}' for k in range(1,len(freqs)+1)]]=Qs[:,:,0]
        data[[f'I_ch{i_Tx:d}_f{k:d}' for k in range(1,len(freqs)+1)]]=Is[:,:,0]
        data[[f'phase_Tx_f{k:d}' for k in range(1,len(freqs)+1)]]=phis[:,:,0]
        data[[f'i_f{k:d}' for k in range(1,len(freqs)+1)]]=indices
        data[[f'f{k:d}' for k in range(1,len(freqs)+1)]]= maxFreq
        
        for r,Rx in enumerate(Rx_ch):
            r2=r+1
            data[[f'Q_{Rx:s}_f{k:d}' for k in range(1,len(freqs)+1)]]=Qs[:,:,r2]
            data[[f'I_{Rx:s}_f{k:d}' for k in range(1,len(freqs)+1)]]=Is[:,:,r2]
            data[[f'A_Rx{r2:d}_f{k:d}' for k in range(1,len(freqs)+1)]]=As[:,:,r2]
            data[[f'phase_Rx{r2:d}_f{k:d}' for k in range(1,len(freqs)+1)]]=phis[:,:,r2]
            
        if len(datamean)==0:
            datamean=data
        else:
            datamean=pd.concat([datamean,data])
            
        # if i_chunk==2:
        #     break


    
    toc = time.perf_counter()
    print(f"Total time for Loading and processing data: {toc - tic:0.1f} seconds") 
    print("LockIn processing in {:0.3f} seconds".format(time_LockIn)) 
    print("Getting indexes in {:0.3f} seconds".format(time_indexes)) 

    return datamean, i_missing, gap,start_ind,freqs


#%% Plots for quality check of processing data


def plot_QandI(datamean,params,Rx_ch,MultiFreq,title=''):

    
    if MultiFreq:
        for i,ch in enumerate(Rx_ch):
            j=i+1
            pl.figure()
            n_freqs=len(params['freqs'])
            for k,f in enumerate(params['freqs']):
                ax=pl.subplot(n_freqs,1,k+1)
            
                ax.plot(datamean.index/SPS,datamean[f'Q_Rx{j:d}_f{k+1:d}'],'x',label='Q Rx')
                ax.plot(datamean.index/SPS,datamean[f'I_Rx{j:d}_f{k+1:d}'],'x',label='I Rx')
                ax.set_xlim(3,datamean.index[-1]/SPS-2)
                ax.plot(ax.get_xlim(),[0,0],'k--',)
                ax.set_ylabel('amplitude (-)')
                ax.set_xlabel('time (s)')
                ax.legend()
                pl.title(f'{ch:s}, f{k+1:d}:{f:.1f}Hz')
    else:
        #cut first 3 second and last second
        
        a=datamean.index.searchsorted(3*SPS)
        b=datamean.index.searchsorted(1*SPS)
        
        for i,ch in enumerate(Rx_ch):
            j=i+1
            pl.figure()
            pl.plot(datamean.index[a:-b]/SPS,datamean[f'Q_Rx{j:d}'].values[a:-b],'x',label='Q Rx')
            pl.plot(datamean.index[a:-b]/SPS,datamean[f'I_Rx{j:d}'].values[a:-b],'x',label='I Rx')
            pl.xlim(3,datamean.index[-b]/SPS)
            # pl.plot(pl.gca().get_xlim(),[0,0],'k--',)
            pl.ylabel('amplitude (-)')
            pl.xlabel('time (s)')
            pl.legend()
            if len(title)>0:
                pl.title(title)
            elif len(Rx_ch)>1:
                pl.title('f{:.1f}Hz, ch{:s}'.format(params['f'],ch))
            else:
                pl.title('f{:.1f}Hz'.format(params['f']))




def plot_QIandH(datamean,params,title='',log=False,xlim=[0.1,1]):
    
    fig,[ax,ax2]=pl.subplots(2,1,sharex=True)
    
    if len(title)>0:
        ax.set_title(title)
    else:
        ax.set_title('name:{:s}, f={:.2f} kHz'.format(params['name'],params['f']))
        
        
        
    ax.plot(datamean.time,datamean.Q_Rx1,'x')
    ax1=ax.twinx()
    ax1b=ax.twinx()
    ax1b.spines.right.set_position(("axes", 1.2))
    
    ax1.plot(datamean.time,datamean.h_Laser,'--k')
    ax1b.plot(datamean.time,datamean.h_GPS-datamean.h_Laser,'--b')
    ax.set_ylabel('amplitude Q (ppt)')
    ax.set_xlabel('time (s)')
    ax1.set_ylabel('h Laser (m)')
    ax1b.set_ylabel('h Laser - h GPS (m)')
    ax1.yaxis.label.set_color('k')
    ax1b.yaxis.label.set_color('b')
    
    
    ax.set_ylim(xlim)
    if log:
        ax.set_yscale('log')
    
    
    ax3=ax2.twinx()
    ax3b=ax2.twinx()
    ax3b.spines.right.set_position(("axes", 1.2))
    
    ax2.plot(datamean.time,datamean.I_Rx1,'x')
    ax3.plot(datamean.time,datamean.h_Laser,'--k')
    ax3b.plot(datamean.time,datamean.h_GPS-datamean.h_Laser,'--b')
    ax2.set_ylabel('amplitude I(ppt)')
    ax2.set_xlabel('time (s)')
    ax3.set_ylabel('h Laser (m)')
    ax3b.set_ylabel('h Laser - h GPS (m)')
    ax3.yaxis.label.set_color('k')
    ax3b.yaxis.label.set_color('b')
    ax2.set_ylim(-0.04,1)
    pl.tight_layout()
    
    return fig

#%% Code for inverting data and fit to climbs 



def Invert_data(datamean,params,
               w_cond=2408,d_coils=0,
               plot=True,method='L-BFGS-B'):

    """
    

    Parameters
    ----------
    datamean : pandas.Dataframe
        Processed LEM data.
    params : dictonary
        Parameters for processed LEM data.
    w_cond : TYPE, optional
        DESCRIPTION. The default is 2408.
    d_coils : TYPE, optional
        DESCRIPTION. The default is 0.
    plot : TYPE, optional
        DESCRIPTION. The default is True.
    method : TYPE, optional
        DESCRIPTION. The default is 'L-BFGS-B'.

    Returns
    -------
    None.

    """
    
    freq=params['f']
    
    if d_coils==0:
        try:
            d_coils=params['d_Rx']
        except KeyError():
            print('Using default coil distance =1.92')
            d_coils=1.92

    df=pd.DataFrame( datamean['Q_Rx1_corr'].values,
                    columns=['HCP{:0.3f}f{:0.1f}h0_quad'.format(d_coils,freq)])
    df2=pd.DataFrame( datamean['I_Rx1_corr'].values,
                     columns=['HCP{:0.3f}f{:0.1f}h0_inph'.format(d_coils,freq)])
    df3=pd.DataFrame( datamean[['Q_Rx1_corr','I_Rx1_corr']].values,
                     columns=['HCP{:0.3f}f{:0.1f}h0_quad'.format(d_coils,freq),
                              'HCP{:0.3f}f{:0.1f}h0_inph'.format(d_coils,freq)])
    
    
     #'L-BFGS-B'  #'ROPE'
    
    
    k= Problem()
    k2= Problem()
    k3= Problem()
    k.createSurvey(df,unit='ppt')
    k2.createSurvey(df2,unit='ppt')
    k3.createSurvey(df3,unit='ppt')
    k.surveys[0].name='Q'
    k2.surveys[0].name='I'
    k3.surveys[0].name='Q+I'
    
    
    k.setInit(depths0=[5], fixedDepths=[False],
              conds0=[0,w_cond], fixedConds=[True, False]) # set initial values
    k2.setInit(depths0=[5], fixedDepths=[False],
              conds0=[0,w_cond], fixedConds=[True, False]) # set initial values
    k3.setInit(depths0=[5], fixedDepths=[False],
              conds0=[0,w_cond], fixedConds=[True, False]) # set initial values
    
    k.invert(forwardModel='Q', method=method, regularization='l2', alpha=0.00,beta=0,
             bnds=[(0.1,60),(w_cond-20,w_cond+20)], rep=500, njobs=-1,relativeMisfit=True)# figure
    k2.invert(forwardModel='I', method=method, regularization='l2', alpha=0.00,beta=0,
             bnds=[(0.1,60),(w_cond-20,w_cond+20)], rep=500, njobs=-1,relativeMisfit=True)# figure
    k3.invert(forwardModel='QP', method=method, regularization='l2', alpha=0.00,beta=0,
             bnds=[(0.1,60),(w_cond-20,w_cond+20)], rep=500, njobs=-1,relativeMisfit=True)# figure
    
    
    
    datamean['hw_invQ']=k.depths[0]
    datamean['hw_invI']=k2.depths[0]
    datamean['hw_invQI']=k3.depths[0]
    
    
    datamean['h_totQ']=datamean['hw_invQ']-datamean.h_Laser
    datamean['h_totI']=datamean['hw_invI']-datamean.h_Laser
    datamean['h_totQI']=datamean['hw_invQI']-datamean.h_Laser



def Fit_climbs(datamean,params,
               t_str,t_stp, h_tot,w_depth=[],shallow=True,
               w_cond=2408,d_coils=0,
               plot=True):
    """
    Fit climbs to physical model (EMagPy)
    
    
    Parameters
    ----------
    datamean : pandas.Dataframe
        Processed LEM data.
    params : dictonary
        Parameters for processed LEM data.
    t_str : list,array
        start times for climbs (up or down)
    t_stp : list,array
        stop times for climbs (up or down)
    h_tot : list,array
        total thickness (ice+snow) at climbs location
    w_depth : TYPE, optional
        water ddepth (ice bottom to sea floor) at climbs location. 
        Needed only if Shallow=True The default is [].
    shallow : TYPE, optional
        if True use 3-layers model for shallow water. The default is True.
    w_cond : TYPE, optional
        Conductivity of wea water. The default is 2408.
    d_coils : TYPE, optional
        distance of coils. The default is 0. If 0 try to read coil distance from params. 
        If not found uses default 1.92
    plot : TYPE, optional
        Plot figures?. The default is True.

    Returns
    -------
    None.

    """
    
    if len(t_str)!=len(t_stp) or len(t_str)!=len(h_tot) or (shallow and len(t_str)!=len(w_depth)):
        print('t_str,t_stp, h_tot,w_depth needs to have same length. Stopping fit of climbs !!!')
        return None
    
    
    freq=params['f']
    
    if d_coils==0:
        try:
            d_coils=params['d_Rx']
        except KeyError():
            d_coils=1.92
            
    
    
    # get data climbs
    lims=[]    
    for i,[st,stp] in enumerate(zip(t_str,t_stp)):
        lims_i=[np.searchsorted(datamean.time.values,st),np.searchsorted(datamean.time.values,stp)]
        lims.append(lims_i)
    lims2=np.array([datamean.index[np.array(lims)[:,0]],datamean.index[np.array(lims)[:,1]]]).transpose()
    
    data_climbs=datamean.iloc[lims[0][0]:lims[0][1]].copy()
    for i in range(1,len(t_stp)):
        data_climbs=pd.concat([data_climbs,datamean.iloc[lims[i][0]:lims[i][1]].copy()])
        
    # add data ice from drillholes
    data_climbs['h_tot_ref']=data_climbs.h_Laser.values
    data_climbs['w_depth_ref']=0.0
    for i,l in enumerate(lims2):
        data_climbs.loc[lims2[i,0]:lims2[i,1],'h_tot_ref']+=h_tot[i]
        data_climbs.loc[lims2[i,0]:lims2[i,1],'w_depth_ref']=w_depth[i]
    data_climbs['w_depth_ref']+=data_climbs['h_tot_ref']
    
    
    # model climbs data
    if not shallow:
        emag=EMagPy_forwardmanualdata(data_climbs.h_Laser.values+np.mean(h_tot[i]),[freq],
                                       d_coils=d_coils,plot=plot,cond=w_cond)
    else:
        emag=EMagPy_forwardmanualdata_shallow(data_climbs['w_depth_ref'],data_climbs['h_tot_ref'],
                                               [freq],
                                               d_coils=d_coils,
                                               plot=plot,cond=[w_cond,0])

    data_climbs['Q_modeled']=emag['HCP{:0.3f}f{:0.1f}h0_quad'.format(d_coils,freq)].values
    data_climbs['I_modeled']=emag['HCP{:0.3f}f{:0.1f}h0_inph'.format(d_coils,freq)].values
    
    
    fitQ=linregress(data_climbs.Q_Rx1,data_climbs.Q_modeled)
    fitI=linregress(data_climbs.I_Rx1,data_climbs.I_modeled)
    
    params['fitQ_climbs']=fitQ
    params['fitI_climbs']=fitI
    
    data_climbs['Q_Rx1_corr']=data_climbs.Q_Rx1*fitQ.slope+fitQ.intercept
    data_climbs['I_Rx1_corr']=data_climbs.I_Rx1*fitI.slope+fitI.intercept
    
    datamean['Q_Rx1_corr']=datamean.Q_Rx1*fitQ.slope+fitQ.intercept
    datamean['I_Rx1_corr']=datamean.I_Rx1*fitI.slope+fitI.intercept

    
    if plot:
        
        data_climbs2=[]
        for i,l in enumerate(lims2):
            data_climbs2.append(data_climbs.loc[lims2[i,0]:lims2[i,1]].copy())
    
        
        fig,[ax,ax2]=pl.subplots(2,1)
        ax.set_title('f={:.2f} kHz, name:{:s}'.format(freq,params['name']))
        for i,d in enumerate(data_climbs2): 
            ax.plot(d.h_tot_ref,d.Q_Rx1,'--',label=f'i={i:d}')
            ax2.plot(d.h_tot_ref,d.I_Rx1,'--') 
        ax.set_ylabel('amplitude Q (ppt)')
        ax.set_xlabel('h water (m)')
        ax2.set_ylabel('amplitude I(ppt)')
        ax2.set_xlabel('h water (m)')
        ax.legend()
        ax2.legend()
        pl.tight_layout()
        
        
        fig,[ax,ax2]=pl.subplots(2,1)
        ax.set_title('f={:.2f} kHz, name:{:s}'.format(freq,params['name']))
        for i,d in enumerate(data_climbs2): 
            ax.plot(d.Q_Rx1,d.Q_modeled,'x',label=f'i={i:d}')
            ax2.plot(d.I_Rx1,d.I_modeled,'x',label=f'i={i:d}')
        xlim=np.array((0,data_climbs.Q_Rx1.max()))
        ax.plot(xlim,xlim*fitQ.slope+fitQ.intercept,'--k',label='fit')
        ax.set_ylabel('Q modeled (ppt)')
        ax.set_xlabel('Q LEM (ppt)')
        ax.legend()  
        xlim=np.array((0,data_climbs.I_Rx1.max()))
        ax2.plot(xlim,xlim*fitI.slope+fitI.intercept,'--k',label='fit')
        ax2.set_ylabel('I modeled (ppt)')
        ax2.set_xlabel('I LEM (ppt')
        ax2.legend()
        
        
        fig,[ax,ax2]=pl.subplots(2,1,sharex=True)
        ax.set_title('f={:.2f} kHz, name:{:s}'.format(freq,params['name']))
        ax.plot(data_climbs.time,data_climbs.Q_Rx1_corr,'x',label='LEM')
        ax.plot(data_climbs.time,data_climbs.Q_modeled,'--k',label='modeled')
        ax.set_ylabel('amplitude Q (ppt)')
        ax.set_xlabel('time (s)')
        ax2.plot(data_climbs.time,data_climbs.I_Rx1_corr,'x',label='LEM')
        ax2.plot(data_climbs.time,data_climbs.I_modeled,'--k',label='modeled')
        ax2.set_ylabel('amplitude I  (ppt)')
        ax2.set_xlabel('time (s)')

    return data_climbs



def Fit_climbs_emp(datamean,params,
                   t_str,t_stp, h_tot,  plot=True,h_lim=15):
    """
    Fit climbs to emphirical model

    Parameters
    ----------
    datamean : pandas.Dataframe
        Processed LEM data.
    params : dictonary
        Parameters for processed LEM data.
    t_str : list,array
        start times for climbs (up or down)
    t_stp : list,array
        stop times for climbs (up or down)
    h_tot : list,array
        total thickness (ice+snow) at climbs location
    plot : TYPE, optional
        Plot figures?. The default is True.
    h_lim : Float, optional
            Maximum distance to water to be used for fit. The default is 15 m.

    Returns
    -------
    None.

    """
    
    if len(t_str)!=len(t_stp) or len(t_str)!=len(h_tot) :
        print('t_str,t_stp, h_tot,w_depth needs to have same length. Stopping fit of climbs !!!')
        return None
    
    
    freq=params['f']
    
    
    
    # get data climbs
    lims=[]    
    for i,[st,stp] in enumerate(zip(t_str,t_stp)):
        lims_i=[np.searchsorted(datamean.time.values,st),np.searchsorted(datamean.time.values,stp)]
        lims.append(lims_i)
    lims2=np.array([datamean.index[np.array(lims)[:,0]],datamean.index[np.array(lims)[:,1]]]).transpose()
    
    data_climbs=datamean.iloc[lims[0][0]:lims[0][1]].copy()
    for i in range(1,len(t_stp)):
        data_climbs=pd.concat([data_climbs,datamean.iloc[lims[i][0]:lims[i][1]].copy()])
        
    # add data ice from drillholes
    data_climbs['h_tot_ref']=data_climbs.h_Laser.values
    
    for i,l in enumerate(lims2):
        data_climbs.loc[lims2[i,0]:lims2[i,1],'h_tot_ref']+=h_tot[i]
    
    
    # fit data
    def func(x, a, b,c):
        return a*np.log(x+b)+c

    # fitQ=optimize.curve_fit(func,  data_climbs.Q_Rx1, data_climbs['h_tot_ref'],[-3.5,0,-3.2])
    # fitI=optimize.curve_fit(func,  data_climbs.Q_Rx1, data_climbs['h_tot_ref'],[1e3,-0.01,-250])
    
    d=data_climbs.query(f'h_tot_ref < {h_lim:.2f}')[['Q_Rx1','I_Rx1','h_tot_ref']]
    
    ind=~np.isnan(np.log(d.Q_Rx1).values)
    fitQ=linregress(np.log(d.Q_Rx1)[ind],d['h_tot_ref'][ind])
    ind=~np.isnan(np.log(d.I_Rx1).values)
    fitI=linregress(np.log(d.I_Rx1)[ind],d['h_tot_ref'][ind])
    
    
    params['fitQ_climbs_emp']=fitQ
    params['fitI_climbs_emp']=fitI
    
    data_climbs['h_water_empQ']=np.log(data_climbs.Q_Rx1)*fitQ.slope+fitQ.intercept
    data_climbs['h_water_empI']=np.log(data_climbs.I_Rx1)*fitI.slope+fitI.intercept
    
    datamean['h_water_empQ']=np.log(datamean.Q_Rx1)*fitQ.slope+fitQ.intercept
    datamean['h_water_empI']=np.log(datamean.I_Rx1)*fitI.slope+fitI.intercept

    datamean['h_tot_empQ']=datamean.h_water_empQ-datamean.h_Laser
    datamean['h_tot_empI']=datamean.h_water_empI-datamean.h_Laser
    
    if plot:
        
        data_climbs2=[]
        for i,l in enumerate(lims2):
            data_climbs2.append(data_climbs.loc[lims2[i,0]:lims2[i,1]].copy())
    
        
        fig,[ax,ax2]=pl.subplots(2,1)
        ax.set_title('f={:.2f} kHz, name:{:s}'.format(freq,params['name']))
        for i,d in enumerate(data_climbs2): 
            ax.plot(d.h_tot_ref,np.log(d.Q_Rx1),'--',label=f'i={i:d}')
            ax2.plot(d.h_tot_ref,np.log(d.I_Rx1),'--') 
        
        ax.plot(d.h_tot_ref,(d.h_tot_ref-fitQ.intercept)/fitQ.slope,':k',label='fit')
        ax2.plot(d.h_tot_ref,(d.h_tot_ref-fitI.intercept)/fitI.slope,':k') 
        
        ax.set_ylabel('log( Q) (-)')
        ax.set_xlabel('h water (m)')
        ax2.set_ylabel('log(I) (ppt)')
        ax2.set_xlabel('h water (m)')
        ax.legend()
        ax2.legend()
        pl.tight_layout()
    
    
        fig,[ax,ax2]=pl.subplots(2,1)
        ax.set_title('f={:.2f} kHz, name:{:s}'.format(freq,params['name']))
        for i,d in enumerate(data_climbs2): 
            ax.plot(d.h_tot_ref,d.Q_Rx1,'--',label=f'i={i:d}')
            ax2.plot(d.h_tot_ref,d.I_Rx1,'--') 
        ax.plot(d.h_tot_ref,np.exp((d.h_tot_ref-fitQ.intercept)/fitQ.slope),':k',label='fit')
        ax2.plot(d.h_tot_ref,np.exp((d.h_tot_ref-fitI.intercept)/fitI.slope),':k') 
            
        ax.set_ylabel('amplitude Q (ppt)')
        ax.set_xlabel('h water (m)')
        ax2.set_ylabel('amplitude I(ppt)')
        ax2.set_xlabel('h water (m)')
        ax.legend()
        ax2.legend()
        pl.tight_layout()
        
        
        fig,ax=pl.subplots(1,1)
        ax.set_title('f={:.2f} kHz, name:{:s}'.format(freq,params['name']))
        ax.plot(data_climbs.h_tot_ref,data_climbs.h_water_empQ,'x',label='h_Q')
        ax.plot(data_climbs.h_tot_ref,data_climbs.h_water_empI,'x',label='h_I')  
        ax.plot([0,30],[0,30],':k')
        ax.set_ylabel('h_water LEM emp  (m)')
        ax.set_xlabel('h_water Laser (m)')
        ax.legend()
        pl.tight_layout()
        
        
        
        # fig,[ax,ax2]=pl.subplots(2,1,sharex=True)
        # ax.set_title('f={:.2f} kHz, name:{:s}'.format(freq,params['name']))
        # ax.plot(data_climbs.time,data_climbs.Q_Rx1_corr,'x',label='LEM')
        # ax.plot(data_climbs.time,np.exp((data_climbs.h_tot_ref-fitQ.intercept)/fitQ.slope),'--k',label='modeled')
        # ax.set_ylabel('amplitude Q (ppt)')
        # ax.set_xlabel('time (s)')
        # ax2.plot(data_climbs.time,data_climbs.I_Rx1_corr,'x',label='LEM')
        # ax2.plot(data_climbs.time,np.exp((data_climbs.h_tot_ref-fitQ.intercept)/fitQ.slope),'--k',label='modeled')
        # ax2.set_ylabel('amplitude I  (ppt)')
        # ax2.set_xlabel('time (s)')

    return data_climbs



def Fit_climbs_emp_corr(datamean,params,
                   t_str,t_stp, h_tot,  plot=True,h_lim=15):
    """
    

    Parameters
    ----------
    datamean : pandas.Dataframe
        Processed LEM data.
    params : dictonary
        Parameters for processed LEM data.
    t_str : list,array
        start times for climbs (up or down)
    t_stp : list,array
        stop times for climbs (up or down)
    h_tot : list,array
        total thickness (ice+snow) at climbs location
    plot : TYPE, optional
        Plot figures?. The default is True.
    h_lim : Float, optional
            Maximum distance to water to be used for fit. The default is 15 m.

    Returns
    -------
    None.

    """
    
    if len(t_str)!=len(t_stp) or len(t_str)!=len(h_tot) :
        print('t_str,t_stp, h_tot,w_depth needs to have same length. Stopping fit of climbs !!!')
        return None
    
    
    freq=params['f']
    
    
    
    # get data climbs
    lims=[]    
    for i,[st,stp] in enumerate(zip(t_str,t_stp)):
        lims_i=[np.searchsorted(datamean.time.values,st),np.searchsorted(datamean.time.values,stp)]
        lims.append(lims_i)
    lims2=np.array([datamean.index[np.array(lims)[:,0]],datamean.index[np.array(lims)[:,1]]]).transpose()
    
    data_climbs=datamean.iloc[lims[0][0]:lims[0][1]].copy()
    for i in range(1,len(t_stp)):
        data_climbs=pd.concat([data_climbs,datamean.iloc[lims[i][0]:lims[i][1]].copy()])
        
    # add data ice from drillholes
    data_climbs['h_tot_ref']=data_climbs.h_Laser.values
    
    for i,l in enumerate(lims2):
        data_climbs.loc[lims2[i,0]:lims2[i,1],'h_tot_ref']+=h_tot[i]
    
    
    # fit data
    def func(x, a, b,c):
        return a*np.log(x+b)+c

    # fitQ=optimize.curve_fit(func,  data_climbs.Q_Rx1_corr, data_climbs['h_tot_ref'],[-3.5,0,-3.2])
    # fitI=optimize.curve_fit(func,  data_climbs.I_Rx1_corr, data_climbs['h_tot_ref'],[1e3,-0.01,-250])
    
    d=data_climbs.query(f'h_tot_ref < {h_lim:.2f}')[['Q_Rx1_corr','I_Rx1_corr','h_tot_ref']]
    
    ind=~np.isnan(np.log(d.Q_Rx1_corr).values)
    fitQ=linregress(np.log(d.Q_Rx1_corr)[ind],d['h_tot_ref'][ind])
    ind=~np.isnan(np.log(d.I_Rx1_corr).values)
    fitI=linregress(np.log(d.I_Rx1_corr)[ind],d['h_tot_ref'][ind])
    
    
    params['fitQ_climbs_emp_corr']=fitQ
    params['fitI_climbs_emp_corr']=fitI
    
    data_climbs['h_water_empQ']=np.log(data_climbs.Q_Rx1_corr)*fitQ.slope+fitQ.intercept
    data_climbs['h_water_empI']=np.log(data_climbs.I_Rx1_corr)*fitI.slope+fitI.intercept
    
    datamean['h_water_empQ']=np.log(datamean.Q_Rx1_corr)*fitQ.slope+fitQ.intercept
    datamean['h_water_empI']=np.log(datamean.I_Rx1_corr)*fitI.slope+fitI.intercept

    datamean['h_tot_empQ']=datamean.h_water_empQ-datamean.h_Laser
    datamean['h_tot_empI']=datamean.h_water_empI-datamean.h_Laser
    
    if plot:
        
        data_climbs2=[]
        for i,l in enumerate(lims2):
            data_climbs2.append(data_climbs.loc[lims2[i,0]:lims2[i,1]].copy())
    
        
        fig,[ax,ax2]=pl.subplots(2,1)
        ax.set_title('f={:.2f} kHz, name:{:s}'.format(freq,params['name']))
        for i,d in enumerate(data_climbs2): 
            ax.plot(d.h_tot_ref,np.log(d.Q_Rx1_corr),'--',label=f'i={i:d}')
            ax2.plot(d.h_tot_ref,np.log(d.I_Rx1_corr),'--') 
        
        ax.plot(d.h_tot_ref,(d.h_tot_ref-fitQ.intercept)/fitQ.slope,':k',label='fit')
        ax2.plot(d.h_tot_ref,(d.h_tot_ref-fitI.intercept)/fitI.slope,':k') 
        
        ax.set_ylabel('log( Q) (-)')
        ax.set_xlabel('h water (m)')
        ax2.set_ylabel('log(I) (ppt)')
        ax2.set_xlabel('h water (m)')
        ax.legend()
        ax2.legend()
        pl.tight_layout()
    
    
        fig,[ax,ax2]=pl.subplots(2,1)
        ax.set_title('f={:.2f} kHz, name:{:s}'.format(freq,params['name']))
        for i,d in enumerate(data_climbs2): 
            ax.plot(d.h_tot_ref,d.Q_Rx1_corr,'--',label=f'i={i:d}')
            ax2.plot(d.h_tot_ref,d.I_Rx1_corr,'--') 
        ax.plot(d.h_tot_ref,np.exp((d.h_tot_ref-fitQ.intercept)/fitQ.slope),':k',label='fit')
        ax2.plot(d.h_tot_ref,np.exp((d.h_tot_ref-fitI.intercept)/fitI.slope),':k') 
            
        ax.set_ylabel('amplitude Q (ppt)')
        ax.set_xlabel('h water (m)')
        ax2.set_ylabel('amplitude I(ppt)')
        ax2.set_xlabel('h water (m)')
        ax.legend()
        ax2.legend()
        pl.tight_layout()
        
        
        fig,ax=pl.subplots(1,1)
        ax.set_title('f={:.2f} kHz, name:{:s}'.format(freq,params['name']))
        ax.plot(data_climbs.h_tot_ref,data_climbs.h_water_empQ,'x',label='h_Q')
        ax.plot(data_climbs.h_tot_ref,data_climbs.h_water_empI,'x',label='h_I')  
        ax.plot([0,30],[0,30],':k')
        ax.set_ylabel('h_water LEM emp  (m)')
        ax.set_xlabel('h_water Laser (m)')
        ax.legend()
        pl.tight_layout()

    return data_climbs





def EMagPy_forwardmanualdata(h_water,freqs,d_coils=1.929,plot=True,cond=2400):

    # parameters for the synthetic model
    nlayer = 2 # number of layers
    depths =h_water[:,None] 
    npos = len(depths) # number of positions/sampling locations
    
    cond=np.array(cond)
    cond[((cond>0)==False)]=0
    
    conds = np.zeros((npos, nlayer))
    conds[:,1]= cond 
    
    
    # defines coils configuration, frequency
    coils = ['HCP{:0.3f}f{:0.1f}h0'.format(d_coils,f) for f in freqs] 
    
    # foward modelling
    k = Problem()
    k.setModels([depths], [conds])
    _ = k.forward(forwardModel='QP', coils=coils, noise=0.0)
    
    if plot:
        k.showResults() # display original model
    # k.show() # display ECa computed from forward modelling
    
    return k.surveys[0].df


def EMagPy_forwardmanualdata_shallow(water_depth,h_water,freqs,d_coils=1.929,plot=True,cond=[0,2400]):

    # parameters for the synthetic model
    nlayer = 3 # number of layers
    depths =np.array([h_water,water_depth]).transpose()
    npos = depths.shape[0] # number of positions/sampling locations
    
    cond=np.array(cond)
    cond[((cond>0)==False)]=0
    
    conds = np.zeros((npos, nlayer))
    conds[:,1:]= cond 
    
    
    # defines coils configuration, frequency
    coils = ['HCP{:0.3f}f{:0.1f}h0'.format(d_coils,f) for f in freqs] 
    
    # foward modelling
    k = Problem()
    k.setModels([depths], [conds])
    _ = k.forward(forwardModel='QP', coils=coils, noise=0.0)
    
    if plot:
        k.showResults() # display original model
    # k.show() # display ECa computed from forward modelling
    
    return k.surveys[0].df



def EMagPy_forwardmanualdata_old(depths,freqs,d_coils=1.929,plot=True):

    # parameters for the synthetic model
    nlayer = 1 # number of layers
    depths =depths[:,None] 
    npos = len(depths) # number of positions/sampling locations
    conds = np.ones((npos, nlayer))*[0, 2400] # EC in mS/m
    # depth of model 
    depths2=depths[:,0]
    
    # defines coils configuration, frequency
    coils = ['HCP{:0.3f}f{:0.1f}h0'.format(d_coils,f) for f in freqs] 
    
    # foward modelling
    
    k = Problem()
    k.setModels([depths], [conds])
    _ = k.forward(forwardModel='QP', coils=coils, noise=0.0)
    
    if plot:
        k.showResults() # display original model
    # k.show() # display ECa computed from forward modelling
    
    return k.surveys[0].df



#%% Code for comparing data along transect


def interpolData_d(data,datamean,proplist):
    """
    Function to interpolate Drillholes data to LEM data along a tansect. 
    Data are interpolated according to distance and inserted as columns in datamean pandas DataFrame.
    
    Parameters
    ----------
    data : Pandas Dataframe, 
        E.G: Drillholes data .
    datamean : Pandas Dataframe
        LEM datamean data.
    proplist : list of strings
        List of parameters to intepolate.

    Returns
    -------
    None.

    """
    d1=np.array(data.d)
    d2=np.array(datamean.d)
    
    ind=np.searchsorted(d1,d2)
    
    ind[ind>len(d1)-1]=len(d1)-1
    ind[ind==0]=1
    d_dist=(d2-d1[ind])/(d1[ind]-d1[ind-1])
    
    for p in proplist:
        x=np.array(getattr(data,p))
        datamean.loc[:,p]=x[ind]+(x[ind]-x[ind-1])*d_dist

#%% Code for plotting on map

def plot_xy(datamean,attr,origin=0,colorlim=[]):
    
    x,y=get_XY(datamean,origin=origin)
    
    
    
    fig,ax=pl.subplots(1,1)
    
    if len(colorlim)>0:
        im=ax.scatter(x,y,c=datamean[attr],cmap=cm.batlow,marker='x',vmin=colorlim[0],vmax=colorlim[1])
    else:
        im=ax.scatter(x,y,c=datamean[attr],cmap=cm.batlow,marker='x')
    c=pl.colorbar(im, label=attr)
    ax.set_ylabel('y (m)')
    ax.set_xlabel('x (m)')
    pl.tight_layout()

    return fig

def get_XY(datamean,origin=0):  
    """
    Transform lat,lon to local coordinates around origin point

    Parameters
    ----------
    point : TYPE
        x,y coordinates to point.
    origin : TYPE
        if int: index of coordinate to se as origin of local coordinate system
        if: [lat,lon] coordinates of origin of local reference system.

    Returns
    -------
    x : local x-coordinate in meters (west-east)
    y : local y-coordinate in meters (south- north)

    """
    
    origin=np.array(origin)
    if len(origin.shape)==0:  
        p0=[datamean.lat.values[origin],datamean.lon.values[origin]]
    elif len(origin.shape)==1:  
            p0=origin    
    
    d=[]
    bearing=[]
    for lat,lon in zip(datamean.lat,datamean.lon):
        d.append(distance.distance(p0,[lat,lon]).m)
        bearing.append(get_bearing(p0,[lat,lon]))
    
    d=np.array(d)
    bearing=np.array(bearing)
    
    y=d*np.cos(bearing/360*2*np.pi)
    x=d*np.sin(bearing/360*2*np.pi)
    return x,y

def get_bearing(start, end):
    # if (type(start) != tuple) or (type(end) != tuple):
    #     raise TypeError("Only tuples are supported as arguments")

    lat1 = math.radians(start[0])
    lat2 = math.radians(end[0])

    diffLong = math.radians(end[1] - start[1])

    x = math.sin(diffLong) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1)
                                           * math.cos(lat2) * math.cos(diffLong))

    initial_bearing = math.atan2(x, y)

    # Now we have the initial bearing but math.atan2 return values
    # from -180° to + 180° which is not what we want for a compass bearing
    # The solution is to normalize the initial bearing as shown below
    initial_bearing = math.degrees(initial_bearing)
    compass_bearing = (initial_bearing + 360) % 360

    return compass_bearing

#%%  Sync INS/Laser and ADC

def sync_ADC_INS(datamean,dataINS,iStart=2,dT_start=0):
    
    try:
        TOW0=get_TOW0(dataINS,iStart=iStart)
        
        TOW=datamean.index/SPS+TOW0+dT_start
        print('TOW0 ADC: {:f}, dT_start: {:f}'.format(TOW0,dT_start))
        
        
        try:
            leapS=dataINS.PGPSP.leapS[0]
        except:
            leapS=18
        
        t = gps_datetime_np(dataINS.PINS1.GPSWeek[0],TOW,leap_seconds=leapS)
        datamean['t']=t
        datamean['TOW']=TOW
        datamean['time']=datamean.TOW-TOW0
    
        # interpolate PINS1 data
        interpolData(dataINS.PINS1,datamean,['TOW','heading','velX','velY','velZ','lat','lon', 'elevation',])
        datamean.rename(columns={'elevation':'h_GPS'}, inplace=True)
        interpolData(dataINS.Laser,datamean,['h_corr', 'roll', 'pitch','signQ', 'T'])
        datamean.rename(columns={"T": "TempLaser",'h_corr':'h_Laser'}, inplace=True)
        
        datamean['diff_hGPSLaser']=datamean.h_GPS-datamean.h_Laser
        
    except AttributeError as error: 
         print(error)
         print("Can't find right INS data. Continue without them.")
         datamean['TOW']=datamean.index/SPS
         TOW0=0
    return TOW0
        
def interpolData(data,datamean,proplist):
    ind=np.searchsorted(data.TOW,datamean.TOW)
    ind[ind>len(data.TOW)-1]=len(data.TOW)-1
    ind[ind==0]=1
    dt=(datamean.TOW-data.TOW[ind])/(data.TOW[ind]-data.TOW[ind-1])
    
    for p in proplist:
        x=getattr(data,p)
        datamean[p]=x[ind]+(x[ind]-x[ind-1])*dt
    


gps_datetime_np=np.vectorize(gps_datetime, doc='Vectorized `gps_datetime`')






#%%  import UAV_GPX data and add to datamean


def load_GPS_UAV(uav_path,datamean,dT=0,lim_syncGPSLaser=[]):
    """
    

    Parameters
    ----------
    uav_path : TYPE
        path data csv.
    datamean : TYPE
        DESCRIPTION.
    dT : TYPE, optional
        time correction. The default is 0.
    lim_syncGPSLaser : TYPE, List
        Limits of time to use for align h_Laser and h_UAV. The default is [].

    Returns
    -------
    UAV : TYPE
        DESCRIPTION.

    """
    
    UAV= pd.read_csv(uav_path)
    
    # UAV=UAV[UAV.time > '2024-04-17 21:52:45']
    # UAV=UAV[UAV.time < '2024-04-17 22:10:45']
    UAV['time']=[datamean.t.iloc[0]+timedelta(milliseconds=UAV['timestamp(ms)'][i]) for i in range(len(UAV))]

    # merge data
    datamean['elevation_UAV']=mergeByTime(UAV.time,UAV['GPS_RAW_INT.alt']/1000,datamean.t, method='linInterpol', maxDelta=0.5,dT=dT)
    datamean['lat_UAV']=mergeByTime(UAV.time,UAV['GPS_RAW_INT.lat'],datamean.t, method='linInterpol', maxDelta=0.5,dT=dT)
    datamean['lon_UAV']=mergeByTime(UAV.time,UAV['GPS_RAW_INT.lon'],datamean.t, method='linInterpol', maxDelta=0.5,dT=dT)

    if len(lim_syncGPSLaser):
        # alline Laser and GPS
        ind=(datamean.time >lim_syncGPSLaser[0]) &( datamean.time <lim_syncGPSLaser[1])
        datamean['h_UAV']=datamean['elevation_UAV']-datamean['elevation_UAV'][ind].mean()+datamean['h_Laser'][ind].mean()
    else:
        ind=19200
        datamean['h_UAV']=datamean['elevation_UAV']-datamean['elevation_UAV'][:ind].mean()+datamean['h_Laser'][:ind].mean()
    
    return UAV



def load_GPX(gpx_path,datamean,dT=0,lim_syncGPSLaser=[]):
    """
    

    Parameters
    ----------
    gpx_path : TYPE
        path gpx data.
    datamean : TYPE
        DESCRIPTION.
    dT : TYPE, optional
        time correction. The default is 0.
    lim_syncGPSLaser : TYPE, List
        Limits of time to use for align h_Laser and h_UAV. The default is [].

    Returns
    -------
    UAV : TYPE
        DESCRIPTION.

    """
    
    UAV= GPXToPandas(gpx_path)
    
    # UAV=UAV[UAV.time > '2024-04-17 21:52:45']
    # UAV=UAV[UAV.time < '2024-04-17 22:10:45']
    

    # merge data
    datamean['elevation_UAV']=mergeByTime(UAV.time,UAV['elevation'],datamean.t, method='linInterpol', maxDelta=0.5,dT=dT)
    datamean['lat_UAV']=mergeByTime(UAV.time,UAV['lat'],datamean.t, method='linInterpol', maxDelta=0.5,dT=dT)
    datamean['lon_UAV']=mergeByTime(UAV.time,UAV['lon'],datamean.t, method='linInterpol', maxDelta=0.5,dT=dT)

    if len(lim_syncGPSLaser):
        # alline Laser and GPS
        ind=(datamean.time >lim_syncGPSLaser[0]) &( datamean.time <lim_syncGPSLaser[1])
        datamean['h_UAV']=datamean['elevation_UAV']-datamean['elevation_UAV'][ind].mean()+datamean['h_Laser'][ind].mean()
    else:
        ind=19200
        datamean['h_UAV']=datamean['elevation_UAV']-datamean['elevation_UAV'][:ind].mean()+datamean['h_Laser'][:ind].mean()
    
    return UAV


def fit_Laser_UAV_GPS(datamean,params,t_str=[],t_stp=[],slope=0):
    
    if len(t_str)==len(t_stp) and len(t_stp)>0:
        # get data climbs
        lims=[]    
        for i,[st,stp] in enumerate(zip(t_str,t_stp)):
            lims_i=[np.searchsorted(datamean.time.values,st),np.searchsorted(datamean.time.values,stp)]
            lims.append(lims_i)
        lims2=np.array([datamean.index[np.array(lims)[:,0]],datamean.index[np.array(lims)[:,1]]]).transpose()
        
        data_climbs=datamean.iloc[lims[0][0]:lims[0][1]].copy()
        for i in range(1,len(t_stp)):
            data=pd.concat([data_climbs,datamean.iloc[lims[i][0]:lims[i][1]].copy()])
    else:
        data=datamean
    
    
    if slope==0:
        fit=linregress(data['h_UAV'].values[np.isnan(data['h_UAV'].values)==False],
                       data['h_Laser'].values[np.isnan(data['h_UAV'].values)==False])
    
        pl.figure()
        pl.plot(data['h_UAV'],data['h_Laser'],'x')
        pl.plot([0,40],[0,40],'--k')
        pl.plot([0,40],np.array([0,40])*fit.slope+fit.intercept,'--g')
        pl.text(25,15,'y=x*{:.4f}+{:.4f}'.format(fit.slope,fit.intercept),color='g')
        pl.text(20,30,'1:1',color='k')
        pl.xlabel('h GPS (m)')
        pl.ylabel('h laser (m)')
     
        params['fit_GPS_UAV']=fit
        slope=fit.slope
    else:
        params['fit_GPS_UAV']=fit
    
   

    
    datamean['h_UAV_orig']=datamean['h_UAV'].values
    datamean['h_UAV']=(datamean['h_UAV'])*slope
    

    fig,ax=pl.subplots(1,1,sharex=True)
    ax.plot(datamean.time,datamean['h_UAV_orig'],label='GPS UAV original')
    ax.plot(datamean.time,datamean['h_UAV'],label='GPS UAV corrected')
    ax.plot(datamean.time,datamean['h_Laser'],label='Laser')
    ax.set_xlabel('time (s)')
    ax.set_ylabel('h (m)')
    ax.legend()

def plot_UAV_GPS(datamean):
    fig,ax=pl.subplots(1,1,sharex=True)
    ax.plot(datamean.time,datamean['elevation_UAV'],label='GPS elevation UAV')
    ax.plot(datamean.time,datamean['h_UAV'],label='h GPS UAV')
    ax.plot(datamean.time,datamean['h_GPS'],label='h GPS 2')
    ax.plot(datamean.time,datamean['h_Laser'],label='Laser')
    ax.set_xlabel('time (s)')
    ax.set_ylabel('h (m)')
    ax.legend()

#%% Code for signal to noise 



def signalStrength(datamean,params,l_period=5):
    l_period=5 # lenght of period after calibration for averaging 
    
    stop_t=params['CalParams']['stop']
    
    t_a=stop_t+0.5
    t_b=t_a+5
    
    i_a=datamean.time.searchsorted(t_a)
    i_b=datamean.time.searchsorted(t_b)
    
    
    values_I=[get_values(datamean['I_Rx1'],i_a,i_b)]
    values_Q=[get_values(datamean['Q_Rx1'],i_a,i_b)]
    
    
    for t_on,t_off in zip(params['CalParams']['on'],params['CalParams']['off']):
        i_a=datamean.time.searchsorted(t_on+0.1)
        i_b=datamean.time.searchsorted(t_off-0.1)
        values_I.append(get_values(datamean['I_Rx1'],i_a,i_b))
        values_Q.append(get_values(datamean['Q_Rx1'],i_a,i_b))
    
    values_I=np.array(values_I)
    values_I=np.array(values_I)
    values_Q=np.array(values_Q)
    values_Q=np.array(values_Q)
    
    ratios_Q=values_Q[0][2]/values_Q[1:][0]  
    ratios_I=values_I[0][2]/values_I[1:][0]    
    
    print('Signal strength:\n---------------')    
    print('I: Std: {:.6f}, max deviation: {:.6f}'.format(values_I[0][2],values_I[0][3]))   
    print('Q: Std: {:.6f}, max deviation: {:.6f}'.format(values_Q[0][2],values_Q[0][3]))  
    for i in range(1,len(params['CalParams']['on'])):
        print('Cal point {:d}'.format(i)) 
        print('\t Amplitude I: {:.6f} \nStd I Air/Amp. I: {:.6f}'.format(values_I[i][0],values_I[0][2]/values_I[i][0]))   
        print('\t Amplitude Q: {:.6f} \nStd Q Air/Amp. Q: {:.6f}'.format(values_Q[i][0],values_Q[0][2]/values_Q[i][0]))  
    
    return ratios_I,ratios_Q,values_I,values_Q 



    
# get 
def get_values(d,i_a,i_b):
    I_med=d.iloc[i_a:i_b].median()
    I_mean=d.iloc[i_a:i_b].mean()
    I_std=d.iloc[i_a:i_b].std()
    I_max=np.abs(d.iloc[i_a:i_b]-I_mean).max()
    return I_med,I_mean,I_std, I_max



#%% Code for exporting data 

def exportThim(datamean,params,): 
    datexp=datamean[['A_Rx1','phase_Rx1','Q_Rx1_corr','I_Rx1_corr','hw_invQ','h_water_empQ','h_water_empI','time','d', 'lat', 'lon',
    'h_Laser',]]
    datexp['Q_Rx']=datexp.A_Rx1*np.sin(datexp.phase_Rx1)
    datexp['I_Rx']=datexp.A_Rx1*np.cos(datexp.phase_Rx1)
    
    
    datexp.to_csv(pcpath+ r'\2024_Fieldwork\LEM\data\export\dataT'+filename+'.csv')
    
    f=open(pcpath+ r'\2024_Fieldwork\LEM\data\export\paramsT'+filename+'.csv','w' )
    
    f.write('file: '+filename)
    f.write('\nfreq: {:.1f} kHz'.format(params['f']))
    
    
    f.write('\n\nZ=g(I-I0 +i(Q-Q0)*exp(i*phi)')
    f.write('\n\tg: {:.2f} ppt'.format(params['CalParams']['g'][0]*params['fitQ_climbs'].slope))
    f.write('\n\tphi: {:.5f} rad'.format(params['CalParams']['phi'][0]))
    f.write('\n\tQ0: {:.5f}'.format(params['CalParams']['Q0'][0]))
    f.write('\n\tI0: {:.5f}'.format(params['CalParams']['I0'][0]))
    
    f.write('\n\nDistance form water derived from Quadrature')
    f.write('\nh_Q=a*ln(Q)+c')
    f.write('\n\ta: {:.2f} m'.format(params['fitQ_climbs_emp_corr'].slope))
    f.write('\n\tc: {:.5f} m'.format(params['fitQ_climbs_emp_corr'].intercept))
    
    f.write('\n\nDistance form water derived from Inphase')
    f.write('\nh_I=a*ln(I)+c')
    f.write('\n\ta: {:.2f} m'.format(params['fitI_climbs_emp_corr'].slope))
    f.write('\n\tc: {:.5f} m'.format(params['fitI_climbs_emp_corr'].intercept))
        
    f.close()


#%%  various functions


# def getCalConstants(datamean,caltimes,start,stop,off,on,ID):

def getCalTimes(dataINS,datamean,t_buf=0.2,t_int0=3,Rx_ch=['ch1']):
    
    try:
        TOW0=TOW_ADC0
    except NameError:
        TOW0=get_TOW0(dataINS)
    
    
    # find start and stop   
    start=dataINS.Cal.TOW[(dataINS.Cal.On==0) * (dataINS.Cal.ID==1)]-TOW0
    stop=dataINS.Cal.TOW[(dataINS.Cal.On==0) * (dataINS.Cal.ID==2)]-TOW0
    off=dataINS.Cal.TOW[(dataINS.Cal.On==0) * (dataINS.Cal.ID==0)]-TOW0
    on=dataINS.Cal.TOW[(dataINS.Cal.On==1)]-TOW0
    ID=dataINS.Cal.ID[(dataINS.Cal.On==1)]
    
    
    off1=[]
    on1=[]
    ID1=[]
    calQs=[]
    calIs=[]
    
    
    if len(start)==1 and len(stop)==1:
        print("One calibration point")
        
        off1=[off]
        on1=[on]
        ID1=[ID]
        
    elif len(start)>1 and len(stop)>1 :
        print("Multiple calibrations")

        for st,stp in zip(start,stop):
            inds=(on.searchsorted(st),off.searchsorted(stp))
            off1.append(off[inds[0]:inds[1]])
            on1.append(on[inds[0]:inds[1]])
            ID1.append(ID[inds[0]:inds[1]])

    else:
        print(" Can not find calibration starting point")
        return    start,stop,off1,on1,ID1, calQs,calIs
    
    # print(start,stop,off1,on1,ID1)
    
    for  st,stp,off2,on2 in zip(start,stop,off1,on1 ):
        Is=[]
        Qs=[]
        for i,ch in enumerate(Rx_ch):
            j=i+1
            lims=[datamean.time.searchsorted(st-t_int0),datamean.time.searchsorted(st-t_buf)]
            lims2=[datamean.time.searchsorted(stp+t_buf),datamean.time.searchsorted(stp+t_int0)]
             
            # Is.append([(datamean[f'I_Rx{j:d}'].iloc[lims[0]:lims[1]].mean() + datamean[f'I_Rx{j:d}'].iloc[lims2[0]:lims2[1]].mean() )/2])
            # Qs.append([(datamean[f'Q_Rx{j:d}'].iloc[lims[0]:lims[1]].mean() + datamean[f'Q_Rx{j:d}'].iloc[lims2[0]:lims2[1]].mean() )/2])
            Is.append([(datamean[f'I_Rx{j:d}'].iloc[lims[0]:lims[1]].median() + datamean[f'I_Rx{j:d}'].iloc[lims2[0]:lims2[1]].median() )/2])
            Qs.append([(datamean[f'Q_Rx{j:d}'].iloc[lims[0]:lims[1]].median() + datamean[f'Q_Rx{j:d}'].iloc[lims2[0]:lims2[1]].median() )/2])
            
            for a,b in zip(on2,off2):
                lims=[datamean.time.searchsorted(a+t_buf),datamean.time.searchsorted(b-t_buf)]
                # Is[i].append([datamean[f'I_Rx{j:d}'].iloc[lims[0]:lims[1]].mean()])
                # Qs[i].append([datamean[f'Q_Rx{j:d}'].iloc[lims[0]:lims[1]].mean()])
                Is[i].append(datamean[f'I_Rx{j:d}'].iloc[lims[0]:lims[1]].median())
                Qs[i].append(datamean[f'Q_Rx{j:d}'].iloc[lims[0]:lims[1]].median())

            
        calIs.append(Is)
        calQs.append(Qs)
            
    return start,stop,off1,on1,ID1, np.array(calQs),np.array(calIs)

def getCalTimes_multi(dataINS,datamean,t_buf=0.2,t_int0=3,Rx_ch=['ch1'],n_freqs=3):
    TOW0=get_TOW0(dataINS)
    # find start and stop   
    start=dataINS.Cal.TOW[(dataINS.Cal.On==0) * (dataINS.Cal.ID==1)]-TOW0
    stop=dataINS.Cal.TOW[(dataINS.Cal.On==0) * (dataINS.Cal.ID==2)]-TOW0
    off=dataINS.Cal.TOW[(dataINS.Cal.On==0) * (dataINS.Cal.ID==0)]-TOW0
    on=dataINS.Cal.TOW[(dataINS.Cal.On==1)]-TOW0
    ID=dataINS.Cal.ID[(dataINS.Cal.On==1)]
    
    
    off1=[]
    on1=[]
    ID1=[]
    calQs=[]
    calIs=[]
    
    
    if len(start)==1 and len(stop)==1:
        print("One calibration point")
        
        off1=[off]
        on1=[on]
        ID1=[ID]
        
    elif len(start)>1 and len(stop)>1 :
        print("Multiple calibrations")

        for st,stp in zip(start,stop):
            inds=(on.searchsorted(st),off.searchsorted(stp))
            off1.append(off[inds[0]:inds[1]])
            on1.append(on[inds[0]:inds[1]])
            ID1.append(ID[inds[0]:inds[1]])

    else:
        print(" Can not find calibration starting point")
        return    start,stop,off1,on1,ID1, calQs,calIs
    
    # print(start,stop,off1,on1,ID1)
    
    Is=np.zeros([len(start),len(Rx_ch),n_freqs,len(on1[0])+1])
    Qs=np.zeros_like(Is)
    
    for  r,[st,stp,off2,on2] in enumerate(zip(start,stop,off1,on1 )):
        
        for i,ch in enumerate(Rx_ch):
            j=i+1

            for k in range(1,n_freqs+1):
                lims=[datamean.time.searchsorted(st-t_int0),datamean.time.searchsorted(st-t_buf)]
                lims2=[datamean.time.searchsorted(stp+t_buf),datamean.time.searchsorted(stp+t_int0)]
                
                # print(r)
                # print(i)
                # print(k)
                
                
                # get median value for free space 
                Is[r,i,k-1,0]=(datamean[f'I_Rx{j:d}_f{k:d}'].iloc[lims[0]:lims[1]].median() + datamean[f'I_Rx{j:d}_f{k:d}'].iloc[lims2[0]:lims2[1]].median() )/2
                Qs[r,i,k-1,0]=(datamean[f'Q_Rx{j:d}_f{k:d}'].iloc[lims[0]:lims[1]].median() + datamean[f'Q_Rx{j:d}_f{k:d}'].iloc[lims2[0]:lims2[1]].median() )/2
                
                for z,[a,b] in enumerate(zip(on2,off2)):
                    lims=[datamean.time.searchsorted(a+t_buf),datamean.time.searchsorted(b-t_buf)]

                    # print(z)
                    
                    # get median value for free space 
                    Is[r,i,k-1,z+1]=datamean[f'I_Rx{j:d}_f{k:d}'].iloc[lims[0]:lims[1]].median()
                    Qs[r,i,k-1,z+1]=datamean[f'Q_Rx{j:d}_f{k:d}'].iloc[lims[0]:lims[1]].median()

            
            
    return start,stop,off1,on1,ID1, Qs,Is


def refCalibration(f,Rs=np.array([79.95,427.7,623,288.3]),Ls=np.array([11.48,11.36,11.18,11.44])*1e-3,Ac= 0.04**2*np.pi,Nc= 320.0,dR= 1.92 ,dB= 0.56 ,dC=1.92-0.225):    
    """   
    Calculate the theoretical magnitude of the normalized secondary effect generated by calibration coil. See notes for formulas and derivation.
    
    Parameters
    ----------
    f : float
        Frequency in Hz.
    Rs : np.array, optional
        Resistance of LR circuite for each state. The default is np.array([79.95,427.7,623,288.3]).
    Ls : np.array, optional
        Inductance of LR calibration circuit for each state. The default is np.array([11.48,11.36,11.18,11.44])*1e-3.
    Ac : float, optional
        Area of calibration coil (m**2) . The default is 0.04**2*np.pi.
    Nc : float, optional
        Turns of calibration coil. The default is 320.
    dR : float, optional
        Distance Transmitter reciver coil (m). The default is 1.92.
    dB : float, optional
        Distance transmitter bucking coil (m). The default is 0.56.
    dC : float, optional
        Distance transmitter calibration coil (m). The default is 1.92-0.225.

    Returns
    -------
    Z,ZI,ZQ,magZ
        Normalized secondary field Z, inphase, and quadrature of Z,Magnitude of Z

    """
    mu0= 1.2566e-6 # Vacuum permeability(N/A**2)

    gc=4*np.pi/3*mu0*Ac**2*Nc**2  # gain term due to coil size and turns
    Cd=1/dC**3*((dR/(dR-dC))**3-(dB/(dC-dB))**3 )  # coil distances therm

    omega=f*2*np.pi


    def funCRL(R,L,omega):
        """
        Compute theoretical I,Q therm depending on calibration coil for given R,L of calibration coil and omega 
        """
        denom=R**2+(omega*L)**2
        I=omega**2*L/denom
        Q=omega*R/denom
        return I,Q


    ZI,ZQ=funCRL(Rs,Ls,omega)
    ZI=ZI*gc*Cd
    ZQ=ZQ*gc*Cd
    magZ=np.sqrt(ZI**2+ZQ**2)
    Z=ZI+1j*ZQ
    return Z,ZI,ZQ,magZ
    
def  fitCalibrationParams(calQ,calI,f,plot=False,CalParams={}):
    """
    Fit calibration parameters to measured secondary signal produced by calibration coil.  Gain g, and phase phi.

    Parameters
    ----------
    calQ : TYPE
        DESCRIPTION.
    calI : TYPE
        DESCRIPTION.
    f : TYPE
        frequency.
    plot : TYPE, optional
        DESCRIPTION. The default is False.
    CalParams : dictionary, optional
        Parameters used to calculate theoretical calibration coild signal. 
        The default is empty. Then use values of LEMV1. Only for backward compatibility.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    
    # get differences from 
    
    calZ=np.sqrt(calI**2+calQ**2)
    # print(calZ)
    
    # get theoretical values
    if len(CalParams)>0:
        Z,ZI,ZQ,magZ=refCalibration(f,**CalParams)
        print('Using data from Logfile for calibration')
    else:  # use old values for backward compatibility 
        if len(calZ)==4:    
            Z,ZI,ZQ,magZ=refCalibration(f)
        else:
            Z,ZI,ZQ,magZ=refCalibration(f,Rs=np.array([77.5,143.5,400, 619,277,125]),
                                        Ls=np.ones(6)*11.365*1e-3,
                                        Ac= 0.04**2*np.pi,Nc= 320,dR= 1.92 ,dB= 0.56 ,dC=1.92-0.225)    
               


    # Sample data
    x = calQ*1j+calI
    
    # Define transformation function
    def trans(X, g, phi):
        return g*X*np.exp(phi*1j)

    # linear fit of magnitude of Z fit
    def linFit2(X, g):    
         return g*X

    params2, covariance2 = curve_fit(linFit2, calZ, magZ,p0=[1],bounds=([0],[1e16]))


    # Get g and phi with least squares fit
    def resFun(par):
        g,phi=par
        Delta=Z-trans(x,g,phi)	
        return np.sqrt(np.sum(np.real(Delta)**2+np.imag(Delta)**2))

    def resFun3(phi):
        Delta=Z-trans(x,params2[0],phi)	
        return np.sqrt(np.sum(np.real(Delta)**2+np.imag(Delta)**2))

   
    res =least_squares(resFun, (params2[0],0), bounds=([0, -np.pi], [1000,np.pi]))
    res3 =least_squares(resFun3, (0), bounds=([ -np.pi], [np.pi]))
    
    
    
    # # Print the fitting parameters
    print(f'Direct fit: g = {res.x[0]:.6f}, phi = {res.x[1]:.4e}')
    print(f'Fit MagZ+fit phi: g = {params2[0]:.6f}, phi = {res3.x[0]:.4e}')

    g=res.x[0]
    phi=res.x[1]


    if plot:

        #  Create a fit curve using the fitted parameters
        Z_fit = trans(x, res.x[0], res.x[1])
        Z_fit3 = trans(x, params2[0], res3.x[0])

        # # Plot the data and the fit curve
        pl.figure(figsize=(12,4))
        ax=pl.subplot(131)
        ax2=pl.subplot(132)
        ax3=pl.subplot(133)
        ax.scatter(calQ, ZQ, label='Data')
        ax.plot(calQ, np.imag(Z_fit), 'rx', label=' Fit g and phi')
        ax.plot(calQ, np.imag(Z_fit3), 'bx', label=' Fit magZ + fit phi')
        ax.set_xlabel('Q Voltage')
        ax.set_ylabel('quadrature(Z)')

        
        ax2.scatter(calI, ZI, label='Data')
        ax2.plot(calI, np.real(Z_fit), 'rx', label=' Fit g and phi')
        ax2.plot(calI, np.real(Z_fit3), 'bx', label=' Fit magZ + fit phi')
        ax2.set_xlabel('I Voltage')
        ax2.set_ylabel('inphase(Z)')
    
        ax3.scatter(calZ, magZ, label='Data')
        X=np.linspace(0,np.max(calZ),10)
        ax3.plot(calZ, linFit2(calZ, params2[0]), 'kx', label='Fit magZ=a*x')
        ax3.plot(X, linFit2(X, params2[0]), 'k:', label='Fitted data')
        ax3.set_xlabel('Z Voltage')
        ax3.set_ylabel('abs(Z)')
        ax2.legend()
        ax.legend()
        ax3.legend()
        ax3.text(0.15,0.13,
                f'Direct fit: g = {res.x[0]:.3f}, phi = {res.x[1]:.3e}',    
                horizontalalignment='left',
                verticalalignment='center',
                fontsize=8,
                transform = ax3.transAxes)
        ax3.text(0.15,0.08,
                f'Fit MagZ+fit phi: g = {params2[0]:.3f}, phi = {res3.x[0]:.3e}',    
                horizontalalignment='left',
                verticalalignment='center',
                fontsize=8,
                transform = ax3.transAxes)
        
        pl.tight_layout()
    
    
        pl.figure(figsize=(5,4))
        ax=pl.subplot(111)
        ax.scatter(ZI, ZQ, label='theoretical values')
        ax.plot(calI*g, calQ*g, 'rx', label='Data*g')
        ax.plot(np.real(Z_fit), np.imag(Z_fit), 'bx', label=' Fitted Data')
        ax.plot([0,0],ax.get_ylim(), '--k')
        ax.plot(ax.get_xlim(),[0,0], '--k')
        ax.set_xlabel('Inphase')
        ax.set_ylabel('quadrature')
        ax.legend()
        ax.text(0.15,0.13,
            f'Fit parameters: g = {res.x[0]:.3f}, phi = {res.x[1]:.3e}',    
            horizontalalignment='left',
            verticalalignment='center',
            fontsize=8,
            transform = ax.transAxes)
        
        
    return g,phi,[res,params2,res3]


def CheckCalibration(dataINS,datamean,params,f,Rx_ch=['ch1'],plot=True):
    """
    Check calibration. looks for calibration messages and fit data to theoretical 
    calibrations signal. Returns calibration parameters.

    Parameters
    ----------
    dataINS : TYPE
        DESCRIPTION.
    datamean : TYPE
        DESCRIPTION.
    params : TYPE
        DESCRIPTION.
    f : TYPE
        frequency.
    Rx_ch : TYPE, optional
        DESCRIPTION. The default is ['ch1'].
    plot : TYPE, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    gs : TYPE
        DESCRIPTION.
    phis : TYPE
        DESCRIPTION.
    calQs2 : TYPE
        DESCRIPTION.
    calIs2 : TYPE
        DESCRIPTION.
    calQ0 : TYPE
        DESCRIPTION.
    calI0 : TYPE
        DESCRIPTION.
    start : TYPE
        DESCRIPTION.
    stop : TYPE
        DESCRIPTION.
    on : TYPE
        DESCRIPTION.
    off : TYPE
        DESCRIPTION.

    """
    
    start,stop,off,on,ID, calQs,calIs =getCalTimes(dataINS,datamean,Rx_ch=Rx_ch)
    
    if plot:
        for  k,[st,stp,off2,on2] in enumerate(zip(start,stop,off,on ) ):
            
            lims=[datamean.time.searchsorted(st-2),datamean.time.searchsorted(stp+2)]
            
            for i,ch in enumerate(Rx_ch):
                j=i+1
                pl.figure()
                ax=pl.subplot(111)
                ax2=ax.twinx()
                ax.plot(datamean.time.iloc[lims[0]:lims[1]],datamean[f'Q_Rx{j:d}'].iloc[lims[0]:lims[1]],'xb',label=f'Q Rx{j:d}')
                ax2.plot(datamean.time.iloc[lims[0]:lims[1]],datamean[f'I_Rx{j:d}'].iloc[lims[0]:lims[1]],'+g',label=f'I Rx{j:d}')
                ylim=ax.get_ylim()
                ylim2=ax2.get_ylim()
                xlim=ax.get_xlim()
                
                for t in off2:
                    ax.plot([t,t],ylim,'k--',)
                for t in on2:
                    ax.plot([t,t],ylim,'r--',)      
                
                Q=calQs[k][i][0]
                I=calIs[k][i][0]
                ax.plot(xlim,[Q,Q],'b-.',)      
                ax2.plot(xlim,[I,I],'g-.',)
                for Q,I ,t1,t2 in zip(calQs[k][i][1:],calIs[k][i][1:],on2,off2):
                    ax.plot([t1-0.5,t2+0.5],[Q,Q],'b--',)      
                    ax2.plot([t1-0.5,t2+0.5],[I,I],'g--',)
                    
                ax.set_ylabel('Quadrature (-)')
                ax2.set_ylabel('InPhase (-)')
                ax.set_xlabel('time (s)')
                ax.legend(loc=2)
                ax2.legend(loc=1)
                pl.title(f'Cal{k+1:d}, {ch:s}' )
    
    # get calibration parameters from params        
    try:
        CalParams={'Rs': params['Rs'],
                   'Ls': np.ones_like(params['Rs'])*params['L_CalCoil'],
                   'Ac': params['A_CalCoil'],
                   'Nc': params['N_CalCoil'],
                   'dR': params['d_Rx'],
                   'dB': params['d_Bx'],
                   'dC': params['d_Cx'],
                   }
    except:
          CalParams={}

                               
    
    
    calQs2=[]
    calIs2=[]
    calQ0=[]
    calI0=[]
    gs=[]
    phis=[]
    
    for  k,[st,stp,off2,on2] in enumerate(zip(start,stop,off,on ) ):
         calQs2.append([])
         calIs2.append([])
         calQ0.append([])
         calI0.append([])
         # print(gs)
         gs.append([])
         phis.append([])
         for i,ch in enumerate(Rx_ch):     
            calQ=(calQs[k,i,1:].transpose()-calQs[k,i,0])
            calI=(calIs[k,i,1:].transpose()-calIs[k,i,0])
            
            g,phi,[res,params2,res3]=fitCalibrationParams(calQ,calI,f,plot=plot, CalParams= CalParams)
            
            calQs2[k].append(calQ)
            calIs2[k].append(calI)
            calQ0[k].append(calQs[k,i,0])
            calI0[k].append(calIs[k,i,0])
            gs[k].append(g)
            phis[k].append(phi)
    
    
    return gs,phis, calQs2,calIs2,calQ0,calI0,start,stop,on,off
    

def CheckCalibration_multiFreq(dataINS,datamean,params,plot=True):
    """
    

    Parameters
    ----------
    dataINS : TYPE
        DESCRIPTION.
    datamean : TYPE
        DESCRIPTION.
    params : TYPE
        DESCRIPTION.
    plot : TYPE, optional
        Create control plots. The default is True.

    Returns
    -------
    gs : TYPE
        gain factors.
    phis : TYPE
        Phase shifts.
    calQs2 : TYPE
        DESCRIPTION.
    calIs2 : TYPE
        DESCRIPTION.
    calQ0 : TYPE
        Offset Q.
    calI0 : TYPE
        Offsets I.
    start : TYPE
        DESCRIPTION.
    stop : TYPE
        DESCRIPTION.
    on : TYPE
        DESCRIPTION.
    off : TYPE
        DESCRIPTION.

    """
    
    Rx_ch=params['Rx_ch']
    freqs=params['freqs']
    
    n_freqs=len(freqs)
    start,stop,off,on,ID, calQs,calIs =getCalTimes_multi(dataINS,datamean,Rx_ch=Rx_ch,n_freqs=n_freqs)
    
    if plot:
        for  k,[st,stp,off2,on2] in enumerate(zip(start,stop,off,on ) ):
            
            lims=[datamean.time.searchsorted(st-2),datamean.time.searchsorted(stp+2)]
            
            for i,ch in enumerate(Rx_ch):
                j=i+1
                pl.figure()
                
                for r in range(1,n_freqs+1):
                    ax=pl.subplot(n_freqs,1,r)
                    ax2=ax.twinx()
                    
                    
                    ax.plot(datamean.time.iloc[lims[0]:lims[1]],datamean[f'Q_Rx{j:d}_f{r:d}'].iloc[lims[0]:lims[1]],'x',label=f'Q Rx{j:d}_f{r:d}')
                    ax2.plot(datamean.time.iloc[lims[0]:lims[1]],datamean[f'I_Rx{j:d}_f{r:d}'].iloc[lims[0]:lims[1]],'+',label=f'I Rx{j:d}_f{r:d}')
                    ylim=ax.get_ylim()
                    ylim2=ax2.get_ylim()
                    xlim=ax.get_xlim()
                    
                    for t in off2:
                        ax.plot([t,t],ylim,'k--',)
                    for t in on2:
                        ax.plot([t,t],ylim,'r--',)      
                    
                    Q=calQs[k,i,r-1,0]
                    I=calIs[k,i,r-1,0]
                    ax.plot(xlim,[Q,Q],'b-.',)      
                    ax2.plot(xlim,[I,I],'g-.',)
                    for Q,I ,t1,t2 in zip(calQs[k,i,r-1,1:],calIs[k,i,r-1,1:],on2,off2):
                        ax.plot([t1-0.5,t2+0.5],[Q,Q],'b--',)      
                        ax2.plot([t1-0.5,t2+0.5],[I,I],'g--',)
                        
                    ax.set_ylabel('Quadrature (-)')
                    ax2.set_ylabel('InPhase (-)')
                    ax.set_xlabel('time (s)')
                    ax.legend(loc=2)
                    ax2.legend(loc=1)
                    pl.title(f'Cal{k+1:d}, {ch:s}, f{r:d}' )
    
    # get calibration parameters from params        
    try:
        CalParams={'Rs': params['Rs'],
                   'Ls': np.ones_like(params['Rs'])*params['L_CalCoil'],
                   'Ac': params['A_CalCoil'],
                   'Nc': params['N_CalCoil'],
                   'dR': params['d_Rx'],
                   'dB': params['d_Bx'],
                   'dC': params['d_Cx'],
                   }
    except:
          CalParams={}
          
         
    
    calQs2=[]
    calIs2=[]
    calQ0=[]
    calI0=[]
    gs=[]
    phis=[]
    
    for  k,[st,stp,off2,on2] in enumerate(zip(start,stop,off,on ) ):
         calQs2.append([[]])
         calIs2.append([[]])
         calQ0.append([[]])
         calI0.append([[]])
         # print(gs)
         gs.append([[]])
         phis.append([[]])
         for i,ch in enumerate(Rx_ch):   
            
             for r,f in enumerate(freqs):
                calQ=-(calQs[k,i,r,1:].transpose()-calQs[k,i,r,0])
                calI=-(calIs[k,i,r,1:].transpose()-calIs[k,i,r,0])
                g,phi,[res,params2,res3]=fitCalibrationParams(calQ,calI,f,
                                                              plot=True, 
                                                              CalParams= CalParams)
                
                calQs2[k][i].append(calQ)
                calIs2[k][i].append(calI)
                calQ0[k][i].append(calQs[k,i,r,0])
                calI0[k][i].append(calIs[k,i,r,0])
                gs[k][i].append(g)
                phis[k][i].append(phi)
    
    
    return gs,phis, calQs2,calIs2,calQ0,calI0,start,stop,on,off



def lookupSectionRaw(file,start,stop,SPS=19200,units='seconds',title='',channels=[1,2,3]):
    
    if units=='seconds':
        i_start=int(start*SPS)
        i_stop=int(stop*SPS)-i_start
    elif units=='samples':
        i_start=int(start)
        i_stop=int(stop)-i_start
        start=start/SPS
        stop=stop/SPS
    else:
        raise ValueError('Units not recognised.') 
        
    data=np.genfromtxt(file, dtype='int32' , delimiter=',', usecols=[0,1,2,3],
                       converters={1:convert,2:convert,3:convert},
                       max_rows=i_stop,skip_header=i_start)
    if title=='':
        title='Raw data: {:.2f}s to {:.2f}s'.format(start,stop)
    return plotraw3(data,title=title,starttime=start,Chs=channels)


def loadSectionRawData(file,start,stop,SPS=19200,units='seconds'):
    
    if units=='seconds':
        i_start=int(start*SPS)
        i_stop=int(stop*SPS)-i_start
    elif units=='samples':
        i_start=int(start)
        i_stop=int(stop)-i_start
        start=start/SPS
        stop=stop/SPS
    else:
        raise ValueError('Units not recognised.') 
        
    return np.genfromtxt(file, dtype='int32' , delimiter=',', usecols=[0,1,2,3],
                       converters={1:convert,2:convert,3:convert},
                       max_rows=i_stop,skip_header=i_start)
  



def get_pointsond(d,data,Dd=1):
    ind=[]
    for p in d:
        ind.append(np.asarray((data.d<p+Dd) * (data.d>p-Dd)).nonzero())
    return ind


def get_arg_at_d(d,data,arg,Dd=1):
    ind=get_pointsond(d,data,Dd=Dd)
    out=np.zeros(len(ind))
    for j,ii in enumerate(ind):
        out[j]=data[arg].values[ii].mean()
    return out

def get_arg_at_d2(d,data,arg,Dd=1):
    
    if data.d.diff().mean()>0: 
        ind=np.searchsorted(data.d.values, d)
    else:
        ind=np.searchsorted(data.d.values[::-1], d)
        ind=len(data.d)-ind
    
    ind[ind==len(data.d)]=len(data.d)-1
    
    out=np.zeros(len(ind))
    for j,ii in enumerate(ind):
        out[j]=data[arg].values[ii].mean()
    return out


def two_smallest(numbers):
    m1 = m2 = float('inf')
    i1=i1=0
    for i,x in enumerate(numbers):
        if x <= m1:
            m1, m2 = x, m1
            i1,i2 = i, i1
        elif x < m2:
            m2 = x
            i2 = i
    return i1,i2, m1, m2

def get_d(x,main,per=30):
    x['d']=np.zeros(len(x))
    x['lat2']=np.zeros(len(x))
    x['lon2']=np.zeros(len(x))
    for  i in range(len(x)):
        # print(i)
        dlat=np.degrees(np.arcsin(per/6.371e6))
        dlon=np.degrees(np.arcsin(per/6.371e6*np.cos(np.radians(x.lat.iloc[i]))))
        main2=main.query('lat < {} and lat > {} and lon < {} and lon > {}'.format(x.lat.iloc[i]+dlat,
                                                                                  x.lat.iloc[i]-dlat,
                                                                                  x.lon.iloc[i]+dlon,
                                                                                  x.lon.iloc[i]-dlon))
        # print(i)
        if len(main2)>1:
            l=np.zeros(len(main2))
            for  j in range(len(l)):
                l[j]=distance.distance([main2.lat.iloc[j],main2.lon.iloc[j]],[x.lat.iloc[i],x.lon.iloc[i]]).m
            i1,i2, l1, l2=two_smallest(l)
            if i2<i1:
                a=i2
                i2=i1
                i1=a
            r1=np.array([main2.lat.iloc[i1],main2.lon.iloc[i1]])
            r2=np.array([main2.lat.iloc[i2],main2.lon.iloc[i2]])
            m=np.array([x.lat.iloc[i],x.lon.iloc[i]])
            # r1=np.array([0,1])
            # r2=np.array([1,1.5])
            # m=np.array([0.5,1.7])
            n=(r2-r1)/np.linalg.norm((r2-r1))
            m2=r1+n*np.dot(n,(m-r1))
            x['d'].iloc[i]=main2.d.iloc[i1]+distance.distance(r1,m2).m
            x['lat2'].iloc[i]=m2[0]
            x['lon2'].iloc[i]=m2[1]
        else:
            x['d'].iloc[i]=np.nan
            x['lat2'].iloc[i]=np.nan
            x['lon2'].iloc[i]=np.nan




def twos_comp(val, bits):
    """
    Compute the 2's complement of int value val.
    Used for converting the hex values into integer by shifting values with first bit =1 to negative values.     
    """
    if (val & (1 << (bits - 1))) != 0: # if sign bit is set e.g., 8bit: 128-255
        val = val - (1 << bits)        # compute negative value
    return val

convert = lambda x: twos_comp(int(x, 16),32)



def find_missing(d,pr=True):   # check missing data
    if isinstance(d,pd.core.frame.DataFrame):
        ind=np.where((d.index[1:]-d.index[:-1])>1)[0]
        gap=d.index[ind+1]-d.index[ind]
        if pr:
            print('Indices: ', ind)
            print('Gap: ', gap)
    elif isinstance(d,np.ndarray):
        ind=np.where((d[1:,0]-d[:-1,0])>1)[0]
        gap=d[:,0][ind+1]-d[:,0][ind]
        if pr:
            print('Indices: ', ind)
            print('Gap: ', gap)
    else:
        ind=np.array([])
        gap=np.array([])
    return ind, gap


def myfft(d):
    # fft= np.array([np.fft.rfft(d[:,i]),np.fft.rfft(d[:,2]),np.fft.rfft(d[:,3])])/len(d[:,1])
    fft= np.array([np.fft.rfft(d[:,i]) for i in range(1,d.shape[1])])/len(d[:,1])
    freq= np.fft.rfftfreq(len(d[:,1]), d=1.0/SPS)
    return fft,freq


def maxFFT(d):
    fft= np.abs(np.fft.rfft(d)/len(d))
    freq= np.fft.rfftfreq(len(d), d=1.0/SPS)
    i_100=freq.searchsorted(100)
    imax=np.argmax(fft[i_100:])+i_100  
    return fft[imax],freq[imax]

def getAfft(d,f):
    fft= np.abs(np.fft.rfft(d)/len(d))
    freq= np.fft.rfftfreq(len(d), d=1.0/SPS)
    return fft[np.searchsorted(freq,f)]

def getNoiseFloor(data,f,df,SPS,plot=False):
    fft=np.abs(np.fft.rfft(data))/data.shape[0]
    fft[0]=0
    freq= np.fft.rfftfreq(data.shape[0], d=1.0/SPS)
    il=freq.searchsorted(f-df)
    ih=freq.searchsorted(f+df)
    NoiseFloor=(np.mean(fft[:il])+np.mean(fft[ih:]))/2
    if plot:
        pl.figure()
        ax=pl.subplot(111)
        ax.plot(freq,np.log2(fft),'-b',label='FFT Amp')
        ylim=ax.get_ylim()
        ax.plot([freq[0],freq[-1]],[np.log2(NoiseFloor),np.log2(NoiseFloor)],'-',color='orange',label='Noise Floor')
        ax.plot([f,f],ylim,'-k',label='f')
        ax.plot([f+df,f+df],ylim,':k',label='f+df')
        ax.plot([f-df,f-df],ylim,':k',label='f-df')
        ax.set_xlabel('frequency (Hz)')
        ax.set_ylabel('amp (bits)')
        ax.legend()
    return NoiseFloor

def getSNR(data,f,df,SPS,plot=False):
    NoiseFloor=getNoiseFloor(data,f,df,SPS,plot=plot)
    A,phase,Q,I=getACorr(data,f,SPS)
    if plot:
        ax=pl.gca()
        xlim=ax.get_xlim()
        ax.plot(xlim,[np.log2(A),np.log2(A)],'--g',label='Amp. LockIn')
    return A/NoiseFloor
    
def correlate(d,f,SPS,phase=0,flowpass=100,lims=[]):
    Icorr=2*d*np.sin(2*np.pi*f*np.arange(len(d))/SPS+phase)
    Qcorr=2*d*np.cos(2*np.pi*f*np.arange(len(d))/SPS+phase)

    #filter the signal using a low pass filter
    flowpass_norm=flowpass/(SPS/2)
    b,a=signal.butter(3,flowpass_norm,'low')

    I=signal.filtfilt(b,a,Icorr)
    Q=signal.filtfilt(b,a,Qcorr)
    
    if len(lims)!=0:
        # lims=[0,len(d)]
        return Q[lims[0]:lims[1]],I[lims[0]:lims[1]]
    else:
        return Q,I
    
def getACorr(d,f,SPS,phase=0,flowpass=100,lims=[]):
    """
    Deriving amplitude and phase and then aplly mean value. Way less freqeuncy sensitive than getAcorr_old.
    """
    Q, I=correlate(d,f,SPS,phase=phase,flowpass=flowpass,lims=lims)
    
    A=np.mean(np.sqrt(Q**2+I**2))
    phase=np.mean(np.arctan2(I,Q))
    return A,phase,np.mean(Q),np.mean(I)

def getACorr_old(d,f,SPS,phase=0,flowpass=100,lims=[]):
    """
    Compute mean of I and Q before deriving amplitude and phase.
    """
    Q, I=correlate(d,f,SPS,phase=phase,flowpass=flowpass,lims=lims)
    Qm=np.mean(Q)
    Im=np.mean(I)
    A=np.mean(np.sqrt(Qm**2+Im**2))
    phase=np.mean(np.arctan2(I,Q))
    return A,phase,Qm,Im

def getACorr_simple(d,f,SPS,phase=0,flowpass=100,lims=[]):
    Q, I=correlate(d,f,SPS,phase=phase,flowpass=flowpass,lims=lims)
    return np.sqrt(np.mean(Q)**2+np.mean(I)**2)


def corrRunWindow(d,f,SPS,window, dWindow=0,phase=0):
    """
    Compute LockIn correlation over a running window. 
    
    Input:  
    ------
    d           array with data with strong signal. E.g. Tx voltage or current.
    f           frequency of reference signal 
    SPS         Sample pro second
    window      Wiondow width
    dWindow     Running window spacing. If dWindow=0 use window
    phase       phase of reference signal relative to beginning of data (rad)
    
    Output:
    -------
    t1,t2:  arrays with start and end times of data blockes
    
    """
    try:
        n=d.shape[1]
        l=d.shape[0]
    except:
        n=1
        l=d.shape[0]
    if dWindow==0:
        dWindow=window
        
    
    L=int(np.floor((l-window)/dWindow)+1)
    
    I=np.zeros([L,n])
    Q=np.zeros_like(I)
    
    for i in range(L):
        a=i*dWindow
        b=a+window
        s=np.sin(2*np.pi*f*np.arange(a,b)/SPS+phase)
        c=np.cos(2*np.pi*f*np.arange(a,b)/SPS+phase)
        if n==1:
            I[i]=np.mean(d[a:b]*s)
            Q[i]=np.mean(d[a:b]*c)
        else:          
            for j in range(n):
                I[i,j]=np.mean(d[a:b,j]*s)
                Q[i,j]=np.mean(d[a:b,j]*c)
   
    A=np.sqrt(Q**2+I**2)
    phase=np.arctan2(I,Q) 
    time=(np.arange(L)*dWindow+window/2)/SPS
    return A,phase,Q,I,time



def maxCorr_optimize(d,df=0.05,n=101,plot=False,flowpass=30,f0=0):
    
    if f0==0:
        Afft,f_fft=maxFFT(d)
    else:
        f_fft=f0
    
    fun =lambda x: -getACorr_simple(d,x,SPS,phase=0,flowpass=flowpass)
    
    f=optimize.fmin(fun, f_fft)
    
    if plot:
        
        freqs=[]
        As=[]
        phases=[]
        Qs=[]
        Is=[]

        
        for i in range(-50,50,1):
            f2=f_fft+df*i
            A,phase,Q,I=getACorr_old(d,f2,SPS)
            As.append(A)
            phases.append(phase)
            Qs.append(Q)
            Is.append(I)
            freqs.append(f2)
    
        imax=np.argmax(As)
        f2=freqs[imax]
        
        pl.figure()
        ax= pl.subplot(111)
        ax2=ax.twinx()
        ax.plot(freqs,As,label='Amp')
        ax.plot(freqs,Is,label='I')
        ax.plot(freqs,Qs,label='Q')
        ax2.plot(freqs,phases,'--k',label='phase')
        ylim=ax.get_ylim()
        ax.plot([f,f],ylim,'-k',label='f_max minimization')
        ax.plot([f2,f2],ylim,'-.k',label='f_max array')
        ax.plot([f_fft,f_fft],ylim,':k',label='fft max')
        ax.set_xlabel('frequency (Hz)')
        ax.set_ylabel('amp (-)')
        ax2.set_ylabel('phase (rad)')
        ax.legend()
    
    A,phase,Q,I=getACorr(d,f,SPS,phase=0,flowpass=flowpass) 
    return f,A,phase,Q,I

def maxCorr(d,df=0.05,n=101,plot=False,f0=0):
    
    if f0==0:
        Afft,f_fft=maxFFT(d)
    else:
        f_fft=f0
    
    
    freqs=[]
    As=[]
    A2s=[]
    phases=[]
    phases2=[]
    Qs=[]
    Is=[]

    for i in range(-int(n/2),int(n/2),1):
        f=f_fft+df*i
        A,phase,Q,I=getACorr_old(d,f,SPS)
        phases.append(phase)
        As.append(A)
        Qs.append(Q)
        Is.append(I)
        freqs.append(f)
        
        if plot:
            A2,phase2,Q2,I2=getACorr(d,f,SPS)
            A2s.append(A2)
            phases2.append(phase2)
        
    
    imax=np.argmax(As)
    f=freqs[imax]
    
    if plot:
        pl.figure()
        ax= pl.subplot(111)
        ax2=ax.twinx()
        ax.plot(freqs,As,label='Amp. rms(mean(I,Q))')
        ax.plot(freqs,A2s,label='Amp. mean(rms(I,Q))')
        ax.plot(freqs,Is,label='I')
        ax.plot(freqs,Qs,label='Q')
        ax2.plot(freqs,phases,'--k',label='phase')
        ax2.plot(freqs,phases2,'--g',label='phase2')
        ylim=ax.get_ylim()
        ax.plot([f,f],ylim,'-k',label='max')
        ax.plot([f_fft,f_fft],ylim,':k',label='fft max')
        ax.set_xlabel('frequency (Hz)')
        ax.set_ylabel('amp (-)')
        ax2.set_ylabel('phase (rad)')
        ax.legend()
        ax2.legend()
    
     
    return f,As[imax],phases[imax],Qs[imax],Is[imax]

def plotraw(d,divider=84.3):
    
    seconds= np.arange(0,len(d[:,2]))/SPS
    fft,freq=myfft(d)
    
    fig=pl.figure( figsize=(8,8) )
    ax = pl.subplot(311)
    ax1 = pl.subplot(312)
    ax2 = pl.subplot(313)
    
    
    ax.plot(seconds,np.log2(np.abs(d[:,2])))
    ax.plot(seconds,np.log2(np.abs(d[:,1])))
    ax.set_ylabel('amplitude (bits)')
    ax1.set_xlabel('time (s)')
    ax.set_ylim(bottom=0)
    
    ax1.plot(seconds,d[:,2])
    ax1.plot(seconds,d[:,1])
    ax1.set_ylabel('amplitude (-)')
    ax1.set_xlabel('time (s)')
    
    ax2.plot(freq/1000 ,np.log2(np.absolute(fft[1,:])),label='output amp ')
    ax2.plot(freq/1000 ,np.log2(np.absolute(fft[0,:])),label='input amp *{:.0f} '.format(divider))
    
    ax2.set_ylabel('amplitude (bits)')
    ax2.set_xlabel('frequency (kHz)')
    ax2.set_ylim(bottom=0)
    # ax.set_xlim([0,1000])
    ax2.legend()
    fig.tight_layout()
    return fig


def plotraw2(d,title='',starttime=0):
    
    seconds= np.arange(0,len(d[:,2]))/SPS+starttime
    fft,freq=myfft(d)
    
    fig=pl.figure( figsize=(8,8) )
    ax0a = pl.subplot(321)
    ax0b = pl.subplot(322)
    ax1a = pl.subplot(323)
    ax1b = pl.subplot(324)
    ax2a = pl.subplot(325)
    ax2b = pl.subplot(326)
    

    # ax0a.plot(seconds,np.log2(np.abs(d[:,0])))
    ax0a.plot(seconds,d[:,1])
    ax0a.set_ylabel('ch1 amplitude ()')
    ax0a.set_xlabel('time (s)')
 
    ax0b.plot(freq/1000 ,np.log2(np.absolute(fft[0,:])),label='output amp ')
    ax0b.set_ylabel('ch1 amplitude )')
    ax0b.set_xlabel('frequency (kHz)')
    ax0b.set_ylim(bottom=0)
    
    ax1a.plot(seconds,d[:,2])
    ax1a.set_ylabel('ch2 amplitude ()')
    ax1a.set_xlabel('time (s)')

    
    ax1b.plot(freq/1000 ,np.log2(np.absolute(fft[1,:])),label='output amp ')
    ax1b.set_ylabel('ch2 amplitude (bits)')
    ax1b.set_xlabel('frequency (kHz)')
    ax1b.set_ylim(bottom=0)
    
    ax2a.plot(seconds,d[:,3])
    ax2a.set_ylabel('ch3 amplitude (bits)')
    ax2a.set_xlabel('time (s)')

    
    ax2b.plot(freq/1000 ,np.log2(np.absolute(fft[2,:])),label='output amp ')
    ax2b.set_ylabel('ch3 amplitude (bits)')
    ax2b.set_xlabel('frequency (kHz)')
    ax2b.set_ylim(bottom=0)
    
    ax0a.set_title(title)
    fig.tight_layout()
    
    return fig

def plotraw3(d,title='',starttime=0,Chs=[1,2]):
    
    seconds= np.arange(0,len(d[:,2]))/SPS+starttime
    fft,freq=myfft(d)
    
    N=len(Chs)
    
    fig,axs=pl.subplots(2,N, figsize=(8,8) )
    
    for i,ch in enumerate(Chs):
        ax=axs[0,i]
        ax2=axs[1,i]
        ax.plot(seconds,d[:,ch])
        ax.set_ylabel(f'ch{ch:d} amplitude (-)')
        ax.set_xlabel('time (s)')
     
        ax2.plot(freq/1000 ,np.log2(np.absolute(fft[ch-1,:])),label='output amp ')
        ax2.set_ylabel(f'ch{ch:d} amplitude FFT (bits)')
        ax2.set_xlabel('frequency (kHz)')
        ax2.set_ylim(bottom=0)
    
    
    axs[0,0].set_title(title)
    fig.tight_layout()
    
    return fig

def plotFFT(data,SPS,ax=0,scale='linear',label='FFT Amp'):
    fft=np.abs(np.fft.rfft(data))/data.shape[0]
    fft[0]=0
    freq= np.fft.rfftfreq(data.shape[0], d=1.0/SPS)
    if ax==0:
        pl.figure()
        ax=pl.subplot(111)
    
    ax.plot(freq,fft,label=label)
    ax.set_xlabel('frequency (Hz)')
    ax.set_ylabel('amp (-)')
    ax.legend()
    ax.set_yscale(scale)
    
    
    
    
def plotFFTsnippets(data,scale='log'):
    n=len(data.snippets)
    fig,axes = pl.subplots(ncols=3, nrows=n, constrained_layout=True)
    for i in range(n):    
        plotFFT(data.snippets[i][:,0],SPS,ax=axes[i,0],scale='log')
        plotFFT(data.snippets[i][:,1],SPS,ax=axes[i,1],scale='log')
        plotFFT(data.snippets[i][:,2],SPS,ax=axes[i,2],scale='log')
        axes[i,0].set_title('ch{:d}, f= {:.0f}Hz'.format(1,data.fmax[i,0]))
        axes[i,1].set_title('ch{:d}, f= {:.0f}Hz'.format(2,data.fmax[i,1]))
        axes[i,2].set_title('ch{:d}, f= {:.0f}Hz'.format(3,data.fmax[i,2]))
        

def getIndexBlocks(d,threshold=0.5,plot=0,window=50,detrend=False,threshold2=0.3,parameter='mean'):
    """
    Get start and end times of blocks when Tx was on. 

    Parameters
    ----------
    d : TYPE
        array with data with strong signal. E.g. Tx voltage or current..
    threshold : TYPE, optional
        trhreshold for detecting Tx on. The default is 0.3.
    window : TYPE, optional
        Window size (samples) for computing running mean or std. The default is 20.
    parameter : TYPE, optional
            Wich type of math to use on running window. The default is 'std'.
    detrend : TYPE, optional
       if yes remove trend . The default is False.
    threshold2 : TYPE, optional
        threshold used for second onset refinement. The default is 0.3.


    Returns
    -------
    i_on,i_off : TYPE
        arrays with start and end times of data blockes

    
    
    """
    if window%2==0: # window size must be even  
        window+=1
    w2=int(window/2)
    index=np.arange(w2,len(d)-w2)
    
    if detrend:
        d=np.array(d,dtype=float)
        d[w2:-w2]-=np.sum(np.lib.stride_tricks.sliding_window_view(d, window), axis=-1)/window
        d[:w2]=0
        d[-w2:]=0

    
    # numpy vectorized function with axis selection
    d_amean = np.sum(np.abs(np.lib.stride_tricks.sliding_window_view(d, window)), axis=-1)/window
    m=d_amean.mean()
    
    d_std=np.std(np.lib.stride_tricks.sliding_window_view(d, window), axis=-1)
    std=d_std.mean()
    
    if parameter=='mean':
        d_on= np.array(d_amean/m>threshold,dtype=int)
    elif parameter=='std':
        d_on= np.array(d_std/std>threshold,dtype=int)
        
    i_on0=np.where(np.diff(d_on)==1)[0]+w2
    i_off=np.where(np.diff(d_on)==-1)[0]+w2
    
    if (i_on0[0]>i_off[0]):
        i_off=i_off[1:]
    if len(i_on0)>len(i_off):
        i_on0=i_on0[:-1]
    elif len(i_on0)<len(i_off):
        i_off=i_off[1:]
        
    i_on=np.zeros_like(i_on0)
    for j,i in enumerate(i_on0):
        m2=np.std(d[i:i_off[j]])
        try:
            i_on[j]=i+np.where(d[i:i+window]/m2>threshold2)[0][0]-1
        except IndexError:
            i_on[j]=i+w2
        
    if plot:   
        pl.figure()
        pl.plot(d/m,label='data')
        pl.plot(index,d_amean/m,label='mean abs. amp')
        pl.plot(index,d_std/std,label='std')
        pl.plot(index,d_on,label='on_off')
        # pl.plot(np.diff(d_on))
        for i in i_on:
            pl.plot([i,i],[-1,1],'k--')
        for i in i_on0:
             pl.plot([i,i],[-1,1],'k:')   
        for i in i_off:
                pl.plot([i,i],[-1,1],'k-.')
        pl.plot([],[],'k--',label='i_on')
        pl.plot([],[],'k.-',label='i_off')
        pl.plot([0,len(d_on)],[threshold,threshold],'k.-',label='threshold')
        pl.legend()
    
    return i_on,i_off



def getIndexBlocks2(d,threshold=0.4,window=30,detrend=False,threshold2=0.3):
    """
    Get start and end times of blocks when Tx was on. 

    Parameters
    ----------
    d : TYPE
        array with data with strong signal. E.g. Tx voltage or current..
    threshold : TYPE, optional
        trhreshold for detecting Tx on. The default is 0.3.
    window : TYPE, optional
        Window size (samples) for computing running mean or std. The default is 20.
    detrend : TYPE, optional
       if yes remove trend . The default is False.
    threshold2 : TYPE, optional
        threshold used for second onset refinement. The default is 0.3.


    Returns
    -------
    i_on,i_off : TYPE
        arrays with start and end times of data blockes

    
    """
    if window%2==0: # window size must be even  
        window+=1
    w2=int(window/2)

    
    if detrend:
        d=np.array(d,dtype=float)
        d[w2:-w2]-=np.sum(np.lib.stride_tricks.sliding_window_view(d, window), axis=-1)/window
        d[:w2]=0
        d[-w2:]=0
        
    
    # numpy vectorized function with axis selection    
    d_std=np.std(np.lib.stride_tricks.sliding_window_view(d, window), axis=-1)
    std=d_std.mean()
    
    d_on= np.array(d_std/std>threshold,dtype=int)
        
    i_on0=np.where(np.diff(d_on)==1)[0]+w2
    i_off=np.where(np.diff(d_on)==-1)[0]+w2
    
    if (i_on0[0]>i_off[0]):
        i_off=i_off[1:]
    if len(i_on0)>len(i_off):
        i_on0=i_on0[:-1]
    elif len(i_on0)<len(i_off):
        i_off=i_off[1:]
        
    i_on=np.zeros_like(i_on0)
    for j,i in enumerate(i_on0):
        m2=np.std(d[i:i_off[j]])
        try:
            i_on[j]=i+np.where(d[i:i+window]/m2>threshold2)[0][0]-1
        except IndexError:
            i_on[j]=i+w2

    return i_on,i_off



def get_snippets(d,i_Tx=1,usecols=[1,2,3],plot=False,threshold=0.2,window=50,dt_min=0,all_fmax=0,add_index=[[],[]],detrend=False):
    
    i_on,i_off=getIndexBlocks(d[:,i_Tx],threshold=threshold,plot=plot,window=window,detrend=detrend)
          
    
    if len(i_on)!=len(i_off):
        raise ValueError('I_on and I_off have not same lenght! You need to fix this or handle exception!')

    # filter times
    di=dt_min*SPS    
    ii=(i_off- i_on)>di
    i_on=i_on[ii]
    i_off=i_off[ii]
    add_index=np.array(add_index)
    if len(add_index[0])!=0:
        i_on=np.insert(i_on,i_on.searchsorted(add_index[:,0]),add_index[:,0])
        i_off=np.insert(i_off,i_off.searchsorted(add_index[:,1]),add_index[:,1])
    if plot:
        ax=pl.gca()
        for i in i_on:
            ax.plot([i,i],[-1,1],'g--')
        for i in i_off:
            ax.plot([i,i],[-1,1],'g.-')
            pl.show()
    
    
    
    
    snippets=[]
    k=0
    Afft=np.zeros([len(i_on),len(usecols)])
    fmax=np.zeros_like(Afft)
    Qs=np.zeros_like(Afft)
    Is=np.zeros_like(Afft)
    As=np.zeros_like(Afft)
    phases=np.zeros_like(Afft)
    NoiseFloor=np.zeros_like(Afft)
    SNR=np.zeros_like(Afft)
    
    for i,j in zip(i_on,i_off):

        snip=d[i:j,:]    
        snippets.append(snip[:,usecols])
        
        # process snip
        if all_fmax:
            for n,c in enumerate(usecols):
                f,A,phase,Q,I= maxCorr(snip[:,c],df=0.025,n=201,plot=False)
                fmax[k,n]=f
            f=fmax[k,np.where(np.array(usecols)==i_Tx)[0][0]]
        else:
            f,A,phase,Q,I= maxCorr(snip[:,i_Tx],df=0.025,n=101,plot=False)
            fmax[k,np.where(np.array(usecols)==i_Tx)[0][0]]=f
            
        for n,c in enumerate(usecols):
            Afft[k,n]=getAfft(snip[:,c],f)
            As[k,n],phases[k,n],Qs[k,n],Is[k,n]=getACorr(snip[:,c],f,SPS)
            NoiseFloor[k,n]=getNoiseFloor(snip[:,c],f,400,SPS,plot=False)
            SNR[k,n]=As[k,n]/NoiseFloor[k,n]
            
        k+=1

    return snipout(snippets,[Afft,fmax, Qs,Is,As,phases,i_on,i_off,SNR,NoiseFloor],
                   ['Afft','fmax', 'Q','I','Amp','phase','i_on','i_off','SNR','NoiseFloor'])


class snipout():
    def __init__(self,snippets,attr,attrNames):
        self.snippets=snippets
        for att,name in zip(attr,attrNames):
            setattr(self,name,att)


def plotsnip_raw(d):
    
    seconds= np.arange(0,len(d[:,0]))/SPS
    fft= np.array([np.fft.rfft(d[:,0]),np.fft.rfft(d[:,1]),np.fft.rfft(d[:,2])])/len(d[:,1])
    freq= np.fft.rfftfreq(len(d[:,0]), d=1.0/SPS)
    
    fig=pl.figure( figsize=(8,8) )
    ax0a = pl.subplot(321)
    ax0b = pl.subplot(322)
    ax1a = pl.subplot(323)
    ax1b = pl.subplot(324)
    ax2a = pl.subplot(325)
    ax2b = pl.subplot(326)
    

    # ax0a.plot(seconds,np.log2(np.abs(d[:,0])))
    ax0a.plot(seconds,d[:,0])
    ax0a.set_ylabel('ch1 amplitude (bits)')
    ax0a.set_xlabel('time (s)')
 
    ax0b.plot(freq/1000 ,np.log2(np.absolute(fft[0,:])),label='output amp ')
    ax0b.set_ylabel('ch1 amplitude (bits)')
    ax0b.set_xlabel('frequency (kHz)')
    ax0b.set_ylim(bottom=0)
    
    ax1a.plot(seconds,d[:,1])
    ax1a.set_ylabel('ch2 amplitude (bits)')
    ax1a.set_xlabel('time (s)')

    
    ax1b.plot(freq/1000 ,np.log2(np.absolute(fft[1,:])),label='output amp ')
    ax1b.set_ylabel('ch2 amplitude (bits)')
    ax1b.set_xlabel('frequency (kHz)')
    ax1b.set_ylim(bottom=0)
    
    ax2a.plot(seconds,d[:,2])
    ax2a.set_ylabel('ch3 amplitude (bits)')
    ax2a.set_xlabel('time (s)')

    
    ax2b.plot(freq/1000 ,np.log2(np.absolute(fft[2,:])),label='output amp ')
    ax2b.set_ylabel('ch3 amplitude (bits)')
    ax2b.set_xlabel('frequency (kHz)')
    ax2b.set_ylim(bottom=0)

    fig.tight_layout()
    return fig



    
def plot_modeledWindow(snip,f, ch=0,window=200, dWindow=0):    
    A,phase,Q,I,time=corrRunWindow(snip,f,SPS,window,dWindow=dWindow)
    
    pl.figure(figsize=(10,8))
    n=snip.shape[1]
    ax0= pl.subplot(n,1,1)
    for i in range(n):
        if i!=0:
                    ax= pl.subplot(n,1,i+1,sharex = ax0)
        else:
            ax=ax0
        ax2=ax.twinx()
        ax.plot(time,A[:,i],label='Amplitude')
        ax.plot(time,I[:,i],label='Inphase')
        ax.plot(time,Q[:,i],label='Quadratue')
        ax2.plot(time,phase[:,i],'--k',label='phase')
        ax.set_ylabel('amp (-)')
        ax2.set_ylabel('phase (rad)')
        ax.legend()
    ax.set_xlabel('time (s)')
    pl.tight_layout()

def plot_spectrograms(data,SPS,scale='dB',**kwargs):    
    pl.figure(figsize=(8,8))
    ax=pl.subplot(211)
    sp,freqs,times,im1=ax.specgram(data, Fs=SPS,mode='magnitude',scale=scale ,cmap=cmCrameri.batlow,**kwargs)
    ax.set_title('Magnitude')
    ax.set_ylabel("frequency (Hz)")
    ax.set_xlabel("time (s)")
    pl.colorbar(im1,ax=ax)
    
    ax2=pl.subplot(212)
    sp,freqs,times,im2=ax2.specgram(data, Fs=SPS, mode='phase', cmap=cmCrameri.batlow,**kwargs)
    ax2.set_title('Phase')
    ax2.set_ylabel("frequency (Hz)")
    ax2.set_xlabel("time (s)")
    pl.colorbar(im2,ax=ax2)
    pl.tight_layout()
    
    
    
def inspect_snippet(data, i,ch=0,window=200,SPS=19200):
    snip=data.snippets[i]
    
    plotsnip_raw(snip)
    
    f,A,phase,Q,I= maxCorr_optimize(snip[:,ch],df=0.01,n=101,plot=True)
    
    plot_modeledWindow(snip,f, ch=0,window=100, dWindow=50)
    
    plot_spectrograms(snip[:,ch],SPS,scale='dB')
    
    return f,A,phase,Q,I
    