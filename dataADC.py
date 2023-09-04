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
from cmcrameri import cm as cmCrameri
from scipy import signal
from scipy import optimize
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo


from INSLASERdata import *

SPS= 19200
ZoneInfo("UTC")

sys.path.append(r'C:\Users\Laktop')
from emagpy_seaice import Problem
from emagpy_seaice import invertHelper 


#%%  functions for laoding and handling data

def loadDataLEM(path,name):
    file=path+'/LEM'+name+'.csv'
    return pd.read_csv(file,header=15)

def processDataLEM(path,name, Tx_ch='ch3', Rx_ch=['ch1','ch2'],
                     plot=False,savefile=True,
                     window=1920,freq=0,phase0=0,SPS=19200,flowpass=50,i_cal=[],
                     INSkargs={},
                     **kwargs):
 
    
    version='v1.0'
    
    file=path+'/ADC'+name+r'.csv'
    fileLASER=path+'/INS'+name+'.csv'
    if savefile:
        fileOutput=path+'/LEM'+name+'.csv'
    
    
    print('Load INS+Laser: ',file)
    dataINS=INSLASERdata(fileLASER,name='\\INS'+name+'.csv',**INSkargs)
    
    print('Reading file: ',file)
    datamean, i_missing, gap,f,phase0=loadADCraw_singleFreq(file,
                                                   f=freq,phase0=phase0,SPS=SPS,
                                                   flowpass=flowpass,window=window,keep_HF_data=False,
                                                   i_Tx=int(Tx_ch[2:]),
                                                   **kwargs)    
    
    print(f'Freq: {f:.2f} Hz')
    print(f'Phase: {phase0:.2f} rad')
    
    tx=int(Tx_ch[2:])
    columns=[]
    
    # normalize Rx by Tx
    NormRxbyTx(datamean,Rx_ch,columns,tx=tx)

    # Calibrate ADC
    A0,phase0=Calibrate(datamean,Rx_ch,i_cal)
    columns.extend(['Q_Rx1','I_Rx1','Q_Rx2','I_Rx2',])
    
    # sync ADC and INS/Laser
    sync_ADC_INS(datamean,dataINS)
    datamean['time']=datamean.TOW-datamean.TOW.iloc[0]
    
    
    if savefile:
        # write file header
        f=open(fileOutput,'w')
        
        f.write("# Signal extracted with LockIn form raw data\n")
        f.write("# Processing date: {:} \tScript verion: {:s}  \n\n".format( datetime.now().strftime("%Y-%m-%d, %H:%M:%S"),version))
        f.write("# Freqeuncy LockIn: {:} Hz\n".format(f))
        f.write("# Phase LockIn: {:}\n".format(phase0))
        f.write("# SPS: {:}\n\n".format(SPS))
        f.write("# Frequency low pass: {:}\n".format(flowpass))
        f.write("# Averaging window size: {:}\n".format(window))
        f.write("# TxChannel: {:}\n".format(Tx_ch))
        f.write("# Rx channels: {:}\n".format(str(Rx_ch)))
        if i_cal!=[]:
            f.write("# Index calibration: [{:},{:}]\n".format(str(i_cal[0]),str(i_cal[1])))
            for i,ch in enumerate(Rx_ch):
                f.write("# A0_Rx{:d}= {:f}, phase0_Rx{:d}= {:f},\n".format(i+1,A0[i],i+1,phase0[i]))
        else:
            f.write("# No calibration")
            
        f.write("#\n")
        f.write("# Missing index: {:}, Gap sizes: {:}\n".format(str( i_missing), str( gap)))
        f.write("##\n")
        f.close()
        
        # columns to save
        columns+=['t', 'time','TOW', 'lat', 'lon', 'h_GPS',  'h_Laser', 'roll', 'pitch','heading', 'velX', 'velY', 'velZ',  
                  'signQ', 'TempLaser',
                  'Q1', 'I1', 'Q2', 'I2', 'Q3', 'I3','A1', 'phase1','A2', 'phase2', 'A3', 'phase3' ] #
        
        #save to file

        try:
            datamean.to_csv(fileOutput,mode='a',index=True,header=True,columns=columns)
        except Exception as e: 
                print("error. Can't save data ")
                print(e)
                print('Columns to write:', columns)
                print('Columns in datamean:', datamean.keys())
    
    
    if plot:
        
        plot_summary(dataINS,getextent(dataINS),heading=False)
        
        for i,ch in enumerate(Rx_ch):
            j=i+1
            pl.figure()
            pl.plot(datamean.index/SPS,datamean[f'Q_Rx{j:d}'],'x',label='Q Rx')
            pl.plot(datamean.index/SPS,datamean[f'I_Rx{j:d}'],'x',label='I Rx')
            pl.plot(pl.gca().get_xlim(),[0,0],'k--',)
            pl.ylabel('amplitude (-)')
            pl.xlabel('time (s)')
            pl.legend()
            pl.title(ch)
    
    return datamean, dataINS



def NormRxbyTx(datamean,Rx_ch,columns,tx=2):
    for i,ch in enumerate(Rx_ch):
        rx=int(ch[2:])
        j=i+1
        datamean[f'A_Rx{j:d}']=datamean[f'A{rx:d}']/datamean[f'A{tx:d}']
        datamean[f'phase_Rx{j:d}']=datamean[f'phase{rx:d}']-datamean[f'phase{tx:d}']   
        columns.append(f'A_Rx{j:d}')
        columns.append(f'phase_Rx{j:d}')
        
def Calibrate(datamean,Rx_ch,i_cal):
    A0=[]
    phase0=[]
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
    
    return A0,phase0



def LockInADCrawfile(path,name, Tx_ch='ch2', Rx_ch=['ch1'],
                     plot=False,
                     window=1920,freq=1063.3985,phase0=0,SPS=19200,flowpass=50,i_cal=[],
                     **kwargs):
    '''

    Parameters
    ----------
    path : TYPE
        DESCRIPTION.
    name : TYPE
        DESCRIPTION.
    Tx_ch : TYPE, optional
        DESCRIPTION. The default is 'ch3'.
    Rx_ch : TYPE, optional
        DESCRIPTION. The default is ['ch1','ch2'].
    plot : TYPE, optional
        DESCRIPTION. The default is False.
    window : TYPE, optional
        DESCRIPTION. The default is 1920.
    freq : TYPE, optional
        DESCRIPTION. The default is 1063.3985.
    phase0 : TYPE, optional
        DESCRIPTION. The default is 0.
    SPS : TYPE, optional
        DESCRIPTION. The default is 19200.
    flowpass : TYPE, optional
        DESCRIPTION. The default is 50.
    i_cal : TYPE, optional
        DESCRIPTION. The default is [].
    **kwargs : TYPE
        DESCRIPTION.

    Returns
    -------
    datamean : TYPE
        DESCRIPTION.

    '''
    
    file=path+r'\\'+name+r'.csv'
    savefile=path+'\\'+name+r'LockIn.csv'
    version='v1.0'
    
    
    print('Reading file: ',file)
    datamean, i_missing, gap,f,phase0=loadADCraw_singleFreq(file,
                                                   f=freq,phase0=phase0,SPS=SPS,
                                                   flowpass=flowpass,window=window,keep_HF_data=False,
                                                   i_Tx=int(Tx_ch[2:]),
                                                   **kwargs)    
    
    print(f'Freq: {f:.2f} Hz')
    print(f'Phase: {phase0:.2f} rad')
    
    tx=int(Tx_ch[2:])
    columns=[]
    
    # normalize Rx by Tx
    for i,ch in enumerate(Rx_ch):
        rx=int(ch[2:])
        j=i+1
        datamean[f'A_Rx{j:d}']=datamean[f'A{rx:d}']/datamean[f'A{tx:d}']
        datamean[f'phase_Rx{j:d}']=datamean[f'phase{rx:d}']-datamean[f'phase{tx:d}']   
        columns.append(f'A_Rx{j:d}')
        columns.append(f'phase_Rx{j:d}')


    A0,phase0=Calibrate(datamean,Rx_ch,i_cal)
 
    
    # write file header
    f=open(savefile,'w')
    
    f.write("# Signal extracted with LockIn form raw data\n")
    f.write("# Processing date: {:} \tScript verion: {:s}  \n\n".format( datetime.now().strftime("%Y-%m-%d, %H:%M:%S"),version))
    f.write("# Freqeuncy LockIn: {:} Hz\n".format(freq))
    f.write("# Phase LockIn: {:}\n".format(phase0))
    f.write("# SPS: {:}\n\n".format(SPS))
    f.write("# Frequency low pass: {:}\n".format(flowpass))
    f.write("# Averaging window size: {:}\n".format(window))
    f.write("# TxChannel: {:}\n".format(Tx_ch))
    f.write("# Rx channels: {:}\n".format(str(Rx_ch)))
    if i_cal!=[]:
        f.write("# Index calibration: [{:},{:}]\n".format(str(i_cal[0]),str(i_cal[1])))
        for i,ch in enumerate(Rx_ch):
            f.write("# A0_Rx{:d}= {:f}, phase0_Rx{:d}= {:f},\n".format(i+1,A0[i],i+1,phase0[i]))
    else:
        f.write("# No calibration")
        
    f.write("#\n")
    f.write("# Missing index: {:}, Gap sizes: {:}\n".format(str( i_missing), str( gap)))
    f.write("##\n")
    f.close()
    
    # columns to save
    columns+=[ 'Q1', 'I1', 'Q2', 'I2', 'Q3', 'I3','A1', 'phase1','A2', 'phase2', 'A3', 'phase3' ] #'t', 'TOW', 'lat', 'lon', 'h_GPS',  'h_Laser', 'roll', 'pitch','heading', 'velX', 'velY', 'velZ',  'signQ', 'TempLaser',
    
    #save to file
    try:
        datamean.to_csv(savefile,mode='a',index=True,header=True,columns=columns)
    except Exception as e: 
            print("error. Can't save data ")
            print(e)
            print('Columns to write:', columns)
            print('Columns to in datamean:', datamean.keys())
    
    if plot:
        for i,ch in enumerate(Rx_ch):
            rx=int(ch[2:])
            j=i+1
            pl.figure()
            pl.plot(datamean.index/SPS,datamean[f'Q_Rx{j:d}'],'x',label='Q Rx')
            pl.plot(datamean.index/SPS,datamean[f'I_Rx{j:d}'],'x',label='I Rx')
            pl.ylabel('amplitude (-)')
            pl.xlabel('time (s)')
            pl.legend()
            pl.title(ch)
        
            pl.figure()
            pl.plot(datamean.index/SPS,datamean[f'A{rx:d}'],'x', label='Amplitude Rx signal')
            pl.plot(datamean.index/SPS,datamean[f'Q{rx:d}'],'x', label='Quadr Rx')
            pl.plot(datamean.index/SPS,datamean[f'I{rx:d}'],'x', label='Inph Rx')
            pl.ylabel('amplitude (-)')
            pl.xlabel('time (s)')
            pl.legend()
            pl.title(ch)

        pl.figure()
        pl.plot(datamean.index/SPS,datamean[f'A{tx:d}'],'x', label='Amplitude Tx')
        pl.plot(datamean.index/SPS,datamean[f'Q{tx:d}'],'x', label='Quadr Tx')
        pl.plot(datamean.index/SPS,datamean[f'I{tx:d}'],'x', label='Inph Tx')
        pl.ylabel('amplitude (-)')
        pl.xlabel('time (s)')
        pl.legend()
        pl.title(Tx_ch)
        
        
        pl.figure()
        ax=pl.subplot(211)
        ax2=pl.subplot(212)
        for i,ch in enumerate(Rx_ch):
            rx=int(ch[2:])
            j=i+1
            ax.plot(datamean.index/SPS,datamean[f'Q_Rx{j:d}'],'x',label=ch)
            ax2.plot(datamean.index/SPS,datamean[f'I_Rx{j:d}'],'x',label=ch)
        ax.set_ylabel('Quadrature (-)')
        ax.set_xlabel('time (s)')
        ax2.set_ylabel('InPhase (-)')
        ax2.set_xlabel('time (s)')
        ax.legend()
        ax2.legend()
    
    return datamean

def loadADCraw_singleFreq(file,window=1920,f=0,phase0=0,SPS=19200,
                          flowpass=50,chunksize=19200,keep_HF_data=False,
                          findFreq=True,i_Tx=3,i_blok=[],
                          **kwargs):
    """
    Read raw data file and extract signal over LockIn with lowpass filter. Average with running window.  
    
    Input:  
    ------
    file          path to file containing rawdata
    
    
    Output:
    -------
    datamean 
    i_missing
    gap
    data
    
    """
    #filter the signal using a low pass filter
    flowpass_norm=flowpass/(SPS/2)
    b,a=signal.butter(3,flowpass_norm,'low')
    
    
    i_missing=np.array([])
    gap=np.array([])
    
    f0=f
    if findFreq:
        data=np.genfromtxt(file, dtype='int32' , delimiter=',', usecols=[0,1,2,3],
                           converters={1:convert,2:convert,3:convert},
                           max_rows=200000)
        
        plot=True
        threshold=0.5
        win_f0=100
        dt_min=0.2
        detrend=False
        df=0.005
        n_fmaxsearch=200
        drop_startup=15000
        
        # find starting poit of data 
        if len(i_blok)==0:
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
            
        elif len(i_blok)==2:
            ia=i_blok[0]
            ib=i_blok[1]
        else:
            raise ValueError('i_block must be a list with two limits. Frequency can not be determined!')
        
        f,A,phase0,Q,I= maxCorr(data[ia:ib,i_Tx],df=df,n=n_fmaxsearch,plot=plot,f0=f0)
        
        if plot:
            plotraw2(data[ia:ib,:])
        
    
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
        datamean=pd.concat([datamean,chunk2.rolling(window,center=True,min_periods=1,step=window).mean()])
        #datamean=pd.concat([datamean,chunk2.rolling(window,center=True,min_periods=1).mean()[int(window/2)::window]])   # using step argument instead of slicing shulud be more efficient but need a higher version of pandas 
        
        
        
        i_rest=int(chunk2.index[-1]-(chunk2.index[-1]-chunk2.index[0])%window)
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





#%%  Sync INS/Laser and ADC

def sync_ADC_INS(datamean,dataINS):
    
    TOW=datamean.index/SPS+get_TOW0(dataINS)
    t = gps_datetime_np(dataINS.PINS1.GPSWeek[0],TOW)
    datamean['t']=t
    datamean['TOW']=TOW

    # interpolate PINS1 data
    interpolData(dataINS.PINS1,datamean,['TOW','heading','velX','velY','velZ','lat','lon', 'height',])
    datamean.rename(columns={'height':'h_GPS'}, inplace=True)
    interpolData(dataINS.Laser,datamean,['h_corr', 'roll', 'pitch','signQ', 'T'])
    datamean.rename(columns={"T": "TempLaser",'h_corr':'h_Laser'}, inplace=True)

def interpolData(data,datamean,proplist):
    ind=np.searchsorted(data.TOW,datamean.TOW)
    ind[ind>len(data.TOW)-1]=len(data.TOW)-1
    ind[ind==0]=1
    dt=(datamean.TOW-data.TOW[ind])/(data.TOW[ind]-data.TOW[ind-1])
    
    for p in proplist:
        x=getattr(data,p)
        datamean[p]=x[ind]+(x[ind]-x[ind-1])*dt
    
def gps_datetime(time_week, time_s, leap_seconds=18):
    '''

    Parameters
    ----------
    time_week : scalar
        GPS week.
    time_s : scalar
        time of week in seconds.
    leap_seconds : TYPE, optional
        Leap seconds. The default is 18.

    Returns
    -------
    datetime
        DESCRIPTION.

    '''
    gps_epoch = datetime(1980, 1, 6, tzinfo=ZoneInfo("UTC"))
    return gps_epoch + timedelta(weeks=time_week, 
                                 seconds=time_s-leap_seconds)

gps_datetime_np=np.vectorize(gps_datetime, doc='Vectorized `gps_datetime`')

def get_TOW0(data,iStart=2):
    '''
    Parameters
    ----------
    data : INSLASERdata object 
        
    iStart : int, optional
        DESCRIPTION. The default is 2.

    Returns
    -------
    Time of week (s) of ADC data start 

    '''
    if data.PSTRB.pin[iStart]!=8:
        raise ValueError(f'Timestamp {iStart} was not on pin 8 of INS.', iStart )
    
    return data.PSTRB.TOW[iStart]/1000

def get_t0(data,iStart=2):
    '''
    Parameters
    ----------
    data : INSLASERdata object 
        
    iStart : int, optional
        DESCRIPTION. The default is 2.

    Returns
    -------
    datetime of ADC data start 

    '''
    TOW=get_TOW0(data,iStart=iStart)
    
    return gps_datetime(data.PINS1.GPSWeek[0], TOW, leap_seconds=18)

#%%  various functions


# def getCalConstants(datamean,caltimes,start,stop,off,on,ID):

def getCalTimes(dataINS,datamean,t_buf=0.2,t_int0=3,Rx_ch=['ch1']):
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
    


def CheckCalibration(dataINS,datamean,Rx_ch=['ch1']):
    
    start,stop,off,on,ID, calQs,calIs =getCalTimes(dataINS,datamean,Rx_ch=Rx_ch)
    
    for  k,[st,stp,off2,on2] in enumerate(zip(start,stop,off,on ) ):
        
        lims=[datamean.time.searchsorted(st-2),datamean.time.searchsorted(stp+2)]
        
        for i,ch in enumerate(Rx_ch):
            j=i+1
            pl.figure()
            ax=pl.subplot(111)
            ax2=ax.twinx()
            ax.plot(datamean.time.iloc[lims[0]:lims[1]],datamean[f'Q_Rx{j:d}'].iloc[lims[0]:lims[1]],'xb',label='Q Rx')
            ax2.plot(datamean.time.iloc[lims[0]:lims[1]],datamean[f'I_Rx{j:d}'].iloc[lims[0]:lims[1]],'xg',label='I Rx')
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
    
    return start,stop,off,on,ID, calQs,calIs
    

def lookupSectionRaw(file,start,stop,SPS=19200,units='seconds',title=''):
    
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
    return plotraw2(data,title=title,starttime=start)




def EMagPy_forwardmanualdata(depths,freqs,d_coils=1.929,plot=True):

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
    fft= np.array([np.fft.rfft(d[:,1]),np.fft.rfft(d[:,2]),np.fft.rfft(d[:,3])])/len(d[:,1])
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
    
    if len(lims)==0:
        lims=[0,len(d)]
    return Q[lims[0]:lims[1]],I[lims[0]:lims[1]]
    
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



def maxCorr_optimize(d,df=0.05,n=101,plot=False,flowpass=30):
    
    Afft,f_fft=maxFFT(d)
    
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



def plotFFT(data,SPS,ax=0,scale='linear'):
    fft=np.abs(np.fft.rfft(data))/data.shape[0]
    fft[0]=0
    freq= np.fft.rfftfreq(data.shape[0], d=1.0/SPS)
    if ax==0:
        pl.figure()
        ax=pl.subplot(111)
    
    ax.plot(freq,fft,label='FFT Amp')
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
    Input:
    ------
    d           array with data with strong signal. E.g. Tx voltage or current.
    threshold    trhreshold for detecting Tx on 
    
    Output:
    -------
    t1,t2:  arrays with start and end times of data blockes
    
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
            f,A,phase,Q,I= maxCorr(snip[:,i_Tx],df=0.025,n=201,plot=False)
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



    
def plot_corrWindow(snip,f, ch=0,window=200, dWindow=0):    
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
    
    plot_corrWindow(snip,f, ch=0,window=100, dWindow=50)
    
    plot_spectrograms(snip[:,ch],SPS,scale='dB')
    
    return f,A,phase,Q,I
    