# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 12:09:02 2022

Class and function used for handling data files with mixed UBX data from ublox GNSS modules and frm LDS70A Laser altimeter. 

@author: Laktop
"""

import numpy as np
import pandas as pd
import re
import math
from datetime import datetime, timedelta
import matplotlib.pyplot as pl
import os,sys
from cmcrameri import cm
from geopy import distance 
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import io
from urllib.request import urlopen, Request
from PIL import Image
import gpxpy
from zoneinfo import ZoneInfo

# %%  data class

class MSG_type:
    
    def __init__(self,MSG):
        self.len=0
        self.MSG=MSG
        
    def addData(self,keys,values):
        self.len=(len(values))
        if len(keys)!=len(values[0]):
            print("Keys and values don't have same length!!")
            print(values[0])
            print(keys)
            
            return
        self.keys=keys.copy()
        for i,k in enumerate(keys):
            setattr(self,k,np.array([l[i] for l in values]))

_MSG_list_=['Laser','PINS1','PSTRB','PINS2','GPGGA','PGPSP','VBat','Temp','Cal','SDwrite']    # NMEA message list to parse
_keyList_=[['h','signQ','T','TOW'],        
          ['TOW','GPSWeek','insStatus','hdwStatus','roll','pitch','heading','velX', 'velY', 'velZ','lat', 'lon', 'elevation','OffsetLLA_N','OffsetLLA_E','OffsetLLA_D'],
          ['GPSWeek','TOW','pin','count'],
          ['TOW','GPSWeek','insStatus','hdwStatus','QuatW','QuatX','QuatY','QuatZ','velX', 'velY', 'velZ','lat', 'lon', 'elevation'],
          ['UTC','lat','lat_unit','lon','lon_unit','Fix','NSat','hDop','MSL','MSL_unit','ondulation','ondulation_unit','last_DGP','DGPS_ID'],
          ['TOW','GPSWeek','status','Latitude','Longitude','elevation_HAE','elevation_MSL','pDOP','hAcc','vAcc','Vel_X','Vel_Y','Velocity_Z','sAcc','cnoMean','towOffset','leapS'],
          ['V','TOW'],
          ['T','sensor','TOW'],
          ['On','ID','TOW'],
          ['TOW']
          ,]  # order matters!!


class INSLASERdata:
    
        
    # extr_list=['PINS1','Laser']
    
    def __init__(self,filepath,name='',load=True, droplaserTow0=True,
                 correct_Laser=True,distCenter=0, pitch0=0, roll0=0,laser_time_offset=0,c_pitch=1,c_roll=1,verbose=False):
        """
            Read GNSS and Laser data from .ubx data file. Additional methods are available for plotting and handling data.    
        
            Inputs:
            ---------------------------------------------------    
            filepath:           file path
           
           
        """
        
        
        if name!='': 
            self.name=name
        else:
             self.name=filepath.split('\\')[-1]   
        
        self.filepath=filepath
        self.ToW=0
        self.distCenter=distCenter
        self.pitch0=pitch0
        self.roll0=roll0
        self.c_pitch=c_pitch
        self.c_roll=c_roll
        
        self.keyList=_keyList_.copy()
        self.MSG_list=_MSG_list_.copy()
        
        
        if verbose:
            self.verbose=True
        else:
                self.verbose=False
        
        if load:
            self.loadData(correct_Laser=correct_Laser,droplaserTow0=droplaserTow0)

    
    def loadData(self,correct_Laser=0,droplaserTow0=True):
        if not os.path.isfile(self.filepath) :
            print("File not found!!!")
            raise FileNotFoundError('File: {:s}'.format(self.filepath))
        
        
        
        print('Reading file: ',self.filepath)
        print('----------------------------')
        file = open(self.filepath, 'rt')
        
        for msg in (self.MSG_list):
            setattr(self,msg+'List',[])
        
        i=0

        self.corrupt=[]
        self.other=[]
        self.dropped=[]
        
        for l in file:
            # print(l)
            # l2=str(l)
            i+=1
            if l[0]=='#':
                continue
            elif l.find('$')!=-1:
                Msg_key,data=parseNMEAfloat(l,self.verbose)
                
                if Msg_key=='PINS1':
                    self.ToW=data[0]

                if Msg_key=='Error':
                    
                    Msg_key,data=parseNMEA(l,self.verbose)
                    
                    if Msg_key=='Error':
                        self.corrupt.append(l)
                        # print("Message {:s} not in NMEA message list".format(Msg_key))
                        try: 
                            self.dropped.index(Msg_key)
                        except ValueError: 
                            self.dropped.append(Msg_key)
                        continue
                    
                try:
                  j=self.MSG_list.index(Msg_key)
                  if len(data)!=len(self.keyList[j]):
                      self.corrupt.append(l)
                      continue
                except ValueError:
                    if self.verbose:
                        print("Message {:s} not in NMEA message list. Dropping it.".format(Msg_key))
                    continue
                
                
                getattr(self,Msg_key+'List').append(data)
                
                
                
            elif l.find('D ')!=-1:
                self.LaserList.append(parseLaser(l,self.verbose)+(self.ToW,) )
            elif l.find('Cal')!=-1:
                self.CalList.append(parseCalibration(l,self.verbose)+(self.ToW,) )    
            elif l.find('Temp')!=-1:
                self.TempList.append(parseTemp(l,self.verbose)+(self.ToW,) ) 
            elif l.find('VBat')!=-1:
                self.VBatList.append((parseVBat(l,self.verbose),self.ToW,) ) 
            elif l.find('SDwrite')!=-1:
                self.SDwriteList.append((self.ToW,))  
            
            else:
                self.other.append(l)
        
        
        file.close()
        
        # drop  laser points with ToW=0        
        try:
            if droplaserTow0:
                for j,l in enumerate(self.LaserList):
                    if l[-1]!=0:
                        break
                self.LaserList=self.LaserList[j:]
        except UnboundLocalError: 
            print(" UnboundLocalError:! Continuing withou dropping TOW=0")
            
            
        for msg in (self.MSG_list):
            setattr(self,msg,MSG_type(msg) )
            if len(getattr(self,msg+'List'))>0:
                values=np.array(getattr(self,msg+'List'))
                keys=self.keyList[self.MSG_list.index(msg)]
                # print(keys)
                getattr(self,msg).addData(keys,values)
            
            # delattr(self,msg+'List')
            
        print("Total lines read: ", i)   
        
        
        # get starting point
        self.TOW0=get_TOW0(self)
        
        # extract single temperature sensors
        for j in range(1,4):
            i=self.Temp.sensor==j
            if np.any(i):
                setattr(self.Temp,f'TOW{j:d}',self.Temp.TOW[i])
                setattr(self.Temp,f'T{j:d}',self.Temp.T[i])
                setattr(self.Temp,f't{j:d}',self.Temp.TOW[i]-self.TOW0)
        
        # correct h with angles from INS
        if correct_Laser:
                self.corr_h_laser()
        
        # parse PGPSP status
        try:
            self.PGPSP.fix=(np.array(self.PGPSP.status,dtype=int) & 0x000001F00 )>>8
        except AttributeError:
            None
            
            
    def corr_h_laser(self):
        """
        correct elevation with angles from INS
        """   
        try:
            i=self.PINS1.TOW.searchsorted( self.Laser.TOW)
            j=np.array(i)-1
            
            pitch=self.PINS1.pitch[j]+(self.PINS1.pitch[i]-self.PINS1.pitch[j])/(self.PINS1.TOW[i]-self.PINS1.TOW[j])*(self.Laser.TOW-self.PINS1.TOW[j])
            roll=self.PINS1.roll[j]+(self.PINS1.roll[i]-self.PINS1.roll[j])/(self.PINS1.TOW[i]-self.PINS1.TOW[j])*(self.Laser.TOW-self.PINS1.TOW[j])
            
            roll-=self.roll0
            pitch-=self.pitch0
            # roll*=self.c_roll
            # pitch*=self.c_pitch
            
            self.Laser.pitch=pitch
            self.Laser.roll=roll
            self.Laser.h_corr=self.Laser.h*(np.cos(pitch)*np.cos(roll))-self.distCenter*np.sin(pitch)
            self.Laser.keys.extend(['h_corr','roll','pitch'])
           
        except Exception as e: 
            print(e)
            print('Failed to correct Laser elevation')
            try:
                self.Laser.h_corr=np.zeros_like(self.Laser.h)
            except AttributeError:
                print('Laser data not found.')
                
                


    def plot_att(self,ax=[]):
        plot_att(self,ax=ax)
        
    def plot_elevation_time(self,ax=[],title=[]):
        plot_elevation_time(self,ax=ax,title=title)

    def plot_longlat(self,z='elevation',ax=[],cmap= cm.batlow):
        plot_longlat(self,z=z,ax=ax,cmap=cmap)
        
    def plot_map(self,z='elevation',ax=[],cmap= cm.batlow):
        plot_map(self,z=z,ax=ax,cmap=cmap)

    def plot_mapOSM(self,z='TOW',ax=[],cmap= cm.batlow,title=[],extent=[]):
        plot_mapOSM(self,z=z,ax=ax,cmap= cmap,title=title,extent=extent)



    def subset(self, timelim=[],timeformat='s'):
        """
        Parameters
        ----------
        timelim : List, optional
            limits in time. The default is [].
        timeformat : string, optional
            units of time limits. Either 'TOW' or 's' (relative to start of data).

        Returns
        -------
        data2 : UBX2data object with subset of data into time limits
            

        """
        d=self.PINS1
        if timeformat=='TOW':
              lim=d.TOW.searchsorted(timelim)
              TOW_lim=timelim
        elif timeformat=='s':
              lim=((d.TOW -d.TOW[0]) ).searchsorted(timelim)
              TOW_lim=d.TOW[lim]
        else:
            print('timeformat not valid')
            return 0
  
        data2=INSLASERdata(self.filepath,name=self.name,load=False)

        for attr in (self.MSG_list):
            # print(attr)
            try:   
                msg_data=getattr(self,attr)
                lim2=msg_data.TOW.searchsorted(TOW_lim)
            except Exception as e: 
                    print(e)
                    print(attr)
                    # continue
            d2=MSG_type(attr)
            
            for a in msg_data.__dict__.keys():
                # print('\t',a)
                try:
                    setattr(d2,a,np.array(getattr(msg_data, a)[lim2[0]:lim2[1]] ))
                except:
                    setattr(d2,a,getattr(msg_data, a))
            setattr(d2,'len',lim2[1]-lim2[0])
            setattr(data2,attr,d2)
        return data2
        
# %% #########function definitions #############

def loadDataHeader(filepath):
    """
    Read data from header of LEM INS__.csv file. Lines must strat with # and contain one = to be parsed. Comments after symbol &.

    Parameters
    ----------
    filepath : TYPE
        Path file.

    Raises
    ------
    FileNotFoundError
        DESCRIPTION.

    Returns
    -------
    values : dictionary
        Dictionary with values from file header

    """
    if not os.path.isfile(filepath) :
        print("File not found!!!")
        raise FileNotFoundError('File: {:s}'.format(filepath))
    
    
    print('Reading header file: ',filepath)
    print('----------------------------')
    file = open(filepath, 'rt')
    
    
    i=0
    values={}
    for l in file:
        i+=1
        if l[0]!='#':
            continue
        
        elif l.find('=')!=-1:
            var,val=l[1:].strip().split('=')
            var=var.replace(' ', '_')
            
            try:
                val=float(val)
            except ValueError:
                try:
                    if len(val.strip().split(','))>1:
                        val=np.array(val.strip().split(',')[:-1],dtype=float)
                except ValueError:
                    continue
            values[var]=val
            
            
        elif l.find('###Data###')!=-1:
            # print("Found end header")
            break

    file.close()
    
    return values




def parseNMEA(l,verbose=True):
    """
    l: string with data
    
    
    
    """
    
    Msg_key=l[l.find('$')+1:l.find(',') ]
    start=l.find('$'+Msg_key)
    end=l.find('*')

    # Check if message is valid
    if (start!=-1 and end!=-1 and start<end and chksum_nmea(l[start:end+3])): 

      try:
          a=np.array(l[start+len(Msg_key)+2:end].split(','),dtype=float)

      except:
          try:
              a=np.array(l[start+len(Msg_key)+2:end].split(','))
          except:
              if verbose: 
                  print('Coud not parse valid NMEA string:', l)  
              return 'Error',l 
    else:
        if verbose:   
            print('Coud not parse valid NMEA string:', l)  
        return 'Error',l             

    return Msg_key,a


def parseNMEAfloat(l,verbose=True):
    """
    l: string with data
    
    
    
    """
    
    Msg_key=l[l.find('$')+1:l.find(',') ]
    start=l.find('$'+Msg_key)
    end=l.find('*')

    # Check if message is valid
    if (start!=-1 and end!=-1 and start<end and chksum_nmea(l[start:end+3])): 

      try:
          a=np.array(l[start+len(Msg_key)+2:end].split(','),dtype=float)

      except:
              # print('Coud not parse valid NMEA string:', l)  
              return 'Error',l 
    else:
        # if verbose: print('Coud not parse string:', l)    
        return 'Error',l             

    return Msg_key,a


def parseCalibration(l,verbose=True):
    """
    l: string with data
    
    return calibration ID and On/Off status 
        case on:    ID is the calibration coil state (code sent to I2C switch)
        case off:   ID is a flag and 0 for switching off, 1 fro starting calibration and 2 for calibration process end
    
    """
    
    on=False
    ID=0

    
    if l[:5]=='CalOn': 
    
        a=l.split()
        try:
                on= True
                ID=int(a[1])
        except:
                on= False
                ID=np.nan
                if verbose: 
                    print('Coud not parse string:', l)
    elif l[:6]=='CalOff': 
            try:
                on= False
                ID=0
            except:
                on= False
                ID=np.nan
                if verbose: 
                    print('Coud not parse string:', l)
    elif l[:6]=='CalEnd': 
            try:
                on= False
                ID=2
            except:
                on= False
                ID=np.nan
                if verbose: 
                    print('Coud not parse string:', l)
    elif l[:8]=='CalStart': 
            try:
                on= False
                ID=1
            except:
                on= False
                ID=np.nan
                if verbose: 
                    print('Coud not parse string:', l)
    else:
        ID= np.nan
        on= False
        if verbose: 
            print('Coud not parse string:', l)                 


    return on,ID

def parseTemp(l,verbose=True):
    """
    l: string with data
    
    return calibration ID and Temperature in Celsius
    
    """
    
    Temp=99
    ID=0
    
    if l[:4]=='Temp': 
    
        a=l.split()
        try:
                Temp= float(a[1])
                ID=int(a[0][4:])
        except Exception as e: 
                print(e)
                Temp= np.nan
                ID=np.nan
                if verbose: 
                    print('Coud not parse string:', l)
    
    else:
        ID= np.nan
        Temp= np.nan
        if verbose: 
            print('Coud not parse string:', l)                 


    return Temp,ID

def parseVBat(l,verbose=True):
   
    """
    l: string with data
    
    return Volatage in Volts
    
    """
    
    V=0
    
    if l[:4]=='VBat': 
    
        a=l.split()
        try:
                V= float(a[1])
        except:
                V= np.nan
                if verbose: 
                    print('Coud not parse string:', l)
    
    else:
        V= np.nan
        if verbose: 
            print('Coud not parse string:', l)               

    return V

def parseLaser(l,verbose=True):
    """
    l: string with data
    TOW: time of Week to append to message data
    
    return elevation, signal quality, temperature 
    
    """
    
    h=[]
    T=[]
    signQ=[]
    TOW=[]

    i=0
    j=0
    t=0
    length=len(l)
    
    if l[0]=='D': 
    
        a=l.split()
        if len(a)==2:
            signQ= np.nan
            T= np.nan
            try:
                h= float(a[1])
            except:
                h= np.nan
                if verbose: 
                    print('Coud not parse string:', l)
        else:
            error=0
            try:
                h= float(a[1])
            except:
                h= np.nan
                error=1
            try:
                signQ= float(a[2])
            except:
                signQ= np.nan
                error=1
            try:
                T= float(a[3]) 
            except:
                T= np.nan 
                error=1
            if error:
                if verbose: 
                    print('Coud not parse string:', l)
    else:
        h= np.nan
        signQ= np.nan
        T= np.nan 
        if verbose: 
            print('Coud not parse string:', l)                 


    return h,signQ,T
    
def chksum_nmea(sentence):
    # From: http://doschman.blogspot.com/2013/01/calculating-nmea-sentence-checksums.html
   
    # This is a string, will need to convert it to hex for 
    # proper comparsion below
    end=sentence.find("*")
    cksum = sentence[end+1:end+3]
    
    # String slicing: Grabs all the characters 
    # between '$' and '*' and nukes any lingering
    # newline or CRLF
    chksumdata = re.sub("(\n|\r\n)","", sentence[sentence.find("$")+1:sentence.find("*")])
    
    # Initializing our first XOR value
    csum = 0 
    
    # For each char in chksumdata, XOR against the previous 
    # XOR'd char.  The final XOR of the last char will be our 
    # checksum to verify against the checksum we sliced off 
    # the NMEA sentence
    
    for c in chksumdata:
       # XOR'ing value of csum against the next char in line
       # and storing the new XOR value in csum
       csum ^= ord(c)
    
    # Do we have a validated sentence?
    try:
        if hex(csum) == hex(int(cksum, 16)):
           return True
    except  ValueError:
        print('Invalid checksum: ',cksum)
    return False




def get_TOW0(data,iStart=2,PIN_start=8,dt=9000):
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
    try:
        if data.PSTRB.pin[iStart]!=PIN_start:
            
            if sum(data.PSTRB.pin==PIN_start)==0:
                try:
                    if len(data.PSTRB.pin)==1:
                        iStart2=0
                    else:
                        iStart2= np.where(np.diff(data.PSTRB.TOW)>dt)[0][0]
                    start =data.PSTRB.TOW[iStart2]/1000
                    print('TW0: {:f}, PIN used for start: {:f} ,iStart: {:f}'.format(start, data.PSTRB.pin[iStart2],iStart2))
                
                except IndexError:
                   raise IndexError(f'No correct time stamp was found. All delta time <{dt:d} ms.', np.diff(data.PSTRB.TOW))
                       
            else:    
                raise ValueError(f'Timestamp {iStart} was not on pin {PIN_start:d} of INS.', iStart)
        else:
            start =data.PSTRB.TOW[iStart]/1000
            print('PIN used for start: {:.0f} ,iStrat: {:.0f}'.format(data.PSTRB.pin[iStart],iStart))
            
    except (AttributeError, IndexError) :
            start =data.PINS1.TOW[0]
            print('No PSTRB timestamps were found. Used first PINS1 datapoint as start time!')
    return start

def get_t0(data,iStart=2,PIN_start=8,dt=9000):
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
    TOW=get_TOW0(data,iStart=iStart,PIN_start=PIN_start,dt=dt)
    
    return gps_datetime(data.PINS1.GPSWeek[0], TOW, leap_seconds=18)


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

# %% plot functions
    
def plot_GPSquality(data,title=''):
        
    fig,[ax,ax2,ax3]=pl.subplots(3,1,sharex=True)
    ax.plot(data.PGPSP.TOW-data.TOW0,data.PGPSP.fix)
    ax.set_ylabel('GPS fix')
    ax2.plot(data.PGPSP.TOW-data.TOW0,data.PGPSP.hAcc,label='horizontal')
    ax2.plot(data.PGPSP.TOW-data.TOW0,data.PGPSP.vAcc,label='vertical')
    ax2.legend()
    ax2.set_ylabel('accuracy (m)')
    ax3.plot(data.PGPSP.TOW-data.TOW0,data.PGPSP.cnoMean)
    ax3.set_ylabel('meanCarrierToNoise ratio')
    ax3.set_xlabel('Time (s)')

    if title!='none':
        ax.set_title(title)
        
    return fig


def check_data(data):
    """
    Check data and produce some plots
    
    Input:
    --------------
    data        Member of class UBXdata
    
    """
    
    print('\n\n###############################\n-----------------------------\n',
          data.name,
          '\n----------------------------\n###############################\n')
    
    for attr in data.MSG_list:
        try:   
            d=getattr(data,attr)
            print('\n',attr,'\n----------------------------')
            print('Length:')
            print(d.len)
            print('Time intervall (s):')
            print((d.TOW[:5]-d.TOW[0]) )
        except Exception as e: 
                print(e)
            
            
    print('\nOthers: \n----------------------------')   
    print('Length:')
    try:
        print(len(data.other))
    except:
        print('no data')
    # for c in data.other[:10]: 
    #     print(c) 
    
    print('\nCorrupted: \n----------------------------')   
    try:
        print(len(data.corrupt))
    except:
        print('no data')
    # for c in data.corrupt[:10]: 
    #     print(c)      
        
    try:
        data.plot_att()
    except AttributeError:
        print('no attitude messages found')
        

    try:
        if len(data.Laser.h)==0:
            raise AttributeError
                
        pl.figure()
        ax=pl.subplot(111)
        ax2=ax.twinx()
        ax.set_ylabel('Laser h (m)')
        ax2.set_ylabel('GNSS Delta_h (m)')
        ax.set_xlabel('time (ms)')
    
        ax.plot(data.Laser.TOW-data.PINS1.TOW[0],data.Laser.h,'--xk',label='Laser')
        # ax.plot(data.Laser.TOW2-data.PINS1.TOW[0],data.Laser.h,'--ob',label='Laser, time2')
        ax2.plot((data.PINS1.TOW-data.PINS1.TOW[0]),data.PINS1.elevation -(data.PINS1.elevation[0] -data.Laser.h[0]),'+:r',label='GPS elevation')
        
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc=0)   
        pl.title(data.name)
    except AttributeError:
        print('No laser data found')






def plot_att(data,ax=[],title='',heading=True):
    if ax==[]:
        fig=pl.figure()
        ax=pl.subplot(111)
    else:
        pl.sca(ax)
    
    d=data.PINS1
    
    if heading:
            ax2=ax.twinx()    
    

    ax.plot((d.TOW-d.TOW[0]),np.degrees(d.pitch),'o-r',label='pitch')
    ax.plot((d.TOW-d.TOW[0]),np.degrees(d.roll),'x-k',label='roll')
    if heading:
        ax2.plot((d.TOW-d.TOW[0]),np.degrees(d.heading),'x-b',label='heading')

    
    
    ax.set_ylabel('pitch/roll (deg)')
    ax.set_xlabel('time (s)')
    if heading:
        ax2.set_ylabel('heading (deg)')
        # ask matplotlib for the plotted objects and their labels
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc=0)   
    else:
        ax.legend(loc=0)
        
    if title=='':
        pl.title(data.name)
    elif title!='none':
        pl.title(title)
    
    
def plot_att_laser(data,ax=[],title='',heading=True):
    if ax==[]:
        fig=pl.figure()
        ax=pl.subplot(111)
    else:
        pl.sca(ax)
        fig=pl.gcf()
    
    d=data.PINS1
    
    if heading:
            ax2=ax.twinx()        
    

    ax.plot((d.TOW-d.TOW[0]),np.degrees(d.pitch),'o-r',label='pitch')
    ax.plot((d.TOW-d.TOW[0]),np.degrees(d.roll),'x-k',label='roll')
    if heading:
        ax2.plot((d.TOW-d.TOW[0]),np.degrees(d.heading),'x-b',label='heading')
    
    
    ax.set_ylabel('pitch/roll (deg)')
    ax.set_xlabel('time (s)')
    if heading:
        ax2.set_ylabel('heading (deg)')
        # ask matplotlib for the plotted objects and their labels
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc=0)   
        ax2.yaxis.label.set_color('b')
        ax2.tick_params(axis='y', colors='b')
    else:
        ax.legend(loc=0)
    
    
    try:
        if len(data.Laser.h)==0:
            raise AttributeError
                
        ax3=ax.twinx()
        fig.subplots_adjust(right=0.75)
        ax3.spines['right'].set_position(("axes", 1.2))
        ax3.yaxis.label.set_color('g')
        ax3.tick_params(axis='y', colors='g')
        ax3.set_ylabel('Laser h (m)')
        ax3.plot((data.Laser.TOW-d.TOW[0]),data.Laser.h,':og',label='Laser')
        lines3, labels3 = ax3.get_legend_handles_labels()
    except AttributeError:
        print('No laser data found')
        lines3=[]
        labels3=[]    
    
    
    if title=='':
        pl.title(data.name)
    elif title!='none':
        pl.title(title)
    
def plot_elevation_time(data,ax=[],title=[]):
    if ax==[]:
        fig=pl.figure()
        ax=pl.subplot(111)
    else:
        pl.sca(ax)
    
    d=data.PINS1

    
    ax.plot((d.TOW-d.TOW[0]),d.elevation,'o-r',label='elevation')
    ax.set_ylabel('elevation (m a.s.l.)')
    ax.set_xlabel('time (s)')
   
    if title:
        pl.title(title)
    pl.tight_layout()
    
    
def plot_longlat(data,z='elevation',ax=[],cmap= cm.batlow):
    if ax==[]:
        fig=pl.figure()
        ax=pl.subplot(111)
    else:
        pl.sca(ax)
    
    d=data.PINS1
    
    ax2=ax.twinx()    
    c=getattr(d,z)
    if z=='elevation':
        label='elevation (m a.s.l.)'
        c=c
    elif z=='TOW':
        label='time (s)'
        c-=c[0]
        c=c
    else:
        label=z
    lat=np.array([distance.distance((d.lat[0],d.lon[0]),(x,d.lon[0])).m for x in d.lat ])
    lon=np.array([distance.distance((d.lat[0],d.lon[0]),(d.lat[0],x)).m for x in d.lon ])
    
    im=ax.scatter(lon,lat,c=c,cmap=cmap,marker='x')
    c=pl.colorbar(im, label=label)
    ax.invert_xaxis()
    ax.set_ylabel('N (m)')
    ax.set_xlabel('W (m)')
    
    pl.title(data.name)
    pl.tight_layout()

        

def plot_summary(data,extent,cmap=cm.batlow,heading=True,title=''):
    
    fig=pl.figure(figsize=(8,10))
    spec = fig.add_gridspec(ncols=1, nrows=9)
    
   
    try:
        cimgt.OSM.get_image = image_spoof # reformat web request for street map spoofing
        osm_img = cimgt.OSM() # spoofed, downloaded street map
        ax0 = fig.add_subplot(spec[0:3, 0],projection=osm_img.crs)
        data.plot_mapOSM(z='TOW',extent=extent,ax=ax0, cmap=cmap  )
    except ValueError:
        print('Can not download map!! No internet??')
    
    if len(title)==0:
        ax0.set_title(data.name.strip('.csv').strip('\\'))
    else:
        ax0.set_title(title)
        
    
    ax1 = fig.add_subplot(spec[3:5, 0])
    ax1.plot((data.PINS1.TOW-data.PINS1.TOW[0]) ,
              [distance.distance((data.PINS1.lat[0],data.PINS1.lon[0]),(data.PINS1.lat[0],x)).m*np.sign(x-data.PINS1.lon[0])  for x in data.PINS1.lon ],
              'x:',label='East-West')
    ax1.plot((data.PINS1.TOW-data.PINS1.TOW[0]) ,
              [distance.distance((data.PINS1.lat[0],data.PINS1.lon[0]),(x,data.PINS1.lon[0])).m*np.sign(x-data.PINS1.lat[0]) for x in data.PINS1.lat ],
              '+:',label='North-South')
    ax1.set_ylabel('Distance (m)')
    ax1.legend()
    
    ax2 = fig.add_subplot(spec[5:7, 0],sharex = ax1)
    ax2.plot((data.Laser.TOW-data.PINS1.TOW[0]) ,data.Laser.h, 'x:',label='original')
    ax2.plot((data.Laser.TOW-data.PINS1.TOW[0]) ,data.Laser.h_corr, '+:k',label='corrected')
    ax2.plot((data.PINS1.TOW-data.PINS1.TOW[0]) ,data.PINS1.elevation -(data.PINS1.elevation[0] -data.Laser.h[0]),'+:r',label='GPS elevation')
    ax2.set_ylabel('h_laser (m)')
    ax2.set_xlim(ax1.get_xlim())
    ax2.legend()
    
    ax3 = fig.add_subplot(spec[7:9, 0],sharex = ax1)
    plot_att(data,ax=ax3,title='none',heading=heading)
    pl.tight_layout()
    return fig

def plot_summary2(data,extent,cmap=cm.batlow,heading=True):
    
    fig=pl.figure(figsize=(8,10))
    spec = fig.add_gridspec(ncols=1, nrows=9)
    
    ax1 = fig.add_subplot(spec[3:5, 0])
    ax1.plot((data.PINS1.TOW-data.PINS1.TOW[0]) ,
              [distance.distance((data.PINS1.lat[0],data.PINS1.lon[0]),(data.PINS1.lat[0],x)).m for x in data.PINS1.lon ],
              'x:',label='East-West')
    ax1.plot((data.PINS1.TOW-data.PINS1.TOW[0]) ,
              [distance.distance((data.PINS1.lat[0],data.PINS1.lon[0]),(x,data.PINS1.lon[0])).m for x in data.PINS1.lat ],
              '+:',label='North-South')
    ax1.set_ylabel('Distance (m)')
    ax1.legend()
    
    ax2 = fig.add_subplot(spec[5:7, 0],sharex = ax1)
    ax2.plot((data.Laser.TOW-data.PINS1.TOW[0]) ,data.Laser.h, 'x:',label='original')
    ax2.plot((data.Laser.TOW-data.PINS1.TOW[0]) ,data.Laser.h_corr, '+:k',label='corrected')
    ax2.plot((data.PINS1.TOW-data.PINS1.TOW[0]) ,data.PINS1.elevation -(data.PINS1.elevation[0] -data.Laser.h[0]),'+:r',label='GPS elevation')
    ax2.set_ylabel('h_laser (m)')
    ax2.set_xlim(ax1.get_xlim())
    ax2.legend()
    
    ax3 = fig.add_subplot(spec[7:9, 0],sharex = ax1)
    plot_att(data,ax=ax3,title='none',heading=heading)
    pl.tight_layout()
    return fig



def plot_h_LatLon(data):
    
    lat=data.PINS1.lat[ data.PINS1.TOW.searchsorted(data.Laser.TOW )]
    lon=data.PINS1.lon[ data.PINS1.TOW.searchsorted(data.Laser.TOW )]
        
    fig,[ax,ax2]=pl.subplots(2,1,sharex=True)
    ax.set_title('name:{:s}'.format(data.name))
    ax.plot(lat, data.Laser.h,'--k')
    
    ax.set_ylabel('h Laser (m)')
    pl.xlabel('time (s)')
    # ax.set_ylim(0.06,1)
    
    # ax3=ax2.twinx()
    ax2.plot(lat,lon,'x')
    ax2.set_ylabel('lon')
    ax2.set_xlabel('lat')
    # ax2.set_ylim(-0.04,0.3)
    pl.tight_layout()

    return fig


def plot_h_LatLon2(data):
    
    lat=data.PINS1.lat[ data.PINS1.TOW.searchsorted(data.Laser.TOW )]
    lon=data.PINS1.lon[ data.PINS1.TOW.searchsorted(data.Laser.TOW )]
    t=data.Laser.TOW-data.PINS1.TOW[0]
    
    fig,[ax,ax2]=pl.subplots(2,1,sharex=True)
    ax.set_title('name:{:s}'.format(data.name))
    ax.plot(t, data.Laser.h,'--k')
    
    ax.set_ylabel('h Laser (m)')
    pl.xlabel('time (s)')
    # ax.set_ylim(0.06,1)
    
    ax3=ax2.twinx()
    ax2.plot(t,lat,'xb')
    ax3.plot(t,lon,'xg')
    ax3.set_ylabel('lon')
    ax2.set_ylabel('lat')
    ax2.set_xlabel('time (s)')

    # ax2.set_ylim(-0.04,0.3)
    pl.tight_layout()

    return fig



def laser_correction(data,show_corr_angles=0,GPS_h=False,heading=False):
    
    fig, [ax2,ax3] = pl.subplots(2, 1, figsize=(8, 8), sharex=True, sharey=False)

    ax2.plot((data.Laser.TOW-data.PINS1.TOW[0]) ,data.Laser.h, 'x:',label='original')
    ax2.plot((data.Laser.TOW-data.PINS1.TOW[0]) ,data.Laser.h_corr, '+:',label='corrected')
    if GPS_h:
        ax2.plot((data.PINS1.TOW-data.PINS1.TOW[0]) ,data.PINS1.elevation -(data.PINS1.elevation[0] -data.Laser.h[0]),'+:r',label='GPS elevation')
    ax2.set_ylabel('h_laser (m)')
    ax2.legend()

    plot_att(data,ax=ax3,title='none',heading=heading)
    
    if show_corr_angles:
        ax3.plot((data.Laser.TOW-data.PINS1.TOW[0]) ,np.degrees(data.Laser.pitch), 'x:',label='pitch laser')
        ax3.plot((data.Laser.TOW-data.PINS1.TOW[0]) ,np.degrees(data.Laser.roll), 'x:',label='roll laser')
        ax3.legend(loc=0)
    return fig

def laser_correction_superimposed(data,GPS_h=False):
    
    fig, ax2 = pl.subplots(1, 1, figsize=(8, 8))

    ax2.plot((data.Laser.TOW-data.PINS1.TOW[0]) ,data.Laser.h, 'x:',label='original')
    ax2.plot((data.Laser.TOW-data.PINS1.TOW[0]) ,data.Laser.h_corr, '+:',label='corrected')
    if GPS_h:
        ax2.plot((data.PINS1.TOW-data.PINS1.TOW[0]) ,data.PINS1.elevation -(data.PINS1.elevation[0] -data.Laser.h[0]),'+:r',label='GPS elevation')
    ax2.set_ylabel('h_laser (m)')
    ax2.legend(loc=2)

    ax3=ax2.twinx()
    ax3.plot((data.Laser.TOW-data.PINS1.TOW[0]) ,data.Laser.pitch, 'x:k',label='pitch laser')
    ax3.plot((data.Laser.TOW-data.PINS1.TOW[0]) ,data.Laser.roll, '+:r',label='roll laser')
    ax3.legend(loc=1)
    ax3.set_ylabel('Angle (rad)')
    ax2.grid()
    return fig








#%%  Sync Laser and GPX data from UAV

def GPXToPandas(gpx_path):
    with open(gpx_path) as f:
        gpx = gpxpy.parse(f)
    
    # Convert to a dataframe one point at a time.
    points = []
    for segment in gpx.tracks[0].segments:
        for p in segment.points:
            points.append({
                'time': p.time,
                'lat': p.latitude,
                'lon': p.longitude,
                'elevation': p.elevation,
            })
    df=pd.DataFrame.from_records(points)
    df.time=df.time.dt.tz_convert('UTC')
    
    # interpolate times if overlapping
    for i in df.index[1:-1]:
        if df.time[i]-df.time[i-1]==timedelta(seconds=0):
            df.loc[i,'time']=df.time[i]+(df.time[i+1]-df.time[i])/2
    
    return df


def mergeByTime(t1,x,t2, method='linInterpol', maxDelta=0,dT=0):
        """
        Find values of t1 in t2 and return the corresponding values x. For missing data different strategies can be chosen.

        Input:
        ------

        t1:         time serie 1
        x:          values corresponding to time series 1
        t2:         time serie 2
        method:     Method to use for missing data.
                        exact:          Keep just
                        linInterpol:    linear interploation between two values, if time interval < maxDelta [s]
                        nearest:        get nearest value. If time lag is > maxDelta return NAN.
                        nearest_2:        Fill up all values of x in x2 in the position t2 corresponding to t1. (loop trough t1). If time lag is > maxDelta return NAN. Nearest value is returned.

        maxDelta:   maximum time difference in [s] of interpolation or nearest
        dT:         Time shift to align timeseries


        **kwargs:   argments to be passed to np.genfromtxt

        Output:
        -------
        x(t1==t2):    time array in datetime format and data array.


        """
        t2=np.array(t2)
        t1=np.array(t1)
        x=np.array(x)
        
        # check
        if len(t1)!=len(x):
                raise ValueError('t1 and x have not the same length!')


        x2=np.zeros([len(t2),1])
        # x2=np.zeros([len(t2),1],dtype=x.dtype)


        # add DeltaT
        try: #try to use datetime format
            t1=t1+pd.Timedelta( seconds=dT)
        except:
            t1+=dT

        if method=='exact':
                # return    np.where(t1[t1.searchsorted(t2)]==t2,x,np.nan)


                for i,a in enumerate(t2):
                        j=np.searchsorted(t1,a)

                        if a>t1[-1] or a<t1[0]:
                                x2[i]=np.nan
                        elif t1[j]==a:

                                x2[i]=x[j]
                        else:
                                x2[i]=np.nan
                return x2


        elif method=='linInterpol':
                for i,a in enumerate(t2):

                        j=np.searchsorted(t1,a)

                        if a>t1[-1] or a<t1[0]:
                                x2[i]=np.nan
                        elif t1[j]==a:

                                x2[i]=x[j]
                        else:

                                try: #try to use datetime format
                                        d=(t1[j]-t1[j-1]).total_seconds()
                                        d2=(a-t1[j-1]).total_seconds()
                                except:
                                        d=(t1[j]-t1[j-1])
                                        d2=(a-t1[j-1])

                                if d>maxDelta:
                                        x2[i]=np.nan

                                else:
                                        x2[i]=x[j-1]+d2/d*(x[j]-x[j-1])

                return x2


        elif method=='nearest':
                try: #try to used datetime
                        (t1[0]-t2[0])<timedelta(seconds=maxDelta)
                        delta=timedelta(seconds=maxDelta)
                except TypeError:
                       delta =maxDelta


                for i,a in enumerate(t2):

                        j=np.searchsorted(t1,a)

                        if j==len(t1): # j exeeding index max
                                j-=1


                        if a>=t1[j]:
                                d=a-t1[j]

                                if j<len(t1)-1:
                                        d2=t1[j+1]-a

                                        if d>delta and d2>delta:
                                                x2[i]=np.nan
                                        elif d>d2:
                                                x2[i]=x[j+1]
                                        else:
                                                x2[i]=x[j]

                                else:
                                        if d>delta :
                                                x2[i]=np.nan
                                        else:
                                                x2[i]=x[j]


                        else:
                                d=t1[j]-a

                                if j>0:
                                        d2=a-t1[j-1]

                                        if d>delta and d2>delta:
                                                x2[i]=np.nan
                                        elif d>d2:
                                                x2[i]=x[j-1]
                                        else:
                                                x2[i]=x[j]

                                else:
                                        if d>delta :
                                                x2[i]=np.nan
                                        else:
                                                x2[i]=x[j]

                return x2


        elif method=='nearest_2':
                try: #try to used datetime
                        (t1[0]-t2[0])<timedelta(seconds=maxDelta)
                        delta=[timedelta(seconds=maxDelta) for i in range(len(x2)) ]

                except TypeError:
                       delta = np.ones([len(x2),1])*maxDelta

                x2*=np.nan

                for i,a in enumerate(t1):

                        j=np.searchsorted(t2,a)


                        if j==len(t2): # j exeeding index max
                                d2=a-t2[j-1]
                                if d2<delta[j-1]:
                                        delta[j-1]=d2
                                        x2[j-1]=x[i]

                        elif j==0:
                                d=t2[j]-a
                                if d<delta[j]:

                                        delta[j]=d
                                        x2[j]=x[i]

                        else:
                                d=t2[j]-a
                                d2=a-t2[j-1]

                                if d>d2:
                                        if d2<delta[j-1]:
                                                delta[j-1]=d2
                                                x2[j-1]=x[i]
                                else:
                                        if d<delta[j]:
                                                delta[j]=d
                                                x2[j]=x[i]



                return x2

        else:
                raise ValueError('Unnown method! ')





# %% plot on map
def plot_map(data,z='elevation',ax=[],cmap= cm.batlow,title=[],timelim=[],timeformat='ms'):
    
    if ax==[]:
        fig=pl.figure()
        ax = pl.axes(projection=ccrs.PlateCarree())
    else:
        ax.axes(projection=ccrs.PlateCarree())
    
    
    
    d=data.PINS1   
    if not timelim==[]:
        if timeformat=='ms':
            lim=d.TOW.searchsorted(timelim)
        if timeformat=='s':
            lim=((d.TOW -d.TOW[0]) ).searchsorted(timelim)
            
            
            
    c=getattr(d,z)[lim[0]:lim[1]] # get data color plot
    
    if z=='elevation':
        label='elevation (m a.s.l.)'
        c=c 
    elif z=='TOW':
        label='time (s)'
        c-=c[0]
        c=c 
    else:
        label=z
        
    ax.coastlines()
    im=ax.scatter(d.lon[lim[0]:lim[1]],d.lat[lim[0]:lim[1]],c=c,cmap=cmap,marker='x')
    c=pl.colorbar(im, label=label)
    
    lon_formatter = LongitudeFormatter(number_format='0.1f',degree_symbol='',dateline_direction_label=True) # format lons
    lat_formatter = LatitudeFormatter(number_format='0.1f',degree_symbol='') # format lats
    ax.xaxis.set_major_formatter(lon_formatter) # set lons
    ax.yaxis.set_major_formatter(lat_formatter) # set lats
    
    if title:
        pl.title(title)
    pl.tight_layout()  
    


def image_spoof(self, tile): # this function pretends not to be a Python script
        url = self._image_url(tile) # get the url of the street map API
        req = Request(url) # start request
        req.add_header('User-agent','Anaconda 3') # add user agent to request
        fh = urlopen(req) 
        im_data = io.BytesIO(fh.read()) # get image
        fh.close() # close url
        img = Image.open(im_data) # open image with PIL
        img = img.convert(self.desired_tile_form) # set image format
        return img, self.tileextent(tile), 'lower' # reformat for cartopy
    
    
def plot_mapOSM(data,z='elevation',ax=[],cmap= cm.batlow,title=[], extent=[]):
    """
    Plot data (z) on Open Street Map layer.
    
    Parameters
    ----------
    data : TYPE
        DESCRIPTION.

    z : TYPE, optional
        DESCRIPTION. The default is 'elevation'.
    ax : TYPE, optional
        DESCRIPTION. The default is [].
    cmap : TYPE, optional
        DESCRIPTION. The default is cm.batlow.
    title : TYPE, optional
        DESCRIPTION. The default is [].
    extent:     limits of map. In [[lon,lon,lat,lat]]

    Returns
    -------
    None.

    """
    cimgt.OSM.get_image = image_spoof # reformat web request for street map spoofing
    osm_img = cimgt.OSM() # spoofed, downloaded street map
    
    if ax==[]:
        fig=pl.figure()
        ax = pl.axes(projection=osm_img.crs)
  
    
    # prepare data
    d=data.PINS1     
    c=np.array( getattr(d,z))
    if z=='elevation':
        label='elevation (m a.s.l.)'
        c=c 
    elif z=='TOW':
        label='time (s)'
        c-=c[0]
        c=c 
    else:
        label=z
    
    if extent==[]:
        dx=(np.max(d.lon)-np.min(d.lon))
        dy=(np.max(d.lat)-np.min(d.lat))
        W=np.max([dx,dy])*3/5
        center=[(np.max(d.lon)+np.min(d.lon))/2, (np.max(d.lat)+np.min(d.lat))/2]
        extent = [center[0]-W,center[0]+W, center[1]-W,center[1]+W] # Contiguous US bounds
        print(extent)
        
    # setup map
    ax.set_extent(extent) # set extents
    ax.set_xticks(np.linspace(extent[0],extent[1],3),crs=ccrs.PlateCarree()) # set longitude indicators
    ax.set_yticks(np.linspace(extent[2],extent[3],4)[1:],crs=ccrs.PlateCarree()) # set latitude indicators
    lon_formatter = LongitudeFormatter(number_format='0.4f',degree_symbol='',dateline_direction_label=True) # format lons
    lat_formatter = LatitudeFormatter(number_format='0.4f',degree_symbol='') # format lats
    ax.xaxis.set_major_formatter(lon_formatter) # set lons
    ax.yaxis.set_major_formatter(lat_formatter) # set lats
    ax.xaxis.set_tick_params()
    ax.yaxis.set_tick_params()
    
    scale = np.ceil(-np.sqrt(2)*np.log(np.divide((extent[1]-extent[0])/2.0,350.0))) # empirical solve for scale based on zoom
    scale = (scale<20) and scale or 19 # scale cannot be larger than 19
    ax.add_image(osm_img, int(scale))

    im=ax.scatter(d.lon,d.lat,c=c,cmap=cmap,marker='x',transform=ccrs.PlateCarree())
    c=pl.colorbar(im, label=label)
    
    if title:
        pl.title(title)
    pl.tight_layout()


def plot_mapOSM2(d,z='TOW',ax=[],cmap= cm.batlow,title=[], extent=[]):
    """
    Plot data (z) on Open Street Map layer.
    
    Parameters
    ----------
    data : TYPE
        DESCRIPTION.

    z : TYPE, optional
        DESCRIPTION. The default is 'elevation'.
    ax : TYPE, optional
        DESCRIPTION. The default is [].
    cmap : TYPE, optional
        DESCRIPTION. The default is cm.batlow.
    title : TYPE, optional
        DESCRIPTION. The default is [].
    extent:     limits of map. In [[lon,lon,lat,lat]]

    Returns
    -------
    None.

    """
    cimgt.OSM.get_image = image_spoof # reformat web request for street map spoofing
    osm_img = cimgt.OSM() # spoofed, downloaded street map
    
    if ax==[]:
        fig=pl.figure()
        ax = pl.axes(projection=osm_img.crs)
  
    
    # prepare data    
    c=np.array( getattr(d,z))
    if z=='elevation':
        label='elevation (m a.s.l.)'
        c=c 
    elif z=='TOW':
        label='time (s)'
        c-=c[0]
        c=c 
    else:
        label=z
    
    if extent==[]:
        dx=(np.max(d.lon)-np.min(d.lon))
        dy=(np.max(d.lat)-np.min(d.lat))
        W=np.max([dx,dy])*3/5
        center=[(np.max(d.lon)+np.min(d.lon))/2, (np.max(d.lat)+np.min(d.lat))/2]
        extent = [center[0]-W,center[0]+W, center[1]-W,center[1]+W] # Contiguous US bounds
        print(extent)
        
    # setup map
    ax.set_extent(extent) # set extents
    ax.set_xticks(np.linspace(extent[0],extent[1],3),crs=ccrs.PlateCarree()) # set longitude indicators
    ax.set_yticks(np.linspace(extent[2],extent[3],4)[1:],crs=ccrs.PlateCarree()) # set latitude indicators
    lon_formatter = LongitudeFormatter(number_format='0.4f',degree_symbol='',dateline_direction_label=True) # format lons
    lat_formatter = LatitudeFormatter(number_format='0.4f',degree_symbol='') # format lats
    ax.xaxis.set_major_formatter(lon_formatter) # set lons
    ax.yaxis.set_major_formatter(lat_formatter) # set lats
    ax.xaxis.set_tick_params()
    ax.yaxis.set_tick_params()
    
    scale = np.ceil(-np.sqrt(2)*np.log(np.divide((extent[1]-extent[0])/2.0,350.0))) # empirical solve for scale based on zoom
    scale = (scale<20) and scale or 19 # scale cannot be larger than 19
    ax.add_image(osm_img, int(scale))

    im=ax.scatter(d.lon,d.lat,c=c,cmap=cmap,marker='x',transform=ccrs.PlateCarree())
    c=pl.colorbar(im, label=label)
    
    if title:
        pl.title(title)
    pl.tight_layout()

def getextent(data):
    minlon=data.PINS1.lon.min()
    maxlon=data.PINS1.lon.max()
    minlat=data.PINS1.lat.min()
    maxlat=data.PINS1.lat.max()
    dlat=maxlat-minlat
    dlon=maxlon-minlon
    return minlon-dlon/10,maxlon+dlon/10,minlat-dlat/10,maxlat+dlat/10


def getextent2(data):
    minlon=data.lon.min()
    maxlon=data.lon.max()
    minlat=data.lat.min()
    maxlat=data.lat.max()
    dlat=maxlat-minlat
    dlon=maxlon-minlon
    return minlon-dlon/10,maxlon+dlon/10,minlat-dlat/10,maxlat+dlat/10