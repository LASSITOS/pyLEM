# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 11:07:51 2022

@author: AcCap
"""

import serial
import time
import os
import sys


BAUDRATE=460800
COMPORT='COM21'


def main(path='./',file='',list=False, COM=COMPORT,baud=BAUDRATE,):  # Read the first sector of the first disk as example.
    
    if COM[:3]!='COM':
        try:
            COM='COM{:d}'.format(int(COM))
        except Exception as e: 
            print("error. COM value not valid. E.g.: COM='COM21' or COM=21 ")
            print(e)
    try:
        baud=int(baud)
    except Exception as e: 
        print("error. baud can not be parsed ")
        print(e)  
        
    print('COM port: {:s}'.format(COM))
    print('baude rate: {:d}'.format(int(baud)))
    
    if list==True or file=='':
        list_files(com=COM,baud=baud)
    else:
        read_file(file,path=path,printcontent=False,com=COM,baud=baud)



def list_files(com=COMPORT,baud=BAUDRATE,timeout=5):
    """
    Print file list on ESP32 SD card.
    Parameters
    ----------
      
    com : stirng, optional
        COM port to use. The default is 'COM21'.
    baud : int, optional
        baud rate. The default is 115200.
    Returns
    -------
    None.
    
   
    """
    
    # setting up serial port
    ser = serial.Serial(com, baud, timeout=0) 
    
    
    try:
        print('getting file list')
        print('#------------------------')

        sweep=True
        STOP=False
        t=time.time()

        bitstring=b'LIST'


        ser.write(bitstring) 
        
        
        while sweep: 
            l=ser.readline()
            if l!=b'':
                print(l)
                t=time.time()
                if l[:6] == b'%% end':
                    STOP=True
                if STOP:  #Stop one line after #STOP is detected
                    sweep=False
                    
                
                    
            time.sleep(0.0005) 
            if (time.time()-t>timeout):
                print('timeout')
                break
    
        ser.close()
        print('done')
        
    except Exception as e: 
        print('error')
        print(e)
        try:
            ser.close()

        except Exception as e: 
            print('error')
            print(e)


def read_file(file,path='',printcontent=False,com=COMPORT,baud=BAUDRATE,timeout=5):
    """
    Parameters
    ----------
    file : string, file to read 
    path :string, optional
        Path for saving file. If not given use folder python is exacuted. The default is ''.
    printcontent : Bool, optional
        Print file content if true . The default is False.
    com : stirng, optional
        COM port to use. The default is 'COM21'.
    baud : int, optional
        baud rate. The default is 115200.
    timeout: int,optional
            Timeout after which the function interrupts if not receiving data
    Returns
    -------
    None.
    Example:  read_file('INS230712_2324.csv',path=r'C:/Users/Laktop/Desktop', printcontent=False,com='COM21',baud=115200)
    
    """
    
    # setting up serial port
    ser = serial.Serial(com, baud, timeout=0) 
    # ser.set_buffer_size(rx_size = 12800, tx_size = 12800)
    
    i_max=timeout/0.00001
    i_max2=timeout/0.0001
    try:
        print('Reading file: '+file)
        print('#------------------------')
        r=False
        sweep=True
        STOP=False
        t=time.time()
        t2=time.time()
        try:
            if path !='':
                if not os.path.isdir(path):
                    os.makedirs(path)
            
            f = open(path+r'/'+file,'wb')
        except Exception as e: 
            print('File can not be created. Choose an other path. ')
            print(e)
            ser.close()
            return
        
        bitstring=b'READTOT:/'+file.encode()+b':'
        # bitstring=r'READ:/'+file+':'
        # print(bitstring)
        # print('#\n#\n#\n#\n#\n')
        

        ser.write(bitstring) 
        
        bites_written=0
        printbites=5000
        i=0
        readend=False
        while sweep: 
            l=ser.readline()
            if l!=b'':
                if printcontent:
                    print(l)
                

                
                if l[:7] == b'&START&' :
                    r=True
                    # bites_written+= f.write(l)
                    print('start reading')
                    break
                
                
                    
            time.sleep(0.0001) 
            i+=1
            if (i>i_max2):
                print('timeout')
                break
        
        
        
        if r:
            print('Start logging file')
            i=0
            while sweep: 
                l=ser.readline()
                if l!=b'':
                    i=0
                    
                    
                    # t=time.time()
                    if printcontent:
                        print(l)
                    
    
                   
                    
                    if l[:6] == b'&STOP&':
                        # time.sleep(0.01) 
                        # l=ser.readline()
                        # bites_written+= f.write(l)
                        print('end of file')
                        sweep=False
                        readend=True
                        break

                    bites_written+= f.write(l)
                    
                    
                    if bites_written>printbites:
                        print('Bites written: {:d}'.format(bites_written)  ) 
                        printbites+=20000
                        

                i+=1
                if (i>i_max):
                    print('timeout')
                    sweep=False
                    break
                # if (time.time()-t>timeout):
                #     print('timeout')
                #     break
        
        f.close()
        
        
        totalsent=0
        sweep=True
        if readend:
            print('parsing ESP info')
            i=0
            while sweep: 
                l=ser.readline()
                if l!=b'':
                    i=0                 
                    # t=time.time()
                    if printcontent:
                        print(l)
                    
    
                   
                    if l[:13] == b'#%TotalBites:':
                        totalsent=int(l[13:])
                        
                    if l[:5] == b'&END&': 
                        sweep=False
                        break
                        

                i+=1
                if (i>i_max):
                    print('timeout')
                    sweep=False
                    break
        
        ser.close()
        print('Total data written: {:.3f} kB'.format(bites_written/1000)  )
        if totalsent>0: 
            print('Total data sent by ESP32: {:.3f} kB'.format(totalsent/1000)  )
        else:
            print('Total data sent by ESP32 is unknown')
        print('file saved: {:s}'.format(path+'\\'+file)  )
        print('Time: {:.1f} s'.format(time.time() -t2) )
    
    except Exception as e: 
        print('error')
        print(e)
        try:
            ser.close()
            f.close()
        except Exception as e: 
            print('error')
            print(e)

#%%
if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Print the arguments passed in from the command line
        # print("Arguments passed in from the command line:")
        kwargs = dict(arg.split('=') for arg in sys.argv[1:] if '=' in arg)
        # Print the keyword arguments
        if kwargs:
           
            # print("Keyword arguments passed in from the command line:")
            for key, value in kwargs.items():
                print(f"{key}: {value}")
            main(**kwargs)
        else:
            main()
    else:
        main()