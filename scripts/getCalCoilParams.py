#%% import module pyLEM
#if you use a direct path

scriptspath=r'C:\Users\AcCap\LEM_code'
sys.path.append(scriptspath)

from pyLEM.dataADC import *
from pyLEM.dataADC import refCalibration
#%% get reference values for calibration (CalCoil)
CalParams = {'Rs': np.array([81.54,147.3,403.8, 622.0,279.3,127.5,]),
             'Ls': np.ones_like(6) * 0.0117 ,
             'Ac': 0.00502,
             'Nc': 320,
             'dR': 2.027,
             'dB': 0.821,
             'dC': 1.695}

f=4000  # frequency in Hz


Z,ZI,ZQ,magZ=refCalibration(f,**CalParams)


'''
Results:
--------
z=np.array([0.05019815+0.01391979j, 0.04321427+0.02164732j,
       0.01873284+0.02572434j, 0.0098748 +0.02088782j,
       0.02841909+0.02699326j, 0.04550323+0.01972998j])
       
I=np.array([0.05019815, 0.04321427, 0.01873284, 0.0098748 , 0.02841909, 0.04550323])
Q=np.array([0.01391979, 0.02164732, 0.02572434, 0.02088782, 0.02699326, 0.01972998])
magZ=np.array([0.05209236, 0.04833301, 0.03182234, 0.02310439, 0.03919542, 0.04959654])
'''


#%% Get calibration parameters gain g and phase phi

calQ=np.array([])
calI=np.array([])

g,phi,[res,params2,res3]=fitCalibrationParams(calQ,calI,f,plot=False,CalParams={})

