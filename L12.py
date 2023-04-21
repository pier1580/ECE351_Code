# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 17:28:13 2023

@author: Ignotus
"""

# ###############################################################
# 
# Chris Pierson #
# ECE 351 #
# Lab 12 #
# Due 4/25/23 #
# 
# ###############################################################

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as sig
# import control as con 
import scipy.fftpack as fftp

#################################################################
                        ##  PART 1 ##
#################################################################

# the other packages you import will go here
import pandas as pd

# load input signal
df = pd.read_csv('NoisySignal.csv')

t = df['0'].values
sensor_sig = df['1'].values

plt.figure(figsize = (10 , 7))
plt.plot(t, sensor_sig)
plt.grid()
plt.title(' Noisy Input Signal ')
plt.xlabel(' Time [s] ')
plt.ylabel(' Amplitude [V] ')
plt.show()

# your code starts here , good luck
fs = 1/1e-6
def fft2(x):  # improved fast fourier transform
    N = len(x)
    X_fft = fftp.fft(x)
    X_fft_shifted = fftp.fftshift(X_fft)
    freq = np.arange(-N/2, N/2)*fs/N
    X_mag = np.abs(X_fft_shifted)/N
    X_phi = np.angle(X_fft_shifted)
    NX = len(X_mag) # gets array length
    for i in range(0,NX):
        if(np.abs(X_mag[i]) < 1e-10): # sets corresponding array element in 
            X_phi[i] = 0              # angle array to 0 if magnitude<1e-10  
    return freq, X_mag, X_phi

def make_stem(ax, x, y, color = 'k', style = 'solid', label = ' ', linewidths = 2.5, ** kwargs):
    ax.axhline(x[0], x[-1], 0, color = 'r')
    ax.vlines(x, 0, y, color = color, linestyles = style, label = label, linewidths = linewidths)
    ax.set_ylim([1.05 * y.min(), 1.05 * y.max()])
    
freq, X_mag, X_phi = fft2(sensor_sig)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize = (10 , 7))

plt.subplot(2 ,1 ,1) # subplot 1: subplot format (row , column , number )
make_stem(ax1, freq, X_mag)
plt.xlim(0,450e3)
plt.ylabel('Magnitude (V)') # label for subplot 1
plt.grid(True) # show grid on plot

plt.title('Full Range') # title for entire figure 

plt.subplot(2, 1, 2) # subplot 2
make_stem(ax2, freq, X_phi)
plt.xlim(0,450e3)
plt.ylabel('Angle (radians)') # label for subplot 2
plt.grid(True) # show grid on plot
plt.xlabel('Frequency (Hz)') # label for subplot 2

plt.show()

fig, (ax3, ax4) = plt.subplots(2, 1, figsize = (10 , 7))

plt.subplot(2 ,1 ,1) # subplot 1: subplot format (row , column , number )
make_stem(ax3, freq, X_mag)
plt.xlim(0,1.79e3)
plt.ylim(0,0.8)
plt.ylabel('Magnitude (V)') # label for subplot 1
plt.grid(True) # show grid on plot

plt.title('Low Frequencies') # title for entire figure 

plt.subplot(2, 1, 2) # subplot 2
make_stem(ax4, freq, X_phi)
plt.xlim(0,1.79e3)
plt.ylabel('Angle (radians)') # label for subplot 2
plt.grid(True) # show grid on plot
plt.xlabel('Frequency (Hz)') # label for subplot 2

plt.show()

fig, (ax5, ax6) = plt.subplots(2, 1, figsize = (10 , 7))

plt.subplot(2 ,1 ,1) # subplot 1: subplot format (row , column , number )
make_stem(ax5, freq, X_mag)
plt.xlim(1.79e3, 2.01e3)
plt.ylabel('Magnitude (V)') # label for subplot 1
plt.grid(True) # show grid on plot

plt.title('Position Sensor Frequencies') # title for entire figure 

plt.subplot(2, 1, 2) # subplot 2
make_stem(ax6, freq, X_phi)
plt.xlim(1.79e3, 2.01e3)
plt.ylabel('Angle (radians)') # label for subplot 2
plt.grid(True) # show grid on plot
plt.xlabel('Frequency (Hz)') # label for subplot 2

plt.show()

fig, (ax7, ax8) = plt.subplots(2, 1, figsize = (10 , 7))

plt.subplot(2 ,1 ,1) # subplot 1: subplot format (row , column , number )
make_stem(ax7, freq, X_mag)
plt.xlim(2.01e3, 450e3)
plt.ylim(0,0.8)
plt.ylabel('Magnitude (V)') # label for subplot 1
plt.grid(True) # show grid on plot

plt.title('High Frequencies') # title for entire figure 

plt.subplot(2, 1, 2) # subplot 2
make_stem(ax8, freq, X_phi)
plt.xlim(2.01e3, 450e3)
plt.ylabel('Angle (radians)') # label for subplot 2
plt.grid(True) # show grid on plot
plt.xlabel('Frequency (Hz)') # label for subplot 2

plt.show()

#################################################################
                        ##  PART 2 ##
#################################################################

R = 39           # Resistor Value (Ohms) started with 159, needed a wider bandwidth
L = 1.4e-3       # Inductor Value (Henrys) 
C = 5e-6         # Capacitor Value (Farads)
beta = 1/(R*C)
w0 = (1/(L*C))**.5

num = [beta,0]             
den = [1, beta, w0**2]

w = np.linspace(-100, 450e4, 10**5)  # start, stop, # of points in freq range
system = num, den
w, Bode_mag, Bode_phase = sig.bode(system, w)

plot0 = plt.figure(figsize =(12 ,8)) # start a new figure        

plt.subplot(211)
plt.semilogx(w/(2*np.pi), Bode_mag) 
plt.ylabel('Magnitude (dB)')
plt.title('All Frequency Bode')
plt.grid(which='both')
plt.xlim(1, 450e3)

plt.subplot(212)
plt.semilogx(w/(2*np.pi), Bode_phase)   
plt.xlabel('Hz') 
plt.ylabel('Phase (degrees)')
plt.grid(which='both')
plt.xlim(1, 450e3)

plt.show()

plot1 = plt.figure(figsize =(12 ,8)) # start a new figure  

plt.subplot(211)
plt.semilogx(w/(2*np.pi), Bode_mag)    
plt.ylabel('Magnitude (dB)')
plt.title('Low Frequency Bode')
plt.grid(which='both')
plt.xlim(1, 1.8e3)

plt.subplot(212)
plt.semilogx(w/(2*np.pi), Bode_phase)   
plt.xlabel('Hz') 
plt.ylabel('Phase (degrees)')
plt.grid(which='both')
plt.xlim(1, 1.8e3)

plt.show()

plot2 = plt.figure(figsize =(12 ,8)) # start a new figure  

plt.subplot(211)
plt.semilogx(w/(2*np.pi), Bode_mag)    
plt.ylabel('Magnitude (dB)')
plt.title('Position Measurement Bode')
plt.grid(which='both')
plt.xlim(1.8e3, 2e3)
plt.ylim(-.3, 0)

plt.subplot(212)
plt.semilogx(w/(2*np.pi), Bode_phase)   
plt.xlabel('Hz') 
plt.ylabel('Phase (degrees)')
plt.grid(which='both')
plt.xlim(1.8e3, 2e3)

plt.show()

plot3 = plt.figure(figsize =(12 ,8)) # start a new figure  

plt.subplot(211)
plt.semilogx(w/(2*np.pi), Bode_mag)    
plt.ylabel('Magnitude (dB)')
plt.title('High Frequency Bode')
plt.grid(which='both')
plt.xlim(2e3, 450e3)

plt.subplot(212)
plt.semilogx(w/(2*np.pi), Bode_phase)   
plt.xlabel('Hz') 
plt.ylabel('Phase (degrees)')
plt.grid(which='both')
plt.xlim(2e3, 450e3)

plt.show()

#################################################################
                        ##  PART 3 ##
#################################################################

z1,z2 = sig.bilinear(num,den,fs)
last1 = sig.lfilter(z1,z2,sensor_sig)

plot4 = plt.figure(figsize =(12, 8)) # start a new figure
plt.plot(t, last1)
plt.grid(True) # show grid on plot
plt.title(' Filtered Input Signal ')
plt.xlabel(' Time [s] ')
plt.ylabel(' Amplitude [V] ')

plt.show()

freq, X_mag, X_phi = fft2(last1)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize = (10 , 7))

plt.subplot(2 ,1 ,1) # subplot 1: subplot format (row , column , number )
make_stem(ax1, freq, X_mag)
plt.xlim(0,450e3)
plt.ylabel('Magnitude (V)') # label for subplot 1
plt.grid(True) # show grid on plot

plt.title('Filtered Full Range') # title for entire figure 

plt.subplot(2, 1, 2) # subplot 2
make_stem(ax2, freq, X_phi)
plt.xlim(0,450e3)
plt.ylabel('Angle (radians)') # label for subplot 2
plt.grid(True) # show grid on plot
plt.xlabel('Frequency (Hz)') # label for subplot 2

plt.show()

fig, (ax3, ax4) = plt.subplots(2, 1, figsize = (10 , 7))

plt.subplot(2 ,1 ,1) # subplot 1: subplot format (row , column , number )
make_stem(ax3, freq, X_mag)
plt.xlim(0,1.79e3)
plt.ylim(0,0.2)
plt.ylabel('Magnitude (V)') # label for subplot 1
plt.grid(True) # show grid on plot

plt.title('Filtered Low Frequencies') # title for entire figure 

plt.subplot(2, 1, 2) # subplot 2
make_stem(ax4, freq, X_phi)
plt.xlim(0,1.79e3)
plt.ylabel('Angle (radians)') # label for subplot 2
plt.grid(True) # show grid on plot
plt.xlabel('Frequency (Hz)') # label for subplot 2

plt.show()

fig, (ax5, ax6) = plt.subplots(2, 1, figsize = (10 , 7))

plt.subplot(2 ,1 ,1) # subplot 1: subplot format (row , column , number )
make_stem(ax5, freq, X_mag)
plt.xlim(1.79e3, 2.01e3)
plt.ylabel('Magnitude (V)') # label for subplot 1
plt.grid(True) # show grid on plot

plt.title('Filtered Position Sensor Frequencies') # title for entire figure 

plt.subplot(2, 1, 2) # subplot 2
make_stem(ax6, freq, X_phi)
plt.xlim(1.79e3, 2.01e3)
plt.ylabel('Angle (radians)') # label for subplot 2
plt.grid(True) # show grid on plot
plt.xlabel('Frequency (Hz)') # label for subplot 2

plt.show()

fig, (ax7, ax8) = plt.subplots(2, 1, figsize = (10 , 7))

plt.subplot(2 ,1 ,1) # subplot 1: subplot format (row , column , number )
make_stem(ax7, freq, X_mag)
plt.xlim(2.01e3, 450e3)
plt.ylim(0,0.1)
plt.ylabel('Magnitude (dB)') # label for subplot 1
plt.grid(True) # show grid on plot

plt.title('Filtered High Frequencies') # title for entire figure 

plt.subplot(2, 1, 2) # subplot 2
make_stem(ax8, freq, X_phi)
plt.xlim(2.01e3, 450e3)
plt.ylabel('Angle (radians)') # label for subplot 2
plt.grid(True) # show grid on plot
plt.xlabel('Frequency (Hz)') # label for subplot 2

plt.show()

plot10 = plt.figure(figsize =(12, 8)) # start a new figure
plt.plot(t, sensor_sig, 'g', label='raw signal')
plt.plot(t, last1, 'b', label='filtered signal')
plt.legend()
plt.grid(True) # show grid on plot
plt.title(' Overlaid Signal ')
plt.xlabel(' Time [s] ')
plt.ylabel(' Amplitude [V] ')

plt.show()

###############################################################################
freq0, X_mag0, X_phi0 = fft2(sensor_sig)

fig, (ax9, ax10) = plt.subplots(2, 1, figsize = (10 , 7))

plt.subplot(2 ,1 ,1).set_title('Unfiltered') # subplot 1: subplot format (row , column , number )
make_stem(ax9, freq0, X_mag0)
plt.xlim(0,450e3)
plt.ylabel('Magnitude (V)') # label for subplot 1
plt.grid(True) # show grid on plot

plt.subplot(2 ,1 ,2).set_title('Filtered') # subplot 1: subplot format (row , column , number )
make_stem(ax10, freq, X_mag)
plt.xlim(0,450e3)
plt.ylabel('Magnitude (V)') # label for subplot 1
plt.grid(True) # show grid on plot

plt.show()

fig, (ax9, ax10) = plt.subplots(2, 1, figsize = (10 , 7))

plt.subplot(2 ,1 ,1).set_title('Unfiltered') # subplot 1: subplot format (row , column , number )
make_stem(ax9, freq0, X_mag0)
plt.xlim(0,1.79e3)
plt.ylabel('Magnitude (V)') # label for subplot 1
plt.grid(True) # show grid on plot

#plt.title('Low Range') # title for entire figure 

plt.subplot(2 ,1 ,2).set_title('Filtered') # subplot 1: subplot format (row , column , number )
make_stem(ax10, freq, X_mag)
plt.xlim(0,1.79e3)
plt.ylabel('Magnitude (V)') # label for subplot 1
plt.grid(True) # show grid on plot


plt.show()

fig, (ax11, ax12) = plt.subplots(2, 1, figsize = (10 , 7))

plt.subplot(2 ,1 ,1).set_title('Unfiltered') # subplot 1: subplot format (row , column , number )
make_stem(ax11, freq0, X_mag0)
plt.xlim(1.79e3, 2.01e3)
plt.ylabel('Magnitude (V)') # label for subplot 1
plt.grid(True) # show grid on plot

#plt.title('Low Range') # title for entire figure 

plt.subplot(2 ,1 ,2).set_title('Filtered') # subplot 1: subplot format (row , column , number )
make_stem(ax12, freq, X_mag)
plt.xlim(1.79e3, 2.01e3)
plt.ylabel('Magnitude (V)') # label for subplot 1
plt.grid(True) # show grid on plot

plt.show()

fig, (ax13, ax14) = plt.subplots(2, 1, figsize = (10 , 7))

plt.subplot(2 ,1 ,1).set_title('Unfiltered') # subplot 1: subplot format (row , column , number )
make_stem(ax13, freq0, X_mag0)
plt.xlim(2.01e3, 450e3)
plt.ylabel('Magnitude (V)') # label for subplot 1
plt.grid(True) # show grid on plot

#plt.title('Low Range') # title for entire figure 

plt.subplot(2 ,1 ,2).set_title('Filtered') # subplot 1: subplot format (row , column , number )
make_stem(ax14, freq, X_mag)
plt.xlim(2.01e3, 450e3)
plt.ylabel('Magnitude (V)') # label for subplot 1
plt.grid(True) # show grid on plot

plt.show()




