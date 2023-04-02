# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 17:28:13 2023

@author: Ignotus
"""

# ###############################################################
# 
# Chris Pierson #
# ECE 351 #
# Lab 10 #
# Due 4/4/23 #
# 
# ###############################################################

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as sig
import control as con 

#################################################################
                        ##  PART 1 ##
#################################################################

R = 1000            # Resistor Value (Ohms)
L = 27e-3           # Inductor Value (Henrys)
C = 100e-9          # Capacitor Value (Farads)
alpha = 1/(R*C) 
beta = 1/(L*C)

num = [alpha,0]             
den = [1, alpha, beta]

w = np.linspace(10**3,10**6,10**3)  # start, stop, # of points in freq range
system = num, den

hand_mag = (w*alpha/(w**2*alpha**2+beta**2-2*w**2*beta+w**4)**.5)

hand_phase = 90 - np.arctan(w*alpha/(beta-w**2))*(180/np.pi) 
for i in range(len(hand_phase)):    # fix phase change at 90 degrees
    if  (hand_phase[i] > 90):
        hand_phase[i] = hand_phase[i] - 180

w, Bode_mag, Bode_phase = sig.bode(system, w)

def dB2Mag(x):              # dB to magnitude
    return 10**(x/20)

def Mag2dB(x):
    return 20 * np.log10(x)

def Rad2Deg(x):
    return 180*x/np.pi

#################### Magnitude Plots #############################
plot0 = plt.figure(figsize =(12 ,8)) # start a new figure        

plt.subplot(211)
plt.semilogx(w, Mag2dB(hand_mag))
plt.ylabel('Magnitude (dB)')
plt.title('Hand-Derived')
plt.grid(which='both')
# plt.axis([1e3, 1e6, -40, 5])

plt.subplot(212)
plt.semilogx(w, Bode_mag)    
plt.ylabel('Magnitude (dB)')
plt.title('scipy.signal.bode')
plt.grid(which='both')
plt.axis([1e3, 1e6, -40, 5])

plt.show()

#################### Phase Plots #################################
plot1 = plt.figure(figsize =(12 ,8)) # start a new figure

plt.subplot(211)
plt.semilogx(w, hand_phase)
plt.ylabel('Phase (degrees)')
plt.title('Hand-Derived')
plt.grid(which='both')
plt.axis([1e3, 1e6, -90, 90])
plt.yticks([90,0,-20, -40, -60,-80,-90])

plt.subplot(212)
plt.semilogx(w, Bode_phase)   
plt.xlabel('$\omega$ (rad/s)') 
plt.ylabel('Phase (degrees)')
plt.title('scipy.signal.bode')
plt.grid(which='both')
plt.axis([1e3, 1e6, -90, 90])

plt.show()

#################### Task 3 Plots #################################
sys = con.TransferFunction(num, den)

mag1, phase1, omega1 = con.bode(sys, w, dB = True, Hz = True, deg = True, plot = False) 

# Modified auto-plot from handout
plot2 = plt.figure(figsize =(12 ,8)) # start a new figure

plt.subplot(211)
plt.semilogx(omega1, Mag2dB(mag1)) # mag1 was not in dB
plt.ylabel('Magnitude (dB)')
plt.title('control.bode output')
plt.grid(which='both')
plt.axis([1e3, 1e6, -40, 5])

plt.subplot(212)
plt.semilogx(omega1, 360 + Rad2Deg(phase1)) # Added 360 to match previous plots
plt.xlabel('Hz)') 
plt.ylabel('Phase (degrees)')
# plt.title('scipy.signal.bode')
plt.grid(which='both')
plt.axis([1e3, 1e6, -90, 90])

plt.show()

#################################################################
                        ##  PART 2 ##
#################################################################

freq1 = 3*50000 # largest freq in function * 3
steps = 1/freq1 
t = np.arange(0, .01, steps) 

def x(t): 
    return np.cos(2*np.pi*100*t)+np.cos(2*np.pi*3024*t)+np.sin(2*np.pi*50000*t)
    
plot1 = plt.figure(figsize =(12 ,8)) # start a new figure
plt.plot(t, x(t))
plt.grid(True) # show grid on plot

plt.show()

z1,z2 = sig.bilinear(num,den,freq1)
last1 = sig.lfilter(z1,z2,x(t))

plot1 = plt.figure(figsize =(12, 8)) # start a new figure
plt.plot(t, last1)
plt.grid(True) # show grid on plot

plt.show()


