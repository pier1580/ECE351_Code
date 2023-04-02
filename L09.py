# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 17:28:13 2023

@author: Ignotus
"""

# ###############################################################
# 
# Chris Pierson #
# ECE 351 #
# Lab 9 #
# Due 3/28/23 #
# 
# ###############################################################

import numpy as np
import matplotlib.pyplot as plt
#import scipy.signal as sig
import scipy.fftpack as fftp

#################################################################
                        ##  PART 1 ##
#################################################################

fs = 100        # sampling frequency
steps = 1/fs    # period
t = np.arange(0,2,steps)
t2 = np.arange(0,16,steps) # for square wave plot

def fft(x):  # fast fourier transform
    N = len(x)
    X_fft = fftp.fft(x)
    X_fft_shifted = fftp.fftshift(X_fft)
    freq = np.arange(-N/2, N/2)*fs/N 
    X_mag = np.abs(X_fft_shifted)/N
    X_phi = np.angle(X_fft_shifted)
    return freq, X_mag, X_phi

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

T = 8
def square_wave(k,t):
    y=0
    for i in range(1,k+1):
        y += (2/(i*np.pi)*(1-np.cos(i*np.pi)))*np.sin(i*((2*np.pi)/T)*t)
    return y

plot1 = np.cos(2*np.pi*t)
plot2 = 5*np.sin(2*np.pi*t)
plot3 = 2*np.cos((2*np.pi*2*t)-2)+(np.sin((2*np.pi*6*t)+3))**2
plot4 = square_wave(15,t2)
plots = plot1, plot2, plot3, plot4

for k in range(2):
    x1 = 2  # zoomed in range  
    for i in range(len(plots)-1+k): #loops through plots 1-4 2x before plot5
        if k == 1 and i == 3: 
            x1=16  # changes range for square wave
            t = np.arange(0,16,steps)
        if k == 0:
            freq, X_mag, X_phi = fft(plots[i]) 
        if k == 1:
            freq, X_mag, X_phi = fft2(plots[i])
            
        plot = plt.figure(figsize =(12 ,8)) # start a new figure
    
        plt.subplot(3 ,1 ,1) # subplot 1: subplot format (row , column , number )
        plt.plot(t, plots[i]) # choosing plot variables for x and y axes
        plt.xlim(0,x1)
        plt.ylabel('x(t)') # label for subplot 1
        plt.grid(True) # show grid on plot
        
        plt.title('Lab 9') # title for entire figure 
        
        plt.subplot(3, 2, 3) # subplot 2
        plt.stem(freq, X_mag)
        plt.xlim(-20,20)
        plt.ylabel('mag') # label for subplot 2
        plt.grid(True) # show grid on plot
        
        plt.subplot(3, 2, 4) # subplot 3
        plt.stem(freq, X_mag)
        plt.xlim(-3,3)
        #plt.ylabel('mag') # label for subplot 3
        plt.grid(True) # show grid on plot
        
        plt.subplot(3, 2, 5) # subplot 4
        plt.stem(freq, X_phi)
        plt.xlim(-20,20)
        plt.ylabel('angle') # label for subplot 4
        plt.grid(True) # show grid on plot
        
        plt.subplot(3, 2, 6) # subplot 5
        plt.stem(freq, X_phi)
        plt.xlim(-3,3)
        #plt.ylabel('angle') # label for subplot 5
        plt.grid(True) # show grid on plot

