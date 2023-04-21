# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 17:28:13 2023

@author: Ignotus
"""

# ###############################################################
# 
# Chris Pierson #
# ECE 351 #
# Lab 11 #
# Due 4/11/23 #
# 
# ###############################################################

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as sig
# import control as con 

from matplotlib import patches

#################################################################
                        ##  PART 1 ##
#################################################################

num = [2,-40]
den = [1, -10, 16]


r, p, k = sig.residuez(num, den)
print('\nTask 3 Output:\n')
# print(r, p, k)
print('Poles: ',p)
print('Residues: ',r,'\n\n')

def zplane (b, a, filename = None):

    ax = plt.subplot (1, 1, 1)
    
    # create the unit circle
    uc = patches.Circle ((0,0), radius=1, fill=False, color = 'black', ls='dashed') 
    ax.add_patch(uc)
    # the coefficients are less than 1, normalize the coefficients
    if np. max (b) > 1:
        kn = np. max (b)
        b = np.array (b)/float (kn)
    else:
        kn = 1

    if np.max (a) > 1:
        kd = np.max(a)
        a = np.array(a)/float (kd)
    else:
        kd = 1
    
    # get the poles and zeros = np.roots (a)
    p = np.roots(a)
    z = np.roots(b)
    k = kn/float (kd)
    
    # plot the zeros and set marker properties
    t1 = plt.plot(z.real, z.imag, 'o', ms=10, label='Zeros') 
    plt.setp(t1, markersize=10.0, markeredgewidth=1.0)
    
    # plot the poles and set marker properties
    t2 = plt.plot(p.real, p.imag, 'x', ms=10, label='Poles') 
    plt.setp( t2, markersize=12.0, markeredgewidth=3.0)
    ax.spines ['left'].set_position('center') 
    ax.spines  ['bottom'].set_position('center')
    ax. spines ['right'].set_visible (False) 
    ax. spines ['top'].set_visible (False)
    plt.legend()
    
    # set the ticks
    
    # r = 1.5; plt.axis ('scaled'); plt.axis ([-r, r,-r, r])
    # ticks = [-1, -.5, .5, 1]; plt.xticks (ticks); plt.yticks(ticks)
    
    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)
    return z, p, k
    
z, p, k = zplane(num, den)
# print(z, p, k)

w, h = sig.freqz(num, den, whole= True)

w = w/np.pi  # Changes frequency range to 0 to 2pi radians
angles = np.unwrap(np.angle(h)) # code taken from python freqz() documentation

for i in range(len(angles)): # Changes output angle to degrees and centers at 0
    angles[i] = angles[i] * 180/np.pi-360

plot1 = plt.figure(figsize =(12 ,8)) # start a new figure

plt.subplot(2 ,1 ,1) # subplot 1: subplot format (row , column , number )
plt.plot(w, 20 * np.log10(abs(h)),'b') # choosing plot variables for x and y axes
plt.ylabel('H(z) Magnitude (dB)') # label for subplot 1
plt.grid(True) # show grid on plot

plt.title('Lab 11 Plots') # title for entire figure 

plt.subplot(2, 1, 2) # subplot 2
plt.plot(w, angles, 'g')
plt.ylabel('Angle (degrees)') # label for subplot 2
plt.grid(True) # show grid on plot
plt.xlabel('Frequency (radians/pi)') # label for subplot 2

plt.show()


        
        