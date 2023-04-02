# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 17:28:13 2023

@author: Ignotus
"""

# ###############################################################
# 
# Chris Pierson #
# ECE 351 #
# Lab 7 #
# Due 3/7/23 #
# 
# ###############################################################

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

#################################################################
                        ##  PART 1 ##
#################################################################
steps = 1e-5
t = np.arange(0, 5 + steps, steps) 

# def prelab(t):
#     return 1/2+np.exp(-6*t)-.5*np.exp(-4*t)

Gnum0 = [1, 9]
Gden0 = [1, -2, -40, -64]

Anum0 = [1, 4]
Aden0 = [1, 4, 3]

Bnum = [1, 26, 168]


print('\nTask 2 Outputs: ')
z,p,k = sig.tf2zpk(Gnum0,Gden0)
print('\nG(s):\nz=',z)
print('p=',p)
print('k=',k)

z,p,k = sig.tf2zpk(Anum0,Aden0)
print('\nA(s):\nz=',z)
print('p=',p)
print('k=',k)

print('\nB: ',np.roots(Bnum))

OpenLnum = sig.convolve(Anum0, Gnum0)
print ('\nOpen Loop Numerator = ' , OpenLnum)

OpenLden = sig.convolve(Aden0, Gden0)
print ('Open Loop Denominator = ', OpenLden)

OpenLStepT, OpenLStepY = sig.step((OpenLnum,OpenLden), T = t)

#Code for plots
plot1 = plt.figure(figsize =(12 ,8)) # start a new figure

plt.plot(OpenLStepT, OpenLStepY) # choosing plot variables for x and y axes
plt.ylabel('H') # label for subplot 1
plt.grid(True) # show grid on plot

plt.title('Open Loop') # title for entire figure 

plt.show()

#################################################################
                        ##  PART 2 ##
#################################################################

ClosedLnum = sig.convolve(Anum0, Gnum0)
print ('\nClosed Loop Numerator = ' , ClosedLnum)

ClosedLden = sig.convolve(Aden0, Gden0) + sig.convolve(Bnum, sig.convolve(Aden0,Gnum0))
print ('Closed Loop Denominator = ', ClosedLden)
print ('Closed Loop Values: ')
z,p,k = sig.tf2zpk(ClosedLnum,ClosedLden)
print('z=',z)
print('p=',np.round(p,2))
print('k=',k)

ClosedLStepT, ClosedLStepY = sig.step((ClosedLnum, ClosedLden), T = t)

#Code for plots
plot1 = plt.figure(figsize =(12 ,8)) # start a new figure

plt.plot(ClosedLStepT, ClosedLStepY) # choosing plot variables for x and y axes
plt.ylabel('H') # label for subplot 1
plt.grid(True) # show grid on plot

plt.title('Closed Loop') # title for entire figure 

plt.show()
