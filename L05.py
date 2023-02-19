# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 14:04:14 2023

@author: Ignotus
"""

# ###############################################################
# 
# Chris Pierson #
# ECE 351 #
# Lab 5 #
# Due 2/21/23 #
# 
# ###############################################################

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

#################################################################
                        ##  PART 1 ##
#################################################################

R = 1000
L = 27E-3
C = 100E-9

steps = 1e-5
t = np.arange(0, 1.2e-3 + steps, steps) 

num = [(1/(R*C)),0]
den = [1, (1/(R*C)), (1/(L*C))]

tout, yout = sig.impulse((num,den), T = t)#impulse function of num & den 

# def h_1(t): # numbers plugged into sine method
#     y = (10358*np.exp(-5000*t)*np.sin(18584*t+105*np.pi/180))
#     return y
# h1 = h_1(t)

def p(R,L,C):
    return (-1/(R*C)+((1/(R*C))**2-4/(L*C))**.5)/2
#print(p(R,L,C).real)

def alpha(p):
    return p(R,L,C).real

def omega(p):
    return p(R,L,C).imag

#print("ap:", alpha(p)) 

def g(R,L,C):
    return 1/(R*C)*p(R, L, C)
#print(g(R,L,C))    

def mag_g(R,L,C):
    return (g(R,L,C).real**2+g(R,L,C).imag**2)**.5                
#print("test: ",(mag_g(R,L,C)/omega(p))*np.exp(alpha(p)*.0003))

def angle_g(R,L,C):  # should return proper angle dep. on quadrant
    if g(R,L,C).real > 0 and g(R,L,C).imag > 0:
        return np.arctan(g(R,L,C).imag/g(R,L,C).real)
    elif g(R,L,C).real < 0 and g(R,L,C).imag > 0:
        return np.arctan(g(R,L,C).imag/g(R,L,C).real)+np.pi
    elif g(R,L,C).real < 0 and g(R,L,C).imag < 0:
        return np.arctan(g(R,L,C).imag/g(R,L,C).real)+np.pi
    elif g(R,L,C).real > 0 and g(R,L,C).imag < 0:
        return np.arctan(g(R,L,C).imag/g(R,L,C).real)
#print(angle_g(R,L,C))

def sine_method(R,L,C,x): #sine method from class
    return mag_g(R,L,C)/omega(p)*np.exp(alpha(p)*t)*np.sin(omega(p)*x+angle_g(R,L,C))

#print(sine_method(R,L,C,.0003))

#Code for plots
plot1 = plt.figure(figsize =(12 ,8)) # start a new figure

plt.subplot(2 ,1 ,1) # subplot 1: subplot format (row , column , number )
plt.plot(t, sine_method(R,L,C,t)) # choosing plot variables for x and y axes
plt.ylabel('Hand Calculation') # label for subplot 1
plt.grid(True) # show grid on plot

plt.title('Lab 5 Plots') # title for entire figure 

plt.subplot(2, 1, 2) # subplot 2
plt.plot(t, yout)
plt.ylabel('Scipy.Signal Output') # label for subplot 2
plt.grid(True) # show grid on plot

plt.show()

#################################################################
                        ##  PART 2 ##
#################################################################

tout, yout2 = sig.step((num,den), T = t)#step response of H(s) w/ sig.step

plot2 = plt.figure(figsize =(12 ,8)) # start a new figure 

plt.subplot(2 ,1 ,1) # subplot 1: subplot format (row , column , number )
plt.plot(t,yout2) # choosing plot variables for x and y axes
plt.ylabel('Step Response of H(s)') # label for subplot 1
plt.grid(True) # show grid on plot

plt.title('Step Response of H(s)') # title for entire figure
plt.show()





