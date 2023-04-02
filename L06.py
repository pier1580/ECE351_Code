# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 17:28:13 2023

@author: Ignotus
"""

# ###############################################################
# 
# Chris Pierson #
# ECE 351 #
# Lab 6 #
# Due 2/28/23 #
# 
# ###############################################################

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

#################################################################
                        ##  PART 1 ##
#################################################################
steps = 1e-5
t = np.arange(0, 2 + steps, steps) 

def prelab(t):
    return 1/2+np.exp(-6*t)-.5*np.exp(-4*t)

num0 = [1, 6, 12]
den0 = [1,10, 24]
den1 = [1,10,24,0]

tout, yout = sig.step((num0,den0), T = t)#impulse function of num & den 

#Code for plots
plot1 = plt.figure(figsize =(12 ,8)) # start a new figure

plt.subplot(2 ,1 ,1) # subplot 1: subplot format (row , column , number )
plt.plot(t, prelab(t)) # choosing plot variables for x and y axes
plt.ylabel('Hand Calculation') # label for subplot 1
plt.grid(True) # show grid on plot

plt.title('Lab 5 Part 1 Plots') # title for entire figure 

plt.subplot(2, 1, 2) # subplot 2
plt.plot(t, yout)
plt.ylabel('sig.step Output') # label for subplot 2
plt.grid(True) # show grid on plot

plt.show()

r,p,k = sig.residue(num0, den1)

print('\nSolution to prelab H(s):\nr=',r)
print('p=',p)
print('k=',k)

#################################################################
                        ##  PART 2 ##
#################################################################

num2 = [25250]
den2 = [1,18,218,2036,9085,25250,0]
den3 = [1,18,218,2036,9085,25250]

r,p,k = sig.residue(num2, den2)

print('\nSolution to lab H(s):\nr=',np.round(r,3))
print('p=',p)
print('k=',k)

def alpha(p):
    return p.real

def omega(p):
    return p.imag

def k_mag(r):
    return np.abs(r)

def k_ang(r):
    return np.angle(r)

def cosine_method(r,p,t): #cosine method from class
    return k_mag(r)*np.exp(alpha(p)*t)*np.cos(omega(p)*t+k_ang(r))

t=np.arange(0,4.5+steps,steps)

yout2=(cosine_method(r[0],p[0],t)+cosine_method(r[1],p[1],t)
       +cosine_method(r[2],p[2],t)+cosine_method(r[3],p[3],t)
        +cosine_method(r[4],p[4],t)+cosine_method(r[5],p[5],t))

tout, yout3 = sig.step((num2,den3), T = t)

#Code for plots
plot2 = plt.figure(figsize =(12 ,8)) # start a new figure

plt.subplot(2 ,1 ,1) # subplot 1: subplot format (row , column , number )
plt.plot(t, yout2) # choosing plot variables for x and y axes
plt.ylabel('Cosine Method') # label for subplot 1
plt.grid(True) # show grid on plot

plt.title('Lab 5 Part 2 Plots') # title for entire figure 

plt.subplot(2, 1, 2) # subplot 2
plt.plot(t, yout3)
plt.ylabel('sig.step Output') # label for subplot 2
plt.grid(True) # show grid on plot

plt.show()