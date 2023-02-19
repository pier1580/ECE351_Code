# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 14:04:14 2023

@author: Ignotus
"""

# ###############################################################
# #
# Chris Pierson #
# ECE 351 #
# Lab 4 #
# Due 2/14/23 #
# #
# #
# ###############################################################

import numpy as np
import matplotlib.pyplot as plt
#import scipy.signal as sig

#################################################################
##  PART 1 ##

steps = 1e-2
#t = np.arange(0,20 + steps, steps)
t = np.arange(-10,10 + steps, steps)
f0 = 0.25
w0 = f0 * 2 * np.pi

def u(t):
    y = np.zeros(t.shape)
    for i in range(len(t)):
        if t[i] < 0:
            y[i] = 0
        else:
            y[i] = 1
    return y
uu = u(t)

def r(t):
    y = np.zeros(t.shape)   
    for i in range(len(t)):
        if t[i] < 0:
            y[i] = 0
        else:
            y[i] = t[i]
    return y

def h_1(t):
    y = np.exp(-t) * u(t)
    return y
h1 = h_1(t)

def h_2(t):
    y = u(t-2)-u(t-6)
    return y
h2 = h_2(t)

def h_3(t):
    y = np.cos(w0*t)*u(t)
    return y
h3 = h_3(t)

# Code for plots
plot1 = plt.figure(figsize =(12 ,8)) # start a new figure , with
# a custom figure size
plt.subplot(3 ,1 ,1) # subplot 1: subplot format (row , column , number )
plt.plot(t, h1) # choosing plot variables for x and y axes
plt.title('Lab 4 Function Plots') # title for entire figure

# (all three subplots )
plt.ylabel('h1(t)') # label for subplot 1
plt.grid(True) # show grid on plot

plt.subplot(3, 1, 2) # subplot 2
plt.plot(t, h2)

plt.ylabel('h2(t)') # label for subplot 2
plt.grid(which ='both') # use major and minor grids


plt.subplot(3, 1, 3) # subplot 3
plt.plot(t, h3)
plt.ylabel('h3(t)') # label for subplot 2
plt.grid(which ='both') # use major and minor grids

plt.show()

#################################################################
##  PART 2 ##

def my_conv(func_1, func_2):
    Nf1 = len(func_1)  #length of input function 1
    Nf2 = len(func_2)  #length of input function 2
    f1Extended = np.append(func_1, np.zeros((1, Nf2-1))) #makes functions same
    f2Extended = np.append(func_2, np.zeros((1, Nf1-1))) #length by adding 0s   
        
    result = np.zeros(f1Extended.shape) #output array

    for i in range(Nf2 + Nf1 - 2):  #loop through all values of both functions
        result[i] = 0
        for j in range(Nf1):  #loop through values of first function
            if(i - j + 1) > 0:
                try:  #attempts to multiply functions and set result to output
                    result[i] += f1Extended[j] * f2Extended[i - j + 1]
                except: #prints array location if there is an exception
                    print(i,j)                    
    return result    
print('length=',2*t[len(t)-1])

t_conv = np.arange(-20,2*t[len(t)-1]+steps,steps)

h1_u = my_conv(h1,uu)
h2_u = my_conv(h2,uu)
h3_u = my_conv(h3,uu)

# Code for plots
plot2 = plt.figure(figsize =(12 ,8)) # start a new figure , with
# a custom figure size
plt.subplot(3 ,1 ,1) # subplot 1: subplot format (row , column , number )
plt.plot(t_conv, h1_u) # choosing plot variables for x and y axes
plt.title('Lab 4 Convolution Plots') # title for entire figure

# (all three subplots )
plt.ylabel('h1(t)*u(t)') # label for subplot 1
plt.grid(True) # show grid on plot

plt.subplot(3, 1, 2) # subplot 2
plt.plot(t_conv, h2_u)

plt.ylabel('h2(t)*u(t)') # label for subplot 2
plt.grid(which ='both') # use major and minor grids


plt.subplot(3, 1, 3) # subplot 3
plt.plot(t_conv, h3_u)
plt.ylabel('h3(t)*u(t)') # label for subplot 2
plt.grid(which ='both') # use major and minor grids


h1 = (1/2*(1-np.exp(-2*t_conv))*u(t_conv)-1/2*(1-np.exp(-2*(t_conv-6)))*u(t_conv-6))
h2 = (t_conv-2)*u(t_conv-2) - ((t_conv-6)*u(t_conv-6))
h3 = (1/w0)*np.sin(w0*t_conv)*u(t_conv)

# Code for plots
plot2 = plt.figure(figsize =(12 ,8)) # start a new figure , with
                                     # a custom figure size
plt.subplot(3 ,1 ,1) # subplot 1: subplot format (row , column , number )
plt.plot(t_conv, h1) # choosing plot variables for x and y axes
plt.title('Lab 4 Hand Convolution Plots') # title for entire figure

# (all three subplots )
plt.ylabel('h1(t)*u(t)') # label for subplot 1
plt.grid(True) # show grid on plot

plt.subplot(3, 1, 2) # subplot 2
plt.plot(t_conv, h2)

plt.ylabel('h2(t)*u(t)') # label for subplot 2
plt.grid(which ='both') # use major and minor grids


plt.subplot(3, 1, 3) # subplot 3
plt.plot(t_conv, h3)
plt.ylabel('h3(t)*u(t)') # label for subplot 2
plt.grid(which ='both') # use major and minor grids



