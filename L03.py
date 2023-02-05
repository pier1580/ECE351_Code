# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 14:04:14 2023

@author: Ignotus
"""

# ###############################################################
# #
# Chris Pierson #
# ECE 351 #
# Lab 3 #
# Due 2/7/23 #
# #
# #
# ###############################################################

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

#################################################################
##  PART 1 ##

steps = 1e-2
t = np.arange(0,20 + steps, steps)

def u(t):
    y = np.zeros(t.shape)
    for i in range(len(t)):
        if t[i] < 0:
            y[i] = 0
        else:
            y[i] = 1
    return y

def r(t):
    y = np.zeros(t.shape)   
    for i in range(len(t)):
        if t[i] < 0:
            y[i] = 0
        else:
            y[i] = t[i]
    return y

def f_1(t):
    y = u(t - 2) - u(t - 9)
    return y
f1 = f_1(t)

def f_2(t):
    y = np.exp(-t) * u(t)
    return y
f2 = f_2(t)

def f_3(t):
    y = r(t - 2)*(u(t - 2) - u(t - 3)) + r(4 - t)*(u(t-3)-u(t-4))
    return y
f3 = f_3(t)

# Code for plots
plot1 = plt.figure(figsize =(12 ,8)) # start a new figure , with
# a custom figure size
plt.subplot(3 ,1 ,1) # subplot 1: subplot format (row , column , number )
plt.plot(t, f1) # choosing plot variables for x and y axes
plt.title('Lab 3 Plots') # title for entire figure

# (all three subplots )
plt.ylabel('f1(t)') # label for subplot 1
plt.grid(True) # show grid on plot

plt.subplot(3, 1, 2) # subplot 2
plt.plot(t, f2)

plt.ylabel('f2(t)') # label for subplot 2
plt.grid(which ='both') # use major and minor grids


plt.subplot(3, 1, 3) # subplot 3
plt.plot(t, f3)
plt.ylabel('f3(t)') # label for subplot 2
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

t_conv = np.arange(0, 2*t[len(t)-1], steps)

f1_f2_conv = my_conv(f1,f2)
f2_f3_conv = my_conv(f2,f3)
f1_f3_conv = my_conv(f1,f3)

plt.figure( figsize = (10 , 7) )
plt.plot(t_conv, f1_f2_conv)
plt.grid()
plt.ylabel('Y')
plt.xlabel('t')
plt.title('Convolution of f1 and f2')

plt.figure( figsize = (10 , 7) )
plt.plot(t_conv, f2_f3_conv)
plt.grid()
plt.ylabel('Y')
plt.xlabel('t')
plt.title('Convolution of f2 and f3')

plt.figure( figsize = (10 , 7) )
plt.plot(t_conv, f1_f3_conv)
plt.grid()
plt.ylabel('Y')
plt.xlabel('t')
plt.title('Convolution of f1 and f3')

builtin_conv1 = sig.convolve(f1,f2)
builtin_conv2 = sig.convolve(f2,f3)
builtin_conv3 = sig.convolve(f1,f3)

plt.figure( figsize = (10 , 7) )
plt.plot(t_conv, builtin_conv1)
plt.grid()
plt.ylabel('Y')
plt.xlabel('t')
plt.title('Built-in Convolution of f1 and f2')

plt.figure( figsize = (10 , 7) )
plt.plot(t_conv, builtin_conv2)
plt.grid()
plt.ylabel('Y')
plt.xlabel('t')
plt.title('Built-in Convolution of f2 and f3')

plt.figure( figsize = (10 , 7) )
plt.plot(t_conv, builtin_conv3)
plt.grid()
plt.ylabel('Y')
plt.xlabel('t')
plt.title('Built-in Convolution of f1 and f3')

