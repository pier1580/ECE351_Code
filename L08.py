# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 17:28:13 2023

@author: Ignotus
"""

# ###############################################################
# 
# Chris Pierson #
# ECE 351 #
# Lab 8 #
# Due 3/21/23 #
# 
# ###############################################################

import numpy as np
import matplotlib.pyplot as plt
#import scipy.signal as sig

#################################################################
                        ##  PART 1 ##
#################################################################
steps = 1e-2
t = np.arange(0, 20 + steps, steps) 

T = 8
w0 = (2*np.pi)/T
ak = 0
a0 = 0

def bk(k):
    return 2/(k*np.pi)*(1-np.cos(k*np.pi))
    
print('a0= 0, a1= 0')
for i in range(1,4):
    rounded = np.round(bk(i),3)   
    print("b",i,'= ',rounded,sep="")

def x(k,t):
    y=0
    for i in range(1,k+1):
        y += 1/2*a0 + bk(i)*np.sin(i*w0*t)
    return y

# Code for plots
plot1 = plt.figure(figsize =(12 ,8)) # start a new figure

plt.subplot(3 ,1 ,1) # subplot 1: subplot format (row , column , number )
plt.plot(t, x(1,t)) # choosing plot variables for x and y axes
plt.ylabel('k=1') # label for subplot 1
plt.grid(True) # show grid on plot

plt.title('Lab 8 plots 1-3') # title for entire figure 

plt.subplot(3, 1, 2) # subplot 2
plt.plot(t, x(3,t))
plt.ylabel('k=3') # label for subplot 2
plt.grid(True) # show grid on plot

plt.subplot(3, 1, 3) # subplot 2
plt.plot(t, x(15,t))
plt.ylabel('k=15') # label for subplot 2
plt.grid(True) # show grid on plot

plot2 = plt.figure(figsize =(12 ,8)) # start a new figure

plt.subplot(3 ,1 ,1) # subplot 1: subplot format (row , column , number )
plt.plot(t, x(50,t)) # choosing plot variables for x and y axes
plt.ylabel('k=50') # label for subplot 1
plt.grid(True) # show grid on plot

plt.title('Lab 8 plots 4-6') # title for entire figure 

plt.subplot(3, 1, 2) # subplot 2
plt.plot(t, x(150,t))
plt.ylabel('k=150') # label for subplot 2
plt.grid(True) # show grid on plot

plt.subplot(3, 1, 3) # subplot 2
plt.plot(t, x(1500,t))
plt.ylabel('k=1500') # label for subplot 2
plt.grid(True) # show grid on plot

plt.show()

