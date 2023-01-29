# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 14:04:14 2023

@author: Ignotus
"""

# ###############################################################
# #
# Chris Pierson #
# ECE 351 #
# Lab 2 #
# Due 1/31/23 #
#  #
# #
# ###############################################################

import numpy as np
import matplotlib.pyplot as plt

#plt.rcParams.update({'fontsize': 14})

#################################################################
##  PART 1 ##

steps = 1e-2
t = np.arange(-5,10 + steps, steps)

print('Number of elements: len(t) = ', len(t), '\nFirst Element: t[0] = ', t[0], '\nLast Element: t[len(t) - 1 = ',t[len(t)-1])

def func1(t):
    y = np.cos(t)
    return y

y = func1(t)

plt.figure(figsize = (10, 7))
#plt.subplot(3, 1, 1)
plt.plot(t, y)
plt.grid()
plt.ylabel('y(t) with Good Resolution')
plt.show()

#################################################################
##  PART 2 ##

def step1(t):
    y = np.zeros(t.shape)
    
    for i in range(len(t)):
        if t[i] < 0:
            y[i] = 0
        else:
            y[i] = 1
    return y

y = step1(t)
plt.plot(t,y)
plt.grid()
plt.title('User Defined Step Function')
plt.show()

def ramp(t):
    y = np.zeros(t.shape)
    
    for i in range(len(t)):
        if t[i] < 0:
            y[i] = 0
        else:
            y[i] = t[i]
    return y

y = ramp(t)
plt.plot(t,y)
plt.grid()
plt.title('User Defined Ramp Function')
plt.show()

#y = ramp(t) - ramp(t-3) + 5 * step1(t-3) - 2 * step1(t-6) - 2 * ramp(t-6)

def func2(t):
    y = ramp(t) - ramp(t-3) + 5 * step1(t-3) - 2 * step1(t-6) - 2 * ramp(t-6)
    return y

y = func2(t)

#plt.subplot(3, 1, 2)
plt.plot(t, y)
plt.grid()
plt.title('User Defined Function')
plt.ylabel('y')
plt.xlabel('t')
plt.show

#################################################################
##  PART 3 ##
plt.clf()
#Time Reversal
# t = np.arange(-10,5 + steps, steps)
# y = func2(-t)
# plt.plot(t, y)
# plt.grid()
# plt.title('Time Reversal')
# plt.xlabel('t')
# plt.ylabel('y')
# plt.show

#Time Shift 1
# t = np.arange(-1,14 + steps, steps)
# y = func2(t-4)
# plt.plot(t, y)
# plt.grid()
# plt.title('Time Shift 1')
# plt.xlabel('t')
# plt.ylabel('y')
# plt.show

#Time Shift 2
# t = np.arange(-14,1 + steps, steps)
# y = func2(-t-4)
# plt.plot(t, y)
# plt.grid()
# plt.title('Time Shift 2')
# plt.xlabel('t')
# plt.ylabel('y')
# plt.show

#Time Scale 1
# t = np.arange(-10,20 + steps, steps)
# y = func2(t/2)
# plt.plot(t, y)
# plt.grid()
# plt.title('Time Scale 1')
# plt.xlabel('t')
# plt.ylabel('y')
# plt.show

#Time Scale 2
# t = np.arange(-10,5 + steps, steps)
# y = func2(2*t)
# plt.plot(t, y)
# plt.grid()
# plt.title('Time Scale 2')
# plt.xlabel('t')
# plt.ylabel('y')
# plt.show


steps =1e-2

t=np.arange(-5,10+steps,steps)
y= func2(t)
dt = np.diff(t)
dy = np.diff(y, axis = 0)/dt


plt.figure(figsize = (15,7))
#plt.plot(t,y, '--', label = 'y(t)')
plt.plot(t,y, 'r--')
plt.plot(t[range(len(dy))],dy)
#plt.plot(t[range(len(dy))], dy, label='dy(t)/dt')
plt.grid()
plt.ylabel('y(t)')
plt.title('Derivative overlayed with User Defined Function')
plt.xlabel('t')
plt.ylim([-2,10])
plt.show()

