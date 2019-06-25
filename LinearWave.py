# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 23:05:43 2019

@author: William
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack as fft
from matplotlib import animation



def main():
    N = 2**10   #spatial discretization
    reg = 1.5   #spatial regularity of inital data
    c = 1       #wave speed
    b = -0.02   #damping term
    a = 0       #constant potential
    savelocation = 'C:'
    
    x = np.linspace(0,1,N)  #discretize unit interval
    u_0 = random_initial_data(N, reg)
    v_0 = random_initial_data(N, reg + 1) 
    
    fig = plt.figure()
    ax = plt.axes(xlim=(0, 1), ylim=(-5, 5))
    line, = ax.plot([], [], lw=2)
    
    # initialization function: plot the background of each frame
    def init():
        line.set_data([], [])
        return line,
    
    # animation function.  This is called sequentially
    def animate(i):
        u,v = linear_wave_solution(0.01*i, u_0, v_0, c= c, b=b, a = a)
        line.set_data(x, u)
        return line,
    
    # call the animator.  
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=2000, interval=20)
    
    anim.save(savelocation, fps=30)
    
    
def linear_wave_solution(t, u_0, v_0, c = 1, b = 0, a = 0):
    '''This function uses the fast sine transform to return the soluton to the
    linear wave equation $\partial_t^2 u = \Delta u - b\partial_t u - au$
    given an initial displacement, $u_0$ and an initial velocity, $v_0$.
    '''
    
    #The DST doesn't take as input the zeros at the start and end of the 
    #displacement and velocity vectors. Below k needs to be complex so np.sqrt
    #will return a complex number instead of an error
    N = np.size(u_0)-1  
    k = np.arange(1,N) + 0j 
    f = u_0[1:len(u_0)-1]   
    g = v_0[1:len(u_0)-1]   
    
    Sf =  fft.dst(f, type = 1)
    Sg =  fft.dst(g, type = 1)
    
    lkminus = (-b - np.sqrt(b**2 - 4*(c**2*k**2*np.pi**2 + a)))/2
    lkplus = (-b + np.sqrt(b**2 - 4*(c**2*k**2*np.pi**2 + a)))/2
    
    Akplus = (lkminus*Sf - Sg)/(lkminus - lkplus)
    Akminus = (Sg - lkplus*Sf)/(lkminus - lkplus)
    
    #The Fourier transform of the solution
    Su = Akplus*np.exp(lkplus*t) + Akminus*np.exp(lkminus*t)
    Sv = lkplus*Akplus*np.exp(lkplus*t) + lkminus*Akminus*np.exp(lkminus*t)
    
    u = fft.idst(Su, type = 1)/(2*N)
    v = fft.idst(Sv, type = 1)/(2*N)
    
    #We need to add the zeros back to the start and end of the displacement and
    #velocity vectors. 
    u= np.insert(u, 0, 0)
    u = np.insert(u, len(u), 0)
    v= np.insert(v, 0, 0)
    v = np.insert(v, len(v), 0)
    
    #Convert to real to get rid of the superfluous + 0j terms
    u = np.real(u)
    v = np.real(v)
    
    return u,v

def random_initial_data(N,s):
    k = np.arange(1,N-1)
    Ff = np.random.randn(N-2)/(k**s+1)
    f = fft.idst(Ff, type = 1)
    
    f= np.insert(f, 0, 0)
    f = np.insert(f, len(f), 0)
    
    return f

def pulse(x):
    '''A pulse to test the linear_wave_solution function to make sure the 
    solutions have the correct properties such as obeying the reflection 
    principle.
    '''
    output = np.zeros(np.size(x))
    for i in range(0,np.size(x)):
        if x[i]<0.4:
            output[i] = 0
        elif x[i]<0.6:
            output[i] = (x[i]-0.4)*(0.6-x[i])
        else:
            output[i] = 0 
    return 400*output
    

if __name__ == "__main__": main()
