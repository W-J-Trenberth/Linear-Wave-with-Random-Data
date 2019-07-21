# -*- coding: utf-8 -*-
"""
@author: William John Trenberth
https://www.maths.ed.ac.uk/~wjtrenberth/
https://github.com/W-J-Trenberth

Python code for plotting solutions to the 2d wave equation on [0,1]\times[0,1]
with random initial data.

The aim in the near future is to merge this with the 1d version and perhaps
generalise to higher dimensions.
"""



import scipy.fftpack as fft
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def main():

    c=1 # The wave speed.
    b=0 # The dissipation.
    a=0 # The linear potential.
    dt=0.002 # The time between frames.
    s = 2.8 # The regularity of the random initial data. 
    
    N = 2**7 # Meshsize
    fps = 25 # frame per sec
    frn = 2000 # frame number of the animation
    
    x = np.linspace(0,1,N)
    x, y = np.meshgrid(x, x)
    
    u_0 = random_initial_data(N,s)
    v_0 = random_initial_data(N,s+1)
        
    maxu_0 = np.max(np.abs(u_0))
    
    def update_plot(frame_number):
        plot[0].remove()
        z, v = linear_wave_solver(frame_number*dt, u_0, v_0, c, b, a)
        plot[0] = ax.plot_surface(x, y, z, cmap="magma")
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    plot = [ax.plot_surface(x, y, u_0, color='0.75', rstride=1, cstride=1)]
    ax.set_zlim(-1.1*maxu_0, 1.1*maxu_0)
    ani = animation.FuncAnimation(fig, update_plot, frn, interval=1000/fps)
    
    fn = '2d_Wave_12'
    ani.save(fn+'.mp4',writer='ffmpeg',fps=fps)

def linear_wave_solver(t, u_0, v_0, c, b, a):
    '''A function to solve the linear wave equation
    $$\partial_t^2u = c^2\Delta u -\partial_t u -au$$ on $[0,1]\times [0,1]$
    with zero Dirichlet boundary conditions and initial data 
    $(u(0), \partial_t u(0)) = (u_0,v_0)$. This function uses a spectral method, 
    inparticular the fast sine transform.
    
    Parameters
    -----------------------------
    t = The time the solution is computed at, a real number.
    u_0 =  The inital displacement, an array of real numbers.
    v_0 = The inital velcoty, a real number, an array of real numbers.
    c = The wave speed, a real number.
    b = The dissipation, a real number.
    a = The linear potential, a real number.
    
    Returns
    ------------------------------
    
    u = The displacement at time t, an array.
    v = The velocity at time t, an array.
    '''
    l = len(u_0[0,:])
    f = u_0[1:l-1, 1:l-1]
    g = v_0[1:l-1, 1:l-1]
    
    
    Sf = fft.dstn(f, type = 1, norm = 'ortho')
    Sg = fft.dstn(g, type = 1, norm = 'ortho')
    
    n = np.arange(1,l-1)
    n = np.tile(n, (l-2,1))
    
    m = np.arange(1,l-1) + 0j
    m = np.tile(m, (l-2,1)).transpose() + 0j
    
    lnmminus = (-b - np.sqrt(b**2 - 4*(c**2*(n**2+m**2)*np.pi**2 + a)))/2
    lnmplus = (-b + np.sqrt(b**2 - 4*(c**2*(n**2+m**2)*np.pi**2 + a)))/2
    
    Anmplus = (lnmminus*Sf - Sg)/(lnmminus - lnmplus)
    Anmminus = (Sg - lnmplus*Sf)/(lnmminus - lnmplus)
    
    #The Fourier transform of the solution
    Su = Anmplus*np.exp(lnmplus*t) + Anmminus*np.exp(lnmminus*t)
    Sv = lnmplus*Anmplus*np.exp(lnmplus*t) + lnmminus*Anmminus*np.exp(lnmminus*t)
    
    u = fft.idstn(Su, type = 1, norm = 'ortho')
    v = fft.idstn(Sv, type = 1, norm = 'ortho')
    
    #We need to add the zeros to the boundary of u and v.
    u = np.pad(u, 1, mode = 'constant')
    v = np.pad(v, 1, mode = 'constant')
    
    #Convert to real to get rid of the superfluous + \eps j terms where \eps is
    #a small real number
    
    u = np.real(u)
    v = np.real(v)
    
    return u, v

def random_initial_data(N, s):
    '''A function generating a random function on $[0,1]\times [0,1]$ with zero
    boundary conditions ampled from the random Fourier series
    $$\sum\limits_{n,m\geq 1}^N \frac{g_{nm}(\omega)}{\langle (n,m)\rangle^s}$$
    
    Parameters
    ----------------------
    N = The Fourier truncation parameter
    s = The regularity of the random data
    
    Returns
    ---------------------
    f =  a random Fourier series as described above.
    '''
    n = np.arange(1,N-1)
    n = np.tile(n, (N-2,1)) + 0j
    m = np.arange(1,N-1)
    m = np.tile(m, (N-2,1)).transpose() + 0j
    
    Ff = (np.random.randn(N-2, N-2) + np.complex(0,1)*np.random.randn(N-2, N-2))/((n**2 + m**2 + 1)**(s/2))
    f = fft.idstn(Ff, type = 1, norm = 'ortho')
    
    f = np.pad(f, 1, mode = 'constant')
    f = np.real(f)
    
    return f


def pulse(x,y):
    '''A pulse centered around (0.5,0.5) for testing.
    '''
    out = 1e43*np.exp(-100*((x-0.5)**2+(y-0.5)**2+1))
    n,m = np.shape(out)
    out[:,0] = 0
    out[:,n-1] = 0
    out[0,:] = 0
    out[n-1,:] = 0
    
    return out

def g(x,y):
    '''A eigenvector of the Laplacian for testing.
    '''
    out = np.sin(2*np.pi*x)*np.sin(2*np.pi*y)
    n,m = np.shape(out)
    out[:,0] = 0
    out[:,n-1] = 0
    out[0,:] = 0
    out[n-1,:] = 0
    
    return out

if __name__  == "__main__": main()
    
    
    