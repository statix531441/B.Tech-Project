import numpy as np
import math
import cmath

j = complex(0,1)

def fft(P, offset=0):
    N = len(P) #assuming this is a power of 2
    if N==1:
        return P
    w = cmath.exp(-j*2*math.pi/N)
    Pe,Po = P[::2],P[1::2]
    ye,yo = fft(Pe,offset),fft(Po,offset)
    y = np.zeros(N).astype('complex128')
    for k in range(int(N/2)):
        y[k] = ye[k] + (w**(k-offset))*yo[k]
        y[k+int(N/2)] = ye[k] - (w**(k-offset))*yo[k]
    return y

def fft2row(P,offset):
    N = len(P[0]) #assuming this is a power of 2
    if N==1:
        return P
    w = cmath.exp(-j*2*math.pi/N)
    Pe,Po = P[:,::2],P[:,1::2]
    ye,yo = fft2row(Pe,offset),fft2row(Po,offset)
    y = np.zeros((len(P),len(P[0]))).astype('complex128')
    for k in range(int(N/2)):
        y[:,k] = ye[:,k] + (w**(k-offset))*yo[:,k]
        y[:,k+int(N/2)] = ye[:,k] - (w**(k-offset))*yo[:,k]
    return y


def fft2(img, xoff=0, yoff=0):
    return fft2row(fft2row(img,xoff).T,yoff).T


def ifft(P,offset=0,summationCompleted=True):
    N = len(P) #assuming this is a power of 2
    if N==1:
        return P
    w = cmath.exp(j*2*math.pi/N)
    Pe,Po = P[::2],P[1::2]
    ye,yo = ifft(Pe,offset,False),ifft(Po,offset,False)
    y = y = np.zeros(N).astype('complex128')
    for n in range(int(N/2)):
        y[n] = ye[n] + (w**(n))*yo[n]
        y[n+int(N/2)] = ye[n] - (w**(n))*yo[n]
    if not summationCompleted:
        return y
    temp = np.array(y)/N
    for n in range(len(temp)):
        temp[n] *= cmath.exp(-j*2*math.pi*n*offset/N)
    return temp

def ifft2row(P,offset=0,summationCompleted=True):
    N = len(P[0]) #assuming this is a power of 2
    if N==1:
        return P
    w = cmath.exp(j*2*math.pi/N)
    Pe,Po = P[:,::2],P[:,1::2]
    ye,yo = ifft2row(Pe,offset,False),ifft2row(Po,offset,False)
    y = np.zeros((len(P),len(P[0]))).astype('complex128')
    for n in range(int(N/2)):
        y[:,n] = ye[:,n] + (w**(n))*yo[:,n]
        y[:,n+int(N/2)] = ye[:,n] - (w**(n))*yo[:,n]
    if not summationCompleted:
        return y
    temp = y/N
    for n in range(len(temp[0])):
        temp[:,n] = temp[:,n]*cmath.exp(-j*2*math.pi*n*offset/N)
    return temp

def ifft2(img, xoff=0, yoff=0):
    return ifft2row(ifft2row(img,xoff).T,yoff).T


def fft2test(img,xoff=0,yoff=0):
    temp = np.apply_along_axis(fft,1,img,xoff)
    return np.apply_along_axis(fft,0,temp,yoff)

def ifft2test(img,xoff=0,yoff=0):
    temp = np.apply_along_axis(ifft,1,img,xoff)
    return np.apply_along_axis(ifft,0,temp,yoff).real



def conv(a, b):
    m1 = len(a)
    m2 = len(b)
    m = 2**(math.ceil(math.log2(m1+m2-1)))
    z1 = np.zeros((m))
    z2 = np.zeros((m))
    z1[:m1] = a
    z2[:m2] = b
    fd1 = fft(z1)
    fd2 = fft(z2)
    fd3 = np.multiply(fd1,fd2)
    c = ifft(fd3).real[:m1+m2-1]
    return c[m2-1:-m2+1]

def conv2(a, b):
    m1 = len(a[0])
    m2 = len(b[0])
    n1 = len(a)
    n2 = len(b)
    m = 2**(math.ceil(math.log2(m1+m2-1)))
    n = 2**(math.ceil(math.log2(n1+n2-1)))
    z1 = np.zeros((n,m))
    z2 = np.zeros((n,m))
    z1[:n1,:m1] = a
    z2[:n2,:m2] = b
    fd1 = fft2(z1)
    fd2 = fft2(z2)
    fd3 = np.multiply(fd1,fd2)
    c = ifft2(fd3).real[:n1+n2-1, :m1+m2-1]
    return c[n2-1:-n2+1, m2-1:-m2+1]














    





