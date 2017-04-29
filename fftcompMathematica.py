from pylab import *
import numpy as np
from scipy.optimize import curve_fit
from scipy.fftpack import fft
from scipy import signal
import sys
import cPickle as pickle


def anaAC(t):
    gamma = 1



    def fkt(x):
        
        x1 = exp(-gamma*abs(x))
        
        #xx = exp(-gamma*x*x)
        return x1


    res = []
    for i in range(0,len(t)):        
        res.append(fkt(t[i]))
    return res


def anaACFFT(fre):
    
    gamma = 1

    def fkt(fr):
        
        k = 2*pi*fr
        
        x1 = sqrt(2.0/pi)/(1+ (k*k))
        x1r = sqrt(2*pi)*x1

        #Mathematica result
        #xx = (1.0/sqrt(2))*exp(-gamma*k*k/4.0)
        
        #Skalierung
        #xrs = sqrt(2*pi)*xx


        return x1r

    res = []

    for i in range(0,len(fre)):
        res.append(fkt(fre[i]))
    return res






#dat1 = np.loadtxt('trajectory.txt')
#t = [row[0] for row in dat1]
#x = [row[1] for row in dat1]

#t = pickle.load( open( "timeTra.p", "rb" ) )
#x = pickle.load( open( "xTra.p", "rb" ) )

#create data

N = 100001
T = 5000.0
t = np.linspace(-T/2,T/2,N)
r = anaAC(t)

plt.plot(t,r)
plt.show()


dt = t[1]-t[0]

#fftanaAC = np.fft.fft(r) * dt
#freq = np.fft.fftfreq(N,dt)

fftanaAC = fft(r) * dt
freq = fftfreq(N,dt)


freq = freq[:N/2 + 1]

afft = anaACFFT(freq)



plt.plot(freq, np.abs(fftanaAC[:N/2+1]),'o')
plt.plot(freq,afft,'r')
plt.show()


#plt.plot(t,x)
#plt.show()

#coVar = np.cov(x)
#autocorr = signal.fftconvolve(x, x[::-1], mode='full')
#result = np.correlate(x, x, mode='full')
#plt.plot(0.001*np.arange(-len(x)+1,len(x)),autocorr/len(x))
#plt.plot(np.arange(-len(x)+1,len(x)),result)
#plt.plot(t,r)
#plt.show()

#plt.plot(fft(autocorr/len(x)))

#y = fft(r)
#y_a = np.abs(y)




#plt.plot(y_a)


#plt.show()

#fs = 1./0.001

#print var(x)

#f, Pxx_den = signal.welch(x, fs)
#f, Pxx_den = signal.periodogram(x, fs)
#plt.plot(f, Pxx_den)
#plt.ylim([0.5e-3, 1])
#plt.xlabel('frequency [Hz]')
#plt.ylabel('PSD [V**2/Hz]')
#plt.show()
