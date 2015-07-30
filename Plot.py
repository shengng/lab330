import numpy as np
import matplotlib.pylab as plt
import csv
import scipy
from scipy.fftpack import fft
from scipy.optimize import curve_fit
from pylab import *
from PyDAQmx import *
# ------------------------parameters--------------------------------------
filename="F:\\temp\\test_PM08-43-03.txt"
x_label='time(s)'
y_label='voltage(V)'
title='System power in long time and behind the waveplate and PBS'
q=1 #sampling period
# --------------------------------------------------------------
a=open(filename,'rb')
value=[[],[],[]]
for i in csv.reader(a):
    tmp=i[0].split("\t")
    if len(i)>0:
        for j in range(3):
            value[j].append(float(tmp[j]))
    else:
        break
a.close()

print(value[1][60])
# ---------------------------calculation--------------------------
offset0= np.average(np.array(value[0][:14]))
offset1= np.average(np.array(value[1][:14]))
offset2= np.average(np.array(value[2][:14]))
value0=(np.array(value[0])-offset0)/value[0][15]
value1=(np.array(value[1])-offset1)/value[1][60]
value2=(np.array(value[2])-offset2)/value[2][15]
# -----------------------------------------------------
x=np.arange(1,len(value[0])+1)*q

# determination=[]
# for i in range(28,40):
#     coeff=np.polyfit(x,value1,i)
#     p=np.poly1d(coeff)
#     temp=[]
#     for j in range(len(x)):
#         temp.append(p(x[j]))
#     slope, intercept, r_value, p_value, std_err = stats.linregress(temp,value1)
#     # print "Coefficients are:",coeff
#     # print "determination:", r_value**2
#     determination.append(r_value**2)
# ma=max(determination)
# print determination.index(ma)

def func(x,a,b,c): return a*np.sin(c*x)+b
popt,pcov=curve_fit(func,x,value1)
print popt
fitting_curve=[]
for i in range(len(x)): fitting_curve.append(func(x[i],popt[0],popt[1],popt[2]))

# ---------------------------------------------------------------------------
plt.plot(x,value0,"ro",label='another_beam_of_PBS')
plt.plot(x,value1,"g^",label='one_beam_of_PBS')
plt.plot(x,fitting_curve,linestyle='--',color='blue')
plt.xlabel(x_label)
plt.ylabel(y_label)
plt.axis([0,33000,0.6,1.1])
plt.title(title)
# plt.legend()
plt.show()
# -------------------
# Number of samplepoints

N = len(value1)
# sample spacing
T = 1.0
# x = np.linspace(0.0, N*T, N)
y = value1
yf = fft(y)
# xf = scipy.fftpack.rfft(np.fft.fftfreq(N, T))
# plt.plot(xf, 2.0/N * np.abs(yf),'ro')
# # plt.axis([-100,100,0,16])
# plt.grid()
# plt.show()

f=scipy.fftpack.fftfreq(N,1.0)

windowWidth = 27
polynomialOrder = 3
smoothY = sgolayfilt(y, polynomialOrder, windowWidth)
plt.plot()