### Buffmo.py
### Morgan Henderson, Fall 2021
### "Buffmo" the Autonomous Bison: Kalman Filter & Truth Model Testing
### Originally created for ASEN 5044 (CU Boulder, Fall 2021)

from numpy import array, zeros, sin, cos, block, eye
from numpy.random import multivariate_normal as mvn, uniform as uni
from scipy.linalg import inv, expm
from scipy.stats import chi2
from matplotlib import pyplot as plt
plt.rcParams["mathtext.fontset"] = "cm"
plt.ion()

# Define parameters
tmax = 100
dt = 0.1
nt = int(tmax/dt)
Om = .75
f = .2
qW = 1
qR = 1

# Define or compute DT inputs & matrices
ts = array([dt*t for t in range(1,nt+1)])
u = array([f*cos(Om*ts),-f*sin(Om*ts)]).T
A = array([[0,1,0,0],[0,0,0,0],[0,0,0,1],[0,0,0,0]])
F = array([[1,dt,0,0],[0,1,0,0],[0,0,1,dt],[0,0,0,1]])
G = array([[.5*dt**2,0],[dt,0],[0,.5*dt**2],[0,dt]])
Gam = array([[0,0],[1,0],[0,0],[0,1]])
W = qW*array([[1,.1],[.1,1]])
Z = dt*block([[-A,Gam@W@Gam.T],[zeros((4,4)),A.T]])
Zhat = expm(Z)
Q = F@Zhat[:4,4:]
H = array([[1,0,0,0],[0,0,1,0]])
R = qR*array([[1,.1],[.1,1]])

# Loop over many rounds of truth model testing
nTMT = 50
NEES = zeros((nTMT,nt))
NIS = zeros((nTMT,nt))
for n in range(nTMT):

    # Simulate a set of noisy truth data & measurements
    xT = zeros((nt+1,4))
    ySim = zeros((nt,2))
    xT[0] = array([uni(-10,10),uni(-2,2),uni(-10,10),uni(-2,2)])
    for t in range(nt):
        xT[t+1] = F@xT[t]+G@u[t]+mvn(zeros((4)),Q)
        ySim[t] = H@xT[t+1]+mvn(zeros((2)),R)

    # Initialize the Kalman filter
    xh = zeros((nt+1,4))
    P = zeros((nt+1,4,4))
    xh[0] = array([0,0,0,0])
    P[0] = 1e3*eye(4)

    # Perform Kalman filter updates, compute NEES/NIS statistics
    for t in range(nt):

        xhm = F@xh[t]
        Pm = F@P[t]@F.T+Q
        Kk = Pm@H.T@inv(H@Pm@H.T+R)
        xh[t+1] = xhm+Kk@(ySim[t]-H@xhm)
        P[t+1] = (eye(4)-Kk@H)@Pm

        xerr = xT[t+1]-xh[t+1]
        yerr = ySim[t]-H@xhm
        Sk = H@Pm@H.T+R
        NEES[n,t] = xerr.T@inv(P[t+1])@xerr
        NIS[n,t] = yerr.T@inv(Sk)@yerr

# Compute NEES/NIS test quantities
alpha = 0.05
exbar = NEES.mean(axis=0)
eybar = NIS.mean(axis=0)
rNEES = array([chi2.ppf(alpha/2,nTMT*4),chi2.ppf(1-alpha/2,nTMT*4)])/nTMT
rNIS = array([chi2.ppf(alpha/2,nTMT*2),chi2.ppf(1-alpha/2,nTMT*2)])/nTMT

# Plot simulated measurements & true states vs. time
fig, ax = plt.subplots()
ax.plot(ySim[:,0],ySim[:,1],'ko',ms=7.5,ls='none',mfc='none',\
    label='Simulated Measurements')
ax.plot(xT[1:,0],xT[1:,2],'k-',lw=3,label='Truth Model Data')
ax.plot(xh[1:,0],xh[1:,2],'r--',lw=3,label='KF Estimates')
ax.set_xlabel(r"Easting Position $\xi$, m",fontsize=36)
ax.set_ylabel(r"Northing Position $\eta$, m",fontsize=36)
ax.tick_params(labelsize=24,direction="in")
ax.legend(fontsize=24,markerscale=1.5)
fig.subplots_adjust(left=.12,right=.96,top=.98,bottom=.12)
fig.show()

# Plot estimation errors & 2-sigma bounds vs. time
labels = [r"$\epsilon_{\xi}$, m",r"$\epsilon_{\dot{\xi}}$, m s$^{-1}$",\
    r"$\epsilon_{\eta}$, m",r"$\epsilon_{\dot{\eta} }$, m s$^{-1}$"]
fig, axs = plt.subplots(2,2)
for i in range(2):
    for j in range(2):
        ij = 2*i+j
        ex = xT[1:,ij]-xh[1:,ij]
        hrange = (ex[10:].max()-ex[10:].min())
        mid = ex[10:].mean()
        lime = [mid-1.01*hrange,mid+1.01*hrange]
        axs[i,j].plot(ts,ex,'r-',lw=2.5)
        axs[i,j].plot(ts,2*P[1:,ij,ij]**.5,'k--',lw=1.5)
        axs[i,j].plot(ts,-2*P[1:,ij,ij]**.5,'k--',lw=1.5)
        axs[i,j].axis([0,dt*(nt+1),lime[0],lime[1]])
        axs[i,j].set_xticks([0,25,50,75,100])
        axs[i,j].tick_params(labelsize=24,direction="in")
        if i==1: axs[i,j].set_xlabel(r"Time (s)",fontsize=36)
        else: axs[i,j].set_xticklabels([])
        axs[i,j].set_ylabel(labels[ij],fontsize=36)
fig.subplots_adjust(left=.1,right=.96,top=.98,bottom=.12,wspace=.3,hspace=.01)
fig.show()

# Plot average NEES statistic vs. time & 95% X^2 bounds
hrange = (rNEES[1]-rNEES[0])/2
mid = (rNEES[1]+rNEES[0])/2
lim = [mid-2.5*hrange,mid+2.5*hrange]
fig, ax = plt.subplots()
ax.plot(ts,exbar,'ro',mfc='none')
ax.plot(ts,rNEES[0]+zeros((nt)),'k--',lw=1.5)
ax.plot(ts,rNEES[1]+zeros((nt)),'k--',lw=1.5)
ax.axis([0,dt*(nt+1),lim[0],lim[1]])
ax.tick_params(labelsize=24,direction="in")
ax.set_xticks([0,25,50,75,100])
ax.set_xlabel(r"Time (s)",fontsize=36)
ax.set_ylabel(r"Average NEES, $\bar{\epsilon}_x$",fontsize=36)
fig.subplots_adjust(left=.12,right=.96,top=.98,bottom=.12)
fig.show()

# Plot average NIS statistic vs. time & 95% X^2 bounds
hrange = (rNIS[1]-rNIS[0])/2
mid = (rNIS[1]+rNIS[0])/2
lim = [mid-2.5*hrange,mid+2.5*hrange]
fig, ax = plt.subplots()
ax.plot(ts,eybar,'ro',mfc='none')
ax.plot(ts,rNIS[0]+zeros((nt)),'k--',lw=1.5)
ax.plot(ts,rNIS[1]+zeros((nt)),'k--',lw=1.5)
ax.axis([0,dt*(nt+1),lim[0],lim[1]])
ax.tick_params(labelsize=24,direction="in")
ax.set_xticks([0,25,50,75,100])
ax.set_xlabel(r"Time (s)",fontsize=36)
ax.set_ylabel(r"Average NIS, $\bar{\epsilon}_y$",fontsize=36)
fig.subplots_adjust(left=.12,right=.96,top=.98,bottom=.12)
fig.show()