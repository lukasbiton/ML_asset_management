# lambda in [eMin,eMax] has random behavior, lambda outside [eMin,eMax] has non-
# random behavior, specifically lambda in [0,eMax] is noise.

''' Making the Marcenko-Pastur pdf'''
import numpy as np
import pandas as pd
#-------------------------------------------------------------------------------
def mpPDF(var,q,pts):
    # Marcenko-Pastur pdf
    # var is sigma**2
    # q = T/N where X is TxN and lambda is the eigenvalues of C=1/T * X'X
    # pts is precision
    # Use the maximum and minimum fomula value for the eignvalues
    eMin, eMax=var*(1-(1./q)**.5)**2,var*(1+(1./q)**.5)**2
    eVal=np.linspace(eMin,eMax,pts)
    pdf = q/(2*np.pi*var*eVal)*((eMax-eVal)*(eVal-eMin))**.5
    pdf = pd.Series(pdf,index=eVal)
    return pdf


''' Testing the Marcenko-Pastur pdf'''
from sklearn.neighbors import KernelDensity
#-------------------------------------------------------------------------------
def getPCA(matrix):
    # Get eVal, eVec form a Hermitian matrix
    eVal,eVec = np.linalg.eigh(matrix) # Gets eigenvalues and eigenvectors of a matrix
    indices = eVal.argsort()[::1] # arguments for sorting eVal desc
    eVal,eVec=eVal[indices],eVec[indices]
    eVal=np.diagflat(eVal) # flattens inputs into diagonal matrix
    return eVal,eVec
#-------------------------------------------------------------------------------
def fitKDE(obs,bWidth=.25,kernel='gaussian',x=None):
    # Fit kernel to a series of obs, and derive the prob of obs
    # x is the array of values on which the fit KDE will be evaluated
    if len(obs.shape)==1:obs=obs.reshape(-1,1) # -1 means we let numpy figure out that column size, effectively this returns a column of singleton arrays
    kde=KernelDensity(kernel=kernel,bandwidth=bWidth).fit(obs)
    if x is None:x=np.unique(obs).reshape(-1,1)
    if len(x.shape)==1:x=x.reshape(-1,1)
    logProb=kde.score_samples(x) # evealuate log(density) on x
    pdf=pd.Series(np.exp(logProb),index=x.flatten())
    return pdf
#-------------------------------------------------------------------------------
# Compute pdf using mpPDF and fitKDE and compare
if False:
    x=np.random.normal(size=(10000,1000))
    eVal0,eVec0=getPCA(np.corrcoef(x,rowvar=0))
    # Marcenko-Pastur pdf
    pdf0=mpPDF(1.,q=x.shape[0]/float(x.shape[1]),pts=1000)
    # Empirical pdf
    pdf1=fitKDE(np.diag(eVal0),bWidth=.01)


'''Plot the results'''
import matplotlib.pyplot as plt
#-------------------------------------------------------------------------------
if False:
    plt.plot(pdf0,'b',label='Marcenko-Pastur')
    plt.plot(pdf1,'r--',label='Empirical KDE')
    plt.legend()
    plt.show()
