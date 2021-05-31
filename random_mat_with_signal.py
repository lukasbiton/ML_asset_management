# In general, not all eigenvectors will be random
# Only a nFact columns among nCols contain some signal, and we add a purely
# random matrix on top

'''Add signal to a random covariance matrix.'''
import numpy as np
import pandas as pd
#-------------------------------------------------------------------------------
def getRndCov(nCols,nFacts):
    w=np.random.normal(size=(nCols,nFacts))
    cov=np.dot(w,w.T) # random cov matrix, however not full rank
    cov+=np.diag(np.random.uniform(size=nCols)) # full rank cov
    return cov

'''Get the correlation matrix.'''
#-------------------------------------------------------------------------------
def cov2corr(cov):
    # Derive the correlation matrix from a covariance matrix
    std=np.sqrt(np.diag(cov)) # standard std formula
    corr = cov/np.outer(std,std) # standard Pearson corr
    corr[corr<-1],corr[corr>1]=-1,1 # numerical error # corr can never be > |1|
    return corr

'''Get the eigenvalues and eigenvectors for a matrix with very high noise
and very low signal.'''
from marcenko_pastur import getPCA
#-------------------------------------------------------------------------------
alpha,nCols,nFact,q=.995,1000,100,10 # parameters
cov=np.cov(np.random.normal(size=(nCols*q,nCols)),rowvar=0) # noise
cov=alpha*cov+(1-alpha)*getRndCov(nCols,nFact) # noise + signal
corr0=cov2corr(cov) # get the correlation
eVal0,eVec0=getPCA(corr0)

from scipy.optimize import minimize
from marcenko_pastur import mpPDF
from marcenko_pastur import fitKDE
#-------------------------------------------------------------------------------
def errPDFs(var,eVal,q,bWidth,pts=1000):
    # Fit error
    pdf0 = mpPDF(var,q,pts) # theoretical pdf
    pdf1=fitKDE(eVal,bWidth,x=pdf0.index.values) # empirical pdf
    sse=np.sum((pdf1-pdf0)**2)
    return sse
#-------------------------------------------------------------------------------
def findMaxEval(eVal,q,bWidth):
    # Find max random eVal by fitting Marcenko's dist
    # Use minimize from scipy.optimize to minimize the difference between
    # theoretical and empirical pdf. The argmin of that function is the variance
    # is the var
    out=minimize(lambda *x:errPDFs(*x),.5,args=(eVal,q,bWidth),bounds=((1E-5,1-1E-5),))
    if out['success']: var=out['x'][0]
    else: var=1
    eMax