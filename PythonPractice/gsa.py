# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 14:03:35 2022

@author: USER
"""

#3rd party Modules
import numpy as np
#import sys
#import warnings
#import matplotlib.pyplot as plt
#import scipy.integrate as integrate
#from tabulate import tabulate                       #Used for printing tables to terminal
#import sobol                                        #Used for generating sobol sequences
from SALib.sample.sobol_sequence import sample as SobolSample
import scipy.stats as sct

class options:
    def __init__(self, run = True, runSobol=True, runMorris=True, nSampSobol=100000, \
                 nSampMorris=4, lMorris=3):
        self.run = run
        if self.run == False:
            self.runSobol = False
            self.runMorris = False
        else:
            self.runSobol=runSobol                            #Whether to run Sobol (True or False)
            self.runMorris=runMorris                          #Whether to run Morris (True or False)
        self.nSampSobol = nSampSobol                      #Number of samples to be generated for GSA
        self.nSampMorris = nSampMorris
        self.lMorris=lMorris
        pass

class results:
    #
    def __init__(self,sobolBase=np.empty, sobolTot=np.empty, fA=np.empty, fB=np.empty, fD=np.empty, fAB=np.empty, \
                 sampD=np.empty,sigma2=np.empty, muStar=np.empty):
        self.sobolBase=sobolBase
        self.sobolTot=sobolTot
        self.fA=fA
        self.fB=fB
        self.fD=fD
        self.fAB=fAB
        self.sampD=sampD
        self.muStar=muStar
        self.sigma2=sigma2
    pass



##--------------------------------------GSA-----------------------------------------------------
def RunGSA(model, options):
    #GSA implements the following local sensitivity analysis methods on "model" object
        # 1) Gets sampling distribution (used only for internal calculations)
        # 2) Calculates Sobol Indices
        # 3) Performs Morris Screenings (not yet implemented)
        # 4) Produces histogram plots for QOI values (not yet implemented)
    # Required Inputs: Object of class "model" and object of class "options"
    # Outputs: Object of class gsa with fisher and sobol elements

    #Get Parameter Distributions
    model=GetSampDist(model, options.gsa)
    test_results = results()
    #Morris Screening
    if options.gsa.runMorris:
        muStar, sigma2 = CaclulateMorris(model, options)
        test_results.muStar=muStar
        test_results.sigma2=sigma2

    #Sobol Analysis
    if options.gsa.runSobol:
        #Make Distribution Samples and Calculate model results
        [fA, fB, fAB, fD, sampD] = GetSamples(model, options.gsa)
        #Calculate Sobol Indices
        [sobolBase, sobolTot]=CalculateSobol(fA, fB, fAB, fD)
        test_results.fD=fD
        test_results.fA=fA
        test_results.fB=fB
        test_results.fAB=fAB
        test_results.sampD=sampD
        test_results.sobolBase=sobolBase
        test_results.sobolTot=sobolTot
        
    return test_results


###----------------------------------------------------------------------------------------------
###-------------------------------------Support Functions----------------------------------------
###----------------------------------------------------------------------------------------------


def GetSamples(model,gsaOptions):
    nSampSobol = gsaOptions.nSampSobol
    # Make 2 POI sample matrices with nSampSobol samples each
    if model.dist.lower()=='uniform' or model.dist.lower()=='saltellinormal':
        (sampA, sampB)=model.sampDist(nSampSobol);                                     #Get both A and B samples so no repeated values
    else:
        sampA = model.sampDist(nSampSobol)
        sampB = model.sampDist(nSampSobol)
    # Calculate matrices of QOI values for each POI sample matrix
    fA = model.evalFcn(sampA).reshape([nSampSobol, model.nQOIs])  # nSampSobol x nQOI out matrix from A
    fB = model.evalFcn(sampB).reshape([nSampSobol, model.nQOIs])  # nSampSobol x nQOI out matrix from B
    # Stack the output matrices into a single matrix
    fD = np.concatenate((fA.copy(), fB.copy()), axis=0)

    # Initialize combined QOI sample matrices
    if model.nQOIs == 1:
        fAB = np.empty([nSampSobol, model.nPOIs])
    else:
        fAB = np.empty([nSampSobol, model.nPOIs, model.nQOIs])
    for iParams in range(0, model.nPOIs):
        # Define sampC to be A with the ith parameter in B
        sampAB = sampA.copy()
        sampAB[:, iParams] = sampB[:, iParams].copy()
        if model.nQOIs == 1:
            fAB[:, iParams] = model.evalFcn(sampAB)
        else:
            fAB[:, iParams, :] = model.evalFcn(sampAB)  # nSampSobol x nPOI x nQOI tensor
        del sampAB
    return fA, fB, fAB, fD, np.concatenate((sampA.copy(), sampB.copy()), axis=0)

def CalculateSobol(fA, fB, fAB, fD):
    #Calculates calculates sobol indices using satelli approximation method
    #Inputs: model object (with evalFcn, sampDist, and nParams)
    #        sobolOptions object
    #Determing number of samples, QOIs, and POIs based on inputs
    nSampSobol=fAB.shape[0]
    if fAB.ndim==1:
        nQOIs=1
        nPOIs=1
    elif fAB.ndim==2:
        nQOIs=1
        nPOIs=fAB.shape[1]
    elif fAB.ndim==3:
        nPOIs=fAB.shape[1]
        nQOIs=fAB.shape[2]
    else:
        raise(Exception('fAB has greater than 3 dimensions, make sure fAB is the squeezed form of nSampSobol x nPOI x nQOI'))
    #QOI variance
    fDvar=np.var(fD, axis=0)

    sobolBase=np.empty((nQOIs, nPOIs))
    sobolTot=np.empty((nQOIs, nPOIs))
    if nQOIs==1:
        #Calculate 1st order parameter effects
        sobolBase=np.mean(fB*(fAB-fA), axis=0)/(fDvar)

        #Caclulate 2nd order parameter effects
        sobolTot=np.mean((fA-fAB)**2, axis=0)/(2*fDvar)

    else:
        for iQOI in range(0,nQOIs):
            #Calculate 1st order parameter effects
            sobolBase[iQOI,:]=np.mean(fB[:,[iQOI]]*(fAB[:,:,iQOI]-fA[:,[iQOI]]),axis=0)/fDvar[iQOI]
            #Caclulate 2nd order parameter effects
            sobolTot[iQOI,:]= np.mean((fA[:,[iQOI]]-fAB[:,:,iQOI])**2,axis=0)/(2*fDvar[iQOI])


    return sobolBase, sobolTot

##-------------------------------------GetMorris-------------------------------------------------------
def CaclulateMorris(model,options):
    #Define delta
    delta=(options.gsa.lMorris+1)/(2*options.gsa.lMorris)
    #Get Parameter Samples- use parameter distribution
    paramsSamp=model.sampDist(options.gsa.nSampMorris)[0]
    #Calulate derivative indices
    d= np.empty((options.gsa.nSampMorris, model.nPOIs, model.nQOIs)) #nQOIs x nPOIs x nSamples
    #Define constant sampling matrices
    J=np.ones((model.nPOIs+1,model.nPOIs))
    B = (np.tril(np.ones(J.shape), -1))
    for iSamp in range(0,options.gsa.nSampMorris):
        #Define Random Sampling matrices
        D=np.diag(np.random.choice(np.array([1,-1]), size=(model.nPOIs,)))
        P=np.identity(model.nPOIs)
        #np.random.shuffle(P)
        jTheta=paramsSamp[iSamp,]*J
        #CalculateMorris Sample matrix
        Bj=np.matmul(jTheta+delta/2*(np.matmul((2*B-J),D)+J),P)
        fBj=model.evalFcn(Bj)
        for k in np.arange(0,model.nPOIs):
            i=np.nonzero(Bj[k+1,:]-Bj[k,:])[0][0]
            print(np.nonzero(Bj[k+1,:]-Bj[k,:]))
            if Bj[k+1,i]-Bj[k,i]>0:
                if model.nQOIs==1:
                    d[iSamp,i]=(fBj[k+1]-fBj[k])/delta
                else:
                    d[iSamp,i,:]=(fBj[k+1]-fBj[k])/delta
            elif Bj[k+1,i]-Bj[k,i]<0:
                if model.nQOIs==1:
                    d[iSamp,i]=(fBj[k]-fBj[k+1])/delta
                else:
                    d[iSamp,i,:]=(fBj[k,:]-fBj[k+1,:])/delta
            else:
                raise(Exception('0 difference identified in Morris'))
    #Compute Indices- all outputs are nQOIs x nPOIs
    muStar=np.mean(np.abs(d),axis=0)
    sigma2=np.var(d, axis=0)

    return muStar, sigma2


##--------------------------------------GetSampDist----------------------------------------------------
def GetSampDist(model, gsaOptions):
    # Determine Sample Function- Currently only 1 distribution type can be defined for all parameters
    if model.dist.lower() == 'normal':  # Normal Distribution
        sampDist = lambda nSampSobol: np.random.randn(nSampSobol,model.nPOIs)*np.sqrt(model.distParms[[1], :]) + model.distParms[[0], :]
    elif model.dist.lower() == 'saltellinormal':
        sampDist = lambda nSampSobol: SaltelliNormal(nSampSobol, model.distParms)
    elif model.dist.lower() == 'uniform':  # uniform distribution
        # doubleParms=np.concatenate(model.distParms, model.distParms, axis=1)
        sampDist = lambda nSampSobol: SaltelliSample(nSampSobol,model.distParms)
    elif model.dist.lower() == 'exponential': # exponential distribution
        sampDist = lambda nSampSobol: np.random.exponential(model.distParms,size=(nSampSobol,model.nPOIs))
    elif model.dist.lower() == 'beta': # beta distribution
        sampDist = lambda nSampSobol:np.random.beta(model.distParms[[0],:], model.distParms[[1],:],\
                                               size=(nSampSobol,model.nPOIs))
    elif model.dist.lower() == 'InverseCDF': #Arbitrary distribution given by inverse cdf
        sampDist = lambda nSampSobol: gsaOptions.fInverseCDF(np.random.rand(nSampSobol,model.nPOIs))
    else:
        raise Exception("Invalid value for options.gsa.dist. Supported distributions are normal, uniform, exponential, beta, \
        and InverseCDF")  # Raise Exception if invalide distribution is entered
    model.sampDist=sampDist
    return model



def SaltelliSample(nSampSobol,distParams):
    nPOIs=distParams.shape[1]
    #baseSample=sobol.sample(dimension=nPOIs*2, n_points=nSampSobol, skip=1099)
    baseSample=SobolSample(nSampSobol,nPOIs*2)
    baseA=baseSample[:,:nPOIs]
    baseB=baseSample[:,nPOIs:2*nPOIs]
    sampA=distParams[[0],:]+(distParams[[1],:]-distParams[[0],:])*baseA
    sampB=distParams[[0],:]+(distParams[[1],:]-distParams[[0],:])*baseB
    return (sampA, sampB)

def SaltelliNormal(nSampSobol, distParms):
    nPOIs=distParms.shape[1]
    #baseSample=sobol.sample(dimension=nPOIs*2, n_points=nSampSobol, skip=1099)
    baseSample=SobolSample(nSampSobol,nPOIs*2)
    baseA=baseSample[:,:nPOIs]
    baseB=baseSample[:,nPOIs:2*nPOIs]
    transformA=sct.norm.ppf(baseA)
    transformB=sct.norm.ppf(baseB)
    sampA=transformA*np.sqrt(distParms[[1], :]) + distParms[[0], :]
    sampB=transformB*np.sqrt(distParms[[1], :]) + distParms[[0], :]
    return (sampA, sampB)
