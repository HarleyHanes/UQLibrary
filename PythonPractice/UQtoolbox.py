#UQtoolbox
#Module for calculating local sensitivity indices, parameter correlation, and Sobol indices for an arbitrary model
#Authors: Harley Hanes, NCSU, hhanes@ncsu.edu
#Required Modules: numpy, seaborne
#Functions: LSA->GetJacobian
#           GSA->ParamSample, PlotGSA, GetSobol

import numpy as np
import sys
#import seaborne as seaborne


#Define class "options", this will be the class used to collect algorithm options for functions
#   -Subclasses: jacOptions, plotOptions, sampOptions
class jacOptions:
    def __init__(self,xDelta=10^(-6), method='complex', scale='y'):
        self.xDelta=xDelta                        #Input perturbation for calculating jacobian
        self.scale=scale                          #scale can be y, n, or both for outputing scaled, unscaled, or both
        self.method=method                        #method used for caclulating Jacobian NOT CURRENTLY IMPLEMENTED
    pass

class sampOptions:
    def __init__(self, nSamp=1000, dist='norm', var=1):
        self.nSamp = nSamp                      #Number of samples to be generated for GSA
        self.dist = dist                        #String identifying sampling distribution for parameters
                                                #       Supported distributions: normal
        pass
class plotOptions:
    def __init__(self,):
        pass

class uqOptions:
    def __init__(self,jac=jacOptions(),plot=plotOptions(),samp=sampOptions()):
        self.jac=jac
        self.plot=plot
        self.samp=samp
    pass

#Define class "model", this will be the class used to collect input information for all functions
class model:
    #Model sets should be initialized with base parameter settings, covariance Matrix, and eval function that
    #   takes in a vector of POIs and outputs a vector of QOIs
    def __init__(self,baseParams=np.empty, covMat=np.empty, evalFcn=np.empty):
        self.basePOIs=baseParams
        self.cov=covMat
        self.evalFcn=evalFcn
        self.baseQOIs=evalFcn(baseParams)
        self.sampDist='null'
        self.nPOIs=len(self.basePOIs)
        self.nQOIs=len(self.baseQOIs)
    pass

# Define class "lsa", this will be the used to collect relative sensitivity analysis outputs
class lsa:
    #
    def __init__(self,jacobian=np.empty, rsi=np.empty, fisherMat=np.empty):
        self.jac=jacobian
        self.rsi=rsi
        self.fisher=fisherMat
    pass

# Local Sensitivity Analysis Functions
def LSA(model, options):
    # LSA implements the following local sensitivity analysis methods on system specific by "model" object
        # 1) Jacobian
        # 2) Scaled Jacobian for Relative Sensitivity Index (RSI)
        # 3) Fisher Information matrix
    # Required Inputs: object of class "model" and object of class options
    # Outputs: Object of class lsa with Jacobian, RSI, and Fisher information matrix
    # Calculate Jacobian
    jacRaw=GetJacobian(model, options.jac, scale=False)
    # Calculate RSI
    jacRSI=GetJacobian(model, options.jac, scale=True)
    # Calculate Fisher Information Matrix
    fisherMat=np.dot(np.transpose(jacRaw), jacRaw)

    #Collect Outputs and return as an lsa object
    return lsa(jacRaw, jacRSI, fisherMat)



def GetJacobian(model, jacOptions, **kwargs):
    # GetJacobian calculates the Jacobian for n QOIs and p POIs
    # Required Inputs: object of class "model" (.cov element not required)
    #                  object of class "jacOptions"
    # Optional Inputs: alternate POI position to estimate Jacobian at (*arg) or complex step size (h)
    if 'scale' in kwargs:                                                   # Determine whether to scale derivatives
                                                                            #   (for use in relative sensitivity indices)
        scale = kwargs["scale"]
        if not isinstance(scale, bool):                                     # Check scale value is boolean
            raise Exception("Non-boolean value provided for 'scale' ")      # Stop compiling if not
    else:
        scale = False                                                       # Function defaults to no scaling

    #Load options parameters for increased readibility
    xBase=model.basePOIs
    xDelta=jacOptions.xDelta

    #Initializae other parameters
    yBase=model.evalFcn(xBase)                                              # Get base QOI values
    nPOIs = model.nPOIs                                                     # Get number of parameters (nPOIs)
    nQOIs = model.nQOIs                                                     # Get number of outputs (nQOIs)

    jac = np.empty(shape=(nQOIs, nPOIs), dtype=float)                       # Define Empty Jacobian Matrix

    for iPOI in range(0, nPOIs):                                            # Loop through POIs
        # Isolate Parameters
        xPert = xBase + np.zeros(shape=xBase.shape)*1j                      # Initialize Complex Perturbed input value
        xPert[iPOI] += xDelta * 1j                                          # Add complex Step in input
        yPert = model.evalFcn(xPert)                                        # Calculate perturbed output
        for jQOI in range(0, nQOIs):                                        # Loop through QOIs
            jac[jQOI, iPOI] = np.imag(yPert[jQOI] / xDelta)                      # Estimate Derivative w/ 2nd order complex
            #Only Scale Jacobian if 'scale' value is passed True in function call
            if scale:
                jac[jQOI, iPOI] *= xBase[iPOI] * np.sign(yBase[jQOI]) / (sys.float_info.epsilon + yBase[jQOI]**2)
                                                                            # Scale jacobian for relative sensitivity
        del xPert, yPert, iPOI, jQOI                                        # Clear intermediate variables
    return jac                                                              # Return Jacobian
#
# #Global Sensitivity Analysis Functions
def GSA(model, options):
    #Get Parameter Distributions
    model=GetSampDist(model, options.samp)
    #Plot Correlations and Distributions
    #PlotGSA(evalMat, sampleMat)
    #Calculate Sobol Indices
    gsaResults.sobol=GetSobol(model, options.samp)
    return gsaResults

def GetSobol(model,sampOptions):
    #GetSobol calculates sobol indices using satelli approximation
    #Inputs: model object (with evalFcn, sampDist, and nParams)
    #        sobolOptions obejct
    #Load options and data
    nSamp=sampOptions.nSamp
    sampDist=model.sampDist
    evalFcn=model.evalFcn
    #Make Parameter Sample Matrices
    sampA=sampDist(nSamp)
    sampB=sampDist(nSamp)
    sampC=np.concatenate((sampA, sampB))
    sampD=sampDist(nSamp)
    #Calculate QOI vectors
    fA=evalFcn(sampA)
    fB=evalFcn(sampB)
    fC=evalFcn(sampC)
    #Initialize combined QOI sample matrices
    fAB=np.empty([nSamp, model.nPOIs])
    fBA=fAB.copy()
    for iParams in range(0,nSamp):
        #Define sampAb to be A with the ith parameter in B
        sampAB=sampA
        sampAB[:, iParams]=sampB[:, iParams]
        #Define sampBa to be B with the ith parameter in A
        sampBA=sampB
        sampBA[:, iParams]=sampA[:, iParams]
        #Calculate QOI sample matrices
        print(evalFcn(sampAB))
        fAB[:,iParams]=evalFcn(sampAB)
        fBA[:,iParams]=evalFcn(sampBA)
    #Expected QOI value
    fCexpected=mean(evalFcn(sampC))
    #QOI value variance
    sobolDen=1/(2*nSamp)*np.sum(fC^2,axis=0)-fCexpected^2


    #Calculate 1st order parameter effects
    sobolResults.base=1/(nSamp*np.sum(fA*fBa-fA*fB,axis=0))/sobolDen

    #Caclulate 2nd order parameter effects
    sobolResults.total=1/(2*nSamp)*np.sum(fA-fAb,axis=0)/sobolDen
    return sobolResults

def GetSampDist(model, sampOptions):
    # Determine Sample Function
    if sampOptions.dist.lower() == 'norm':  # Normal Distribution
        sampDist = lambda nSamp: np.random.randn(nSamp,model.nPOIs)*np.sqrt(np.diag(model.cov))+model.basePOIs
    else:
        raise Exception("Invalid value for options.samp.dist")  # Raise Exception if invalide distribution is entered
    model.sampDist=sampDist
    return model
#
#
# def PlotGSA(evalMat, sampleMat):
#     #Intialize Variables
#
#     #Plot POI-POI correlation
#
#     #Plot POI-QOI correlation
#
#     #Plot QOI distribution
#
