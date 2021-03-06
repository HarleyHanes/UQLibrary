#SensitivityAnalysis
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
class uqOptions:
    def __init__(self,jac=jacOptions(),plot=plotOptions(),samp=sampOptions()):
        self.jac=jac
        self.plot=plot
        self.samp=samp
    pass

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
                                                #       Supported distributions:
        self.var=var                            # sample variance, DON'T LIKE THIS CURRENTLY- variance and mean are
                                                                                    # drawn from two seperate places
        pass

#

#Define class "model", this will be the class used to collect input information for all functions
class model:
    #Model sets should be initialized with base parameter settings, covariance Matrix, and eval function that
    #   takes in a vector of POIs and outputs a vector of QOIs
    def __init__(self,baseParams=np.empty, covMat=np.empty, evalFcn=np.empty):
        self.basePOIs=baseParams
        self.cov=covMat
        self.evalFcn=evalFcn
        self.baseQOIs=evalFcn(baseParams)
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
    # Required Inputs: object of class "model"
    # Optional Inputs: Analysis location, h  (Currently all settings for GetJacobian
    # Outputs: Object of class lsa with Jacobian, RSI, and Fisher information matrix
    # Next Steps: Allow GetJacobian to pass keywords from kwargs only when they're present

    # Check for unrecognized inputs
    checkDict=kwargs.copy()                                                       #Create a copy dictionary
    checkDict['h'] = 1                                                            #Add h and xBase if not already there
    checkDict['xBase'] = 1
    if checkDict.keys()>{"h","xBase"}:                                            #Check for extraneous inputs
        raise Exception('Unrecognized inputs to LSA detected, only provide  h= and xBase= as additional arguments')
                                                                                #Stop compiling if present
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
    xBase=jacOptions.xBase
    xDelta=jacOptions.xDelta

    #Initializae other parameters
    yBase=model.evalFcn(jacOptions.xBase)                                   # Get base QOI values
    nPOIs = model.nPOIs                                                     # Get number of parameters (nPOIs)
    nQOIs = model.nQOIs                                                     # Get number of outputs (nQOIs)

    jac = np.empty(shape=(nQOIs, nPOIs), dtype=float)                       # Define Empty Jacobian Matrix

    for iPOI in range(0, nPOIs):                                            # Loop through POIs
        # Isolate Parameters
        xPert = xBase + np.zeros(shape=xBase.shape)*1j                      # Initialize Complex Perturbed input value
        xPert[iPOI] += xDelta * 1j                                          # Add complex Step in input
        yPert = model.evalFcn(xPert)                                        # Calculate perturbed output
        for jQOI in range(0, nQOIs):                                        # Loop through QOIs
            jac[jQOI, iPOI] = np.imag(yPert[jQOI] / h)                      # Estimate Derivative w/ 2nd order complex
            #Only Scale Jacobian if 'scale' value is passed True in function call
            if scale:
                jac[jQOI, iPOI] *= xBase[iPOI] * np.sign(yBase[jQOI]) / (sys.float_info.epsilon + yBase[jQOI]**2)
                                                                            # Scale jacobian for relative sensitivity
        del xPert, yPert, iPOI, jQOI                                        # Clear intermediate variables
    return jac                                                              # Return Jacobian
#
# #Global Sensitivity Analysis Functions
def GSA(model, options):
    #Get Parameter Space Sample
    [evalMat, sampleMat]=ParamSample(sampOptions)
    #Plot Correlations and Distributions
    PlotGSA(evalMat, sampleMat)
    #Calculate Sobol Indices
    sobol=GetSobol(evalMat, sampleMat)
    return evalMat, sampleMat, sobol

def ParamSample(params,sampOptions):
    #Intialize Variables
    nPOIs=len(params)
    evalMat=np.empty(shape=(sampOptions.nSamp, nPOIs), dtype=float)

    #Sample Parameter Space
    if sampOptions.dist.lower()=='norm':                                             #Normal Distribution
        sampleMat=np.random.randn(sampleSize, nPOIs)*sqrt(sampOptions.var)+params    #Sample normal distribution with mu=params, sigma^2=sampOptions.var
    else:
        raise Exception("Invalid value for options.samp.dist")                       #Raise Exception if invalide distribution is entered

    #Evaluate over Parameter Space Sample
    for iSample in range(0,sampleSize):
        evalMat[iSample,]=evalFcn(sampleMat[iSample,])
    return evalMat, sampleMat
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
