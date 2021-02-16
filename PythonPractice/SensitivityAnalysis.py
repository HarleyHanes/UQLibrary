#SensitivityAnalysis
#Module for calculating local sensitivity indices, parameter correlation, and Sobol indices for an arbitrary model
#Authors: Harley Hanes, NCSU, hhanes@ncsu.edu
#Required Modules: numpy, seaborne
#Functions: LSA->GetJacobian
#           GSA->ParamSample, PlotGSA, GetSobol

import numpy as np
import sys
#import seaborne as seaborne

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
def LSA(model, **kwargs):
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
    jacRaw=GetJacobian(model, h=kwargs['h'], xBase=kwargs['xBase'], scale=False)
    # Calculate RSI
    jacRSI=GetJacobian(model, h=kwargs['h'], xBase=kwargs['xBase'], scale=False)
    # Calculate Fisher Information Matrix
    fisherMat=np.dot(np.transpose(jacRaw), jacRaw)

    #Collect Outputs and return as an lsa object
    return lsa(jacRaw, jacRSI, fisherMat)



def GetJacobian(model, **kwargs):
    # GetJacobian calculates the Jacobian for n QOIs and p POIs
    # Required Inputs: object of class "model" (.cov element not required)
    # Optional Inputs: alternate POI position to estimate Jacobian at (*arg) or complex step size (h)


    # Manage Function Inputs
    if 'h' in kwargs:                                                       # Assign complex step value
        h=kwargs["h"]
        if not isinstance(h, float):                                        # Check provided 'h' value is float

            raise Exception("Non-float value provided for 'h': h=" + str(h))            # Stop compiling if 'h' not float
    else:
        h = 1e-6                                                              # Give default h value of 1e-6
    if 'xBase' in kwargs:                                                   # Assign base parameter values
        xBase=kwargs["xBase"]
        if not isinstance(xBase, (np.ndarray, np.generic)):                 # Check base values entered as np array
            raise Exception("Non-np.array value provided for 'xBase' ")     # Stop compiling if not
    else:
        xBase=model.basePOIs                                                # Give default xBase of model's base POIs
    if 'scale' in kwargs:                                                   # Determine whether to scale derivatives
                                                                            #   (for use in relative sensitivity indices)
        scale = kwargs["scale"]
        if not isinstance(scale, bool):                                     # Check scale value is boolean
            raise Exception("Non-boolean value provided for 'scale' ")      # Stop compiling if not
    else:
        scale = False                                                       # Function defaults to no scaling

    yBase=model.evalFcn(xBase)                                              # Get base QOI values
    nPOIs = model.nPOIs                                                     # Get number of parameters (nPOIs)
    nQOIs = model.nQOIs                                                     # Get number of outputs (nQOIs)

    jac = np.empty(shape=(nQOIs, nPOIs), dtype=float)                       # Define Empty Jacobian Matrix

    for iPOI in range(0, nPOIs):                                            # Loop through POIs
        # Isolate Parameters
        xPert = xBase + np.zeros(shape=xBase.shape)*1j                      # Define Perturbed value
        xPert[iPOI] += h * 1j                                               # Complex Step
        yPert = model.evalFcn(xPert)                                        # Get Perturbed Output
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
# def GSA(params, evalFcn):
#     #Get Parameter Space Sample
#     [evalMat, sampleMat]=ParamSample
#     #Plot Correlations and Distributions
#     PlotGSA(evalMat, sampleMat)
#     #Calculate Sobol Indices
#     sobol=GetSobol(evalMat, sampleMat)
#     return evalMat, sampleMat, sobol
#
# def ParamSample:
#     #Intialize Variables
#     sampleSize=5;
#     nPOIs=len(params)
#     evalMat=np.empty(shape=(sampleSize, nPOIs), dtype=float)
#     #Sample Parameter Space
#     sampleMat=(np.random.randn(sampleSize, nPOIs)+1)*params     #Sample normal distribution with mu=epsilon=base param value
#     #Evaluate over Parameter Space Sample
#     for iSample in range(0,sampleSize):
#         evalMat[iSample,]=evalFcn(sampleMat[iSample,])
#
#     return evalMat, sampleMat
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
