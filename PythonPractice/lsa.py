# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 14:03:22 2022

@author: USER
"""
#3rd party Modules
import numpy as np
import sys
#import warnings
#import matplotlib.pyplot as plt
#import scipy.integrate as integrate
#from tabulate import tabulate                       #Used for printing tables to terminal
#import sobol                                        #Used for generating sobol sequences
#import SALib.sample as sample
#import scipy.stats as sct

class options:
    def __init__(self,run=True, runActiveSubspace=True, xDelta=10**(-12),\
                 method='complex', scale='y', subspaceRelTol=.001):
        self.run=run                              #Whether to run lsa (True or False)
        self.xDelta=xDelta                        #Input perturbation for calculating jacobian
        self.scale=scale                          #scale can be y, n, or both for outputing scaled, unscaled, or both
        self.method=method                        #method used for approximating derivatives
        if self.run == False:
            self.runActiveSubspace = False
        else:
            self.runActiveSubspace=runActiveSubspace
        self.subspaceRelTol=subspaceRelTol
        if not self.scale.lower() in ('y','n','both'):
            raise Exception('Error! Unrecgonized scaling output, please enter y, n, or both')
        if not self.method.lower() in ('complex','finite'):
            raise Exception('Error! unrecognized derivative approx method. Use complex or finite')
        if self.xDelta<0 or not isinstance(self.xDelta,float):
            raise Exception('Error! Non-compatibale xDelta, please use a positive floating point number')
        if self.subspaceRelTol<0 or self.subspaceRelTol>1 or not isinstance(self.xDelta,float):
            raise Exception('Error! Non-compatibale xDelta, please use a positive floating point number less than 1')
    pass


##--------------------------------------LSA-----------------------------------------------------
# Local Sensitivity Analysis main
class results:
    def __init__(self,jacobian=np.empty, rsi=np.empty, fisher=np.empty, reducedModel=np.empty, activeSpace="", inactiveSpace=""):
        self.jac=jacobian
        self.rsi=rsi
        self.fisher=fisher
        self.reducedModel=reducedModel
        self.activeSpace=activeSpace
        self.inactiveSpace=inactiveSpace
    pass


def RunLSA(model, options):
    # LSA implements the following local sensitivity analysis methods on system specified by "model" object
        # 1) Jacobian
        # 2) Scaled Jacobian for Relative Sensitivity Index (RSI)
        # 3) Fisher Information matrix
    # Required Inputs: object of class "model" and object of class "options"
    # Outputs: Object of class lsa with Jacobian, RSI, and Fisher information matrix

    # Calculate Jacobian
    jacRaw=GetJacobian(model.evalFcn, model.basePOIs, options.lsa, scale=False, yBase=model.baseQOIs)
    # Calculate relative sensitivity index (RSI)
    jacRSI=GetJacobian(model.evalFcn, model.basePOIs, options.lsa, scale=True, yBase=model.baseQOIs)
    # Calculate Fisher Information Matrix from jacobian
    fisherMat=np.dot(np.transpose(jacRaw), jacRaw)

    #Active Subspace Analysis
    if options.lsa.runActiveSubspace:
        reducedModel, activeSpace, inactiveSpace = GetActiveSubspace(model, options.lsa)
        #Collect Outputs and return as an lsa object
        return results(jacobian=jacRaw, rsi=jacRSI, fisher=fisherMat,\
                          reducedModel=reducedModel, activeSpace=activeSpace,\
                          inactiveSpace=inactiveSpace)
    else:
        return results(jacobian=jacRaw, rsi=jacRSI, fisher=fisherMat)
    
###----------------------------------------------------------------------------------------------
###-------------------------------------Support Functions----------------------------------------
###----------------------------------------------------------------------------------------------

  
    
  
##--------------------------------------GetJacobian-----------------------------------------------------
def GetJacobian(evalFcn, xBase, lsaOptions, **kwargs):
    # GetJacobian calculates the Jacobian for n QOIs and p POIs
    # Required Inputs: object of class "model" (.cov element not required)
    #                  object of class "lsaOptions"
    # Optional Inputs: alternate POI position to estimate Jacobian at (*arg) or complex step size (h)
    if 'scale' in kwargs:                                                   # Determine whether to scale derivatives
                                                                            #   (for use in relative sensitivity indices)
        scale = kwargs["scale"]
        if not isinstance(scale, bool):                                     # Check scale value is boolean
            raise Exception("Non-boolean value provided for 'scale' ")      # Stop compiling if not
    else:
        scale = False                                                       # Function defaults to no scaling
    if 'yBase' in kwargs:
        yBase = kwargs["yBase"]
    else:
        yBase = evalFcn(xBase)

    #Load options parameters for increased readibility
    xDelta=lsaOptions.xDelta

    #Initialize base QOI value, the number of POIs, and number of QOIs
    nPOIs = np.size(xBase)
    nQOIs = np.size(yBase)

    jac = np.empty(shape=(nQOIs, nPOIs), dtype=float)                       # Define Empty Jacobian Matrix

    for iPOI in range(0, nPOIs):                                            # Loop through POIs
        # Isolate Parameters
        if lsaOptions.method.lower()== 'complex':
            xPert = xBase + np.zeros(shape=xBase.shape)*1j                  # Initialize Complex Perturbed input value
            xPert[iPOI] += xDelta * 1j                                      # Add complex Step in input
        elif lsaOptions.method.lower() == 'finite':
            xPert=xBase*(1+xDelta)
        yPert = evalFcn(xPert)                                        # Calculate perturbed output
        for jQOI in range(0, nQOIs):                                        # Loop through QOIs
            if lsaOptions.method.lower()== 'complex':
                jac[jQOI, iPOI] = np.imag(yPert[jQOI] / xDelta)                 # Estimate Derivative w/ 2nd order complex
            elif lsaOptions.method.lower() == 'finite':
                jac[jQOI, iPOI] = (yPert[jQOI]-yBase[jQOI]) / xDelta
            #Only Scale Jacobian if 'scale' value is passed True in function call
            if scale:
                jac[jQOI, iPOI] *= xBase[iPOI] * np.sign(yBase[jQOI]) / (sys.float_info.epsilon + yBase[jQOI])
                                                                            # Scale jacobian for relative sensitivity
        del xPert, yPert, iPOI, jQOI                                        # Clear intermediate variables
    return jac                                                              # Return Jacobian




##--------------------------------------------Parameter dimension reduction------------------------------------------------------
def GetActiveSubspace(model,lsaOptions):
    eliminate=True
    inactiveIndex=np.zeros(model.nPOIs)
    #Calculate Jacobian
    jac=GetJacobian(model.evalFcn, model.basePOIs, lsaOptions, scale=False, yBase=model.baseQOIs)
    while eliminate:
        #Caclulate Fisher
        fisherMat=np.dot(np.transpose(jac), jac)
        #Perform Eigendecomp
        eigenValues, eigenVectors =np.linalg.eig(fisherMat)
        #Eliminate dimension/ terminate
        if np.min(eigenValues) < lsaOptions.subspaceRelTol * np.max(eigenValues):
            #Get inactive parameter
            inactiveParamReducedIndex=np.argmax(np.absolute(eigenVectors[:, np.argmin(np.absolute(eigenValues))]))
            inactiveParam=inactiveParamReducedIndex+np.sum(inactiveIndex[0:(inactiveParamReducedIndex+1)]).astype(int)
                #This indexing may seem odd but its because we're keeping the full model parameter numbering while trying
                # to index within the reduced model so we have to add to the index the previously removed params
            #Record inactive param in inactive space
            inactiveIndex[inactiveParam]=1
            #Remove inactive elements of jacobian
            jac=np.delete(jac,inactiveParamReducedIndex,1)
        else:
            #Terminate Active Subspace if singular values within tolerance
            eliminate=False
    #Define active and inactive spaces
    activeSpace=model.POInames[inactiveIndex == False]
    inactiveSpace=model.POInames[inactiveIndex == True]
    #Create Reduced model
    reducedModel=model.copy()
    # reducedModel.basePOIs=reducedModel.basePOIs[inactiveIndex == False]
    # reducedModel.POInames=reducedModel.POInames[inactiveIndex == False]
    # reducedModel.evalFcn = lambda reducedPOIs: model.evalFcn(
    #     np.array([x for x, y in zip(reducedPOIs,model.basePOIs) if inactiveIndex== True]))
    # #reducedModel.evalFcn=lambda reducedPOIs: model.evalFcn(np.where(inactiveIndex==False, reducedPOIs, model.basePOIs))
    # reducedModel.baseQOIs=reducedModel.evalFcn(reducedModel.basePOIs)
    return reducedModel, activeSpace, inactiveSpace

def ModelReduction(reducedModel,inactiveParam,model):
    #Record Index of reduced param
    inactiveIndex=np.where(reducedModel.POInames==inactiveParam)[0]
    #confirm exactly parameter matches
    if len(inactiveIndex)!=1:
        raise Exception("More than one or no POIs were found matching that name.")
    #Remove relevant data elements
    reducedModel.basePOIs=np.delete(reducedModel.basePOIs, inactiveIndex)
    reducedModel.POInames=np.delete(reducedModel.POInames, inactiveIndex)
    reducedModel.evalFcn=lambda reducedPOIs: model.evalFcn(np.where(inactiveIndex==True,reducedPOIs,model.basePOIs))
    print('made evalFcn')
    print(reducedModel.evalFcn(reducedModel.basePOIs))
    return reducedModel

def GetReducedPOIs(reducedPOIs,droppedIndices,model):
    fullPOIs=model.basePOIs
    reducedCounter=0
    print(droppedIndices)
    for i in np.arange(0,model.nPOIs):
        print(i)
        if droppedIndices==i:
            fullPOIs[i]=reducedPOIs[reducedCounter]
            reducedCounter=reducedCounter+1
    print(fullPOIs)
    return fullPOIs