#UQtoolbox
#Module for calculating local sensitivity indices, parameter correlation, and Sobol indices for an arbitrary model
#Authors: Harley Hanes, NCSU, hhanes@ncsu.edu
#Required Modules: numpy, seaborne
#Functions: LSA->GetJacobian
#           GSA->ParamSample, PlotGSA, GetSobol

import numpy as np
import sys
#import seaborne as seaborne
###----------------------------------------------------------------------------------------------
###-------------------------------------Class Definitions----------------------------------------
###----------------------------------------------------------------------------------------------

##--------------------------------------uqOptions--------------------------------------------------
#Define class "uqOptions", this will be the class used to collect algorithm options for functions
#   -Subclasses: jacOptions, plotOptions, sampOptions
#--------------------------------------jacOptions------------------------------------------------
class jacOptions:
    def __init__(self,xDelta=10^(-6), method='complex', scale='y'):
        self.xDelta=xDelta                        #Input perturbation for calculating jacobian
        self.scale=scale                          #scale can be y, n, or both for outputing scaled, unscaled, or both
        self.method=method                        #method used for caclulating Jacobian NOT CURRENTLY IMPLEMENTED
    pass
#--------------------------------------sampOptions------------------------------------------------
class sampOptions:
    def __init__(self, nSamp=1000, dist='norm', var=1):
        self.nSamp = nSamp                      #Number of samples to be generated for GSA
        self.dist = dist                        #String identifying sampling distribution for parameters
                                                #       Supported distributions: normal
        pass
#--------------------------------------plotOptions------------------------------------------------
class plotOptions:
    def __init__(self,):
        pass
#--------------------------------------uqOptions------------------------------------------------
#   Class holding the above options subclasses
class uqOptions:
    def __init__(self,jac=jacOptions(),plot=plotOptions(),samp=sampOptions()):
        self.jac=jac
        self.plot=plot
        self.samp=samp
    pass

##-------------------------------------model------------------------------------------------------
#Define class "model", this will be the class used to collect input information for all functions
class model:
    #Model sets should be initialized with base parameter settings, covariance Matrix, and eval function that
    #   takes in a vector of POIs and outputs a vector of QOIs
    def __init__(self,basePOIs=np.empty, covMat=np.empty, evalFcn=np.empty):
        self.basePOIs=basePOIs
        self.cov=covMat
        self.evalFcn=evalFcn
        self.baseQOIs=evalFcn(basePOIs)
        self.sampDist='null'
        self.nPOIs=len(self.basePOIs)
        self.nQOIs=len(self.baseQOIs)
    pass

##------------------------------------results-----------------------------------------------------
#-------------------------------------lsaResults--------------------------------------------------
# Define class "lsa", this will be the used to collect relative sensitivity analysis outputs
class lsaResults:
    #
    def __init__(self,jacobian=np.empty, rsi=np.empty, fisher=np.empty):
        self.jac=jacobian
        self.rsi=rsi
        self.fisher=fisher
    pass
#-------------------------------------gsaResults--------------------------------------------------
# Define class "gsaResults" which holds sobol analysis results
class gsaResults:
    #
    def __init__(self,sobolBase=np.empty, sobolTot=np.empty):
        self.sobolBase=sobolBase
        self.sobolTot=sobolTot
    pass
##------------------------------------results-----------------------------------------------------
# Define class "results" which holds a gsaResults object and lsaResults object

class results:
    def __init__(self,lsa=lsaResults(), gsa=gsaResults()):
        self.lsa=lsa
        self.gsa=gsa


###----------------------------------------------------------------------------------------------
###-------------------------------------Main Functions----------------------------------------
###----------------------------------------------------------------------------------------------
#   The following functions are the primary functions for running the package. RunUQ runs both local sensitivity
#   analysis and global sensitivity analysis while printing to command window summary statistics. However, local
#   sensitivity analysis and global sensitivity analysis can be run independently with LSA and GSA respectively

##--------------------------------------RunUQ-----------------------------------------------------
def RunUQ(model, options):
    #RunUQ is the primary call function for UQtoolbox and runs both the local sensitivity analysis and global sensitivity
    #   analysis while printing summary statistics to the command window.
    #Inputs: model object, options object
    #Outpts: results object, a list of summary results is printed to command window

    #Run Local Sensitivity Analysis
    results.lsa = LSA(model, options)

    #Run Global Sensitivity Analysis
    results.gsa = GSA(model, options)

    #Print Results
    print("Base Parameters: " + str(model.basePOIs))
    print("Base values: " + str(model.baseQOIs))
    print("Jacobian: " + str(results.lsa.jac))
    print("Relative Sensitivities: " + str(results.lsa.rsi))
    print("Fisher Matrix: " + str(results.lsa.fisher))
    print("1st Order Sobol Indices: " + str(results.gsa.sobolBase))
    print("2st Order Sobol Indices: " + str(results.gsa.sobolTot))
    return results

# Top order functions- These functions are the main functions for each component of our analysis they include,

##--------------------------------------LSA-----------------------------------------------------
# Local Sensitivity Analysis main
def LSA(model, options):
    # LSA implements the following local sensitivity analysis methods on system specified by "model" object
        # 1) Jacobian
        # 2) Scaled Jacobian for Relative Sensitivity Index (RSI)
        # 3) Fisher Information matrix
    # Required Inputs: object of class "model" and object of class "options"
    # Outputs: Object of class lsa with Jacobian, RSI, and Fisher information matrix

    # Calculate Jacobian
    jacRaw=GetJacobian(model, options.jac, scale=False)
    # Calculate relative sensitivity index (RSI)
    jacRSI=GetJacobian(model, options.jac, scale=True)
    # Calculate Fisher Information Matrix from jacobian
    fisherMat=np.dot(np.transpose(jacRaw), jacRaw)

    #Collect Outputs and return as an lsa object
    return lsaResults(jacobian=jacRaw, rsi=jacRSI, fisher=fisherMat)


##--------------------------------------GSA-----------------------------------------------------
def GSA(model, options):
    #GSA implements the following local sensitivity analysis methods on "model" object
        # 1) Gets sampling distribution (used only for internal calculations)
        # 2) Calculates Sobol Indices
        # 3) Performs Morris Screenings (not yet implemented)
        # 4) Produces histogram plots for QOI values (not yet implemented)
    # Required Inputs: Object of class "model" and object of class "options"
    # Outputs: Object of class gsa with fisher and sobol elements

    #Get Parameter Distributions
    model=GetSampDist(model, options.samp)
    #Plot Correlations and Distributions
    #PlotGSA(evalMat, sampleMat)
    #Calculate Sobol Indices
    [sobolBase, sobolTot]=GetSobol(model, options.samp)
    return gsaResults(sobolBase=sobolBase, sobolTot=sobolTot)

###----------------------------------------------------------------------------------------------
###-------------------------------------Support Functions----------------------------------------
###----------------------------------------------------------------------------------------------

##--------------------------------------GetJacobian-----------------------------------------------------
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

    #Initialize base QOI value, the number of POIs, and number of QOIs
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


##--------------------------------------GetSobol----------------------------------------------------
# GSA Component Functions
def GetSobol(model,sampOptions):
    #GetSobol calculates sobol indices using satelli approximation method
    #Inputs: model object (with evalFcn, sampDist, and nParams)
    #        sobolOptions object
    #Load options and data
    nSamp=sampOptions.nSamp
    sampDist=model.sampDist
    evalFcn=model.evalFcn
    #Make 3 POI sample matrices with nSamp samples each and concatenate the first two
    sampA=sampDist(nSamp)
    sampB=sampDist(nSamp)
    sampD=sampDist(nSamp)     #THIS CURRENTLY UNUSED, CHECK IF BUG
    sampC=np.concatenate((sampA, sampB))
    #Calculate matrices of QOI values for each POI sample matrix
    fA=evalFcn(sampA)                                                   #nSamp x nQOI out matrix from A
    fB=evalFcn(sampB)                                                   #nSamp x nQOI out matrix from B
    fC=np.concatenate((fA, fB))                                         #CONFIRM THIS VALID INSTEAD OF RECALCULATING FROM sampC
    #Initialize combined QOI sample matrices
    fAb=np.empty([nSamp, model.nQOIs, model.nPOIs])
    fBa=fAb.copy()
    for iParams in range(0,model.nPOIs):
        #Define sampAb to be A with the ith parameter in B
        sampAb=sampA
        sampAb[:, iParams]=sampB[:, iParams]
        #Define sampBa to be B with the ith parameter in A
        sampBa=sampB
        sampBa[:, iParams]=sampA[:, iParams]                        #nSamp x nPOI matrix
        #Calculate QOI values for each combined sample matrix
        print(sampAb)
        print(evalFcn(sampAb))
        fAb[:,:,iParams]=evalFcn(sampAb)                              #nSamp x nQOI x nPOI tensor
        fBa[:,:,iParams]=evalFcn(sampBa)                              #nSamp x nQOI x nPOI tensor
    #Expected QOI value
    fCexpected=np.mean(evalFcn(sampC),axis=0)                       #1 x nQOI vector of average of evalFcn over C param
                                                                    #    sample at n QOIs
    #QOI value variance
    sobolDen=1/(2*nSamp)*np.sum(fC**2,axis=0)-fCexpected**2           #Check correct syntax on ^2


    # Check correct formula, double divide looks wrong
    # Should be able to remove for loops, just check how products of tensors work
    #Calculate 1st order parameter effects
    sobolBase=np.empty((model.nPOIs,model.nQOIs))
    for iPOI in range(0,model.nPOIs):
        sobolBase[iPOI,:]=1/(nSamp*np.sum(fA*fBa[:,:,iPOI]-fA*fB,axis=0))/sobolDen
                #Check correct formula, double divide looks wrong

    #Caclulate 2nd order parameter effects
    sobolTot=np.empty((model.nPOIs,model.nQOIs))
    for iPOI in range(0,model.nPOIs):
        sobolTot[iPOI,:]=1/(2*nSamp)*np.sum(fA-fAb[:,:,iPOI],axis=0)/sobolDen
    return sobolBase, sobolTot

##--------------------------------------GetSampDist----------------------------------------------------
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
