#UQtoolbox
#Module for calculating local sensitivity indices, parameter correlation, and Sobol indices for an arbitrary model
#Authors: Harley Hanes, NCSU, hhanes@ncsu.edu
#Required Modules: numpy, seaborne
#Functions: LSA->GetJacobian
#           GSA->ParamSample, PlotGSA, GetSobol

import numpy as np
import sys
import warnings
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
        self.method=method                        #method used for approximating derivatives
    pass
#--------------------------------------sampOptions------------------------------------------------
class sampOptions:
    def __init__(self, nSamp=10000, dist='norm', distParms=(0,1)):
        self.nSamp = nSamp                      #Number of samples to be generated for GSA
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
    def __init__(self,basePOIs=np.empty(0), cov=np.empty(0), evalFcn=np.empty(0), dist='unif',distParms='null'):
        self.basePOIs=basePOIs
        if np.ndim(self.basePOIs)>1:                                    #Check to see if basePOIs is a vector
            self.basePOIs=np.squeeze(self.basePOIs)                     #Make a vector if an array with 1 dim greater than 1
            if np.ndim(self.basePOIs)!=1:                               #Issue an error if basePOIs is a matrix or tensor
                raise Exception("Error! More than one dimension of size 1 detected for model.basePOIs, model.basePOIs must be dimension 1")
            else:                                                       #Issue a warning if dimensions were squeezed out of base POIs
                warnings.warn("Warning: model.basePOIs was reduced a dimension 1 array. No entries were deleted.")
        self.nPOIs=len(self.basePOIs)
        self.evalFcn=evalFcn
        self.baseQOIs=evalFcn(basePOIs)
        self.nQOIs=len(self.baseQOIs)
        self.cov=cov
        if self.cov.size!=0 and np.shape(self.cov)!=(self.nPOIs,self.nPOIs):
            raise Exception("Error! model.cov is not an nPOI x nPOI array")
        self.dist = dist                        #String identifying sampling distribution for parameters
                                                #       Supported distributions: unif, normal, exponential, beta, inverseCDF
        if isinstance(distParms,str):
            if self.dist.lower()=='unif':
                self.distParms=[[.8],[1.2]]*np.ones((2,self.nPOIs))*self.basePOIs
            elif self.dist.lower()=='norm':
                if cov.size()==0:
                    self.distParms=[[1],[.2]]*np.ones((2,self.nPOIs))*self.basePOIs
                else:
                    self.distParms=[self.basePOIs, np.diag(self.cov,k=0)]

        else:
            self.distParms=distParms
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
    #Set seed for reporducibility
    np.random.seed(10)
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
    print("Total Sobol Indices: " + str(results.gsa.sobolTot))
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
    #Make 2 POI sample matrices with nSamp samples each and then combine the first two
    sampA=sampDist(nSamp)
    sampB=sampDist(nSamp)
    #sampC=sampDist(nSamp)
    sampD=np.concatenate((sampA, sampB),axis=0)
    #Calculate matrices of QOI values for each POI sample matrix
    fA=evalFcn(sampA).reshape([nSamp,model.nQOIs])                      #nSamp x nQOI out matrix from A
    fB=evalFcn(sampB).reshape([nSamp,model.nQOIs])                      #nSamp x nQOI out matrix from B
    fD=np.concatenate((fA, fB),axis=0)
    #Initialize combined QOI sample matrices
    if model.nQOIs==1:
        fAB=np.empty([nSamp,model.nPOIs])
    else:
        fAB=np.empty([nSamp, model.nPOIs, model.nQOIs])
    for iParams in range(0,model.nPOIs):
        #Define sampC to be A with the ith parameter in B
        sampAB=sampA
        sampAB[:, iParams]=sampB[:, iParams]
        if model.nQOIs==1:
            fAB[:,iParams]=evalFcn(sampAB)
        else:
            fAB[:,iParams,:]=evalFcn(sampAB)                           #nSamp x nPOI x nQOI tensor
    #QOI variance
    fDvar=np.var(fD)
    #fDvar=np.sum(fD**2)/(2*nSamp)-(np.sum(fD,axis=0)/(2*nSamp))**2


    sobolBase=np.empty((model.nQOIs,model.nPOIs))
    sobolTot=np.empty((model.nQOIs,model.nPOIs))
    if model.nQOIs==1:
        #Calculate 1st order parameter effects
        sobolBase=np.mean((fAB-fA)*fB, axis=0)/fDvar
        #Caclulate 2nd order parameter effects
        sobolTot=np.mean((fA-fAB)**2,axis=0)/fDvar

        #sobolTot=(np.dot(fA.transpose(),fA)-2*np.dot(fA.transpose(),fC)+np.dot(fC.transpose(),fC))/(fDvar*2*nSamp)
    else:
        for iQOI in range(0,model.nQOIs):
            #Calculate 1st order parameter effects
            sobolBase[iQOI,:]=1/(nSamp)*(np.sum(fAB[:,:,iQOI]*fA[:, [iQOI]],axis=0)-np.sum(fA[:,iQOI]*fB[:,iQOI],axis=0))/(nSamp*fDvar[iQOI])
            #Caclulate 2nd order parameter effects
            sobolTot[iQOI,:]=(np.sum(fA[:,iQOI]**2,axis=0)-2*np.sum(fA[:, [iQOI]]*fAB[:,:,iQOI],axis=0)\
                                          +np.sum(fAB[:,:,iQOI]**2,axis=0))/(2*nSamp*fDvar[iQOI])

    print(sampA[:10])

    return sobolBase, sobolTot

##--------------------------------------GetSampDist----------------------------------------------------
def GetSampDist(model, sampOptions):
    # Determine Sample Function- Currently only 1 distribution type can be defined for all parameters
    if model.dist.lower() == 'norm':  # Normal Distribution
        sampDist = lambda nSamp: np.random.randn(nSamp,model.nPOIs)*np.sqrt(np.diag(model.cov))+model.basePOIs
    elif model.dist.lower() == 'unif':  # uniform distribution
        sampDist = lambda nSamp: model.distParms[[0],:]+(model.distParms[[1],:]-model.distParms[[0],:])\
                                 *np.random.uniform(0,1,size=(nSamp,model.nPOIs))
    elif model.dist.lower() == 'exponential': # exponential distribution
        sampDist = lambda nSamp: np.random.exponential(model.distParms,size=(nSamp,model.nPOIs))
    elif model.dist.lower() == 'beta': # beta distribution
        sampDist = lambda nSamp:np.random.beta(model.distParms[[0],:], model.distParms[[1],:],\
                                               size=(nSamp,model.nPOIs))
    elif model.dist.lower() == 'InverseCDF': #Arbitrary distribution given by inverse cdf
        sampDist = lambda nSamp: sampOptions.fInverseCDF(np.random.rand(nsamp,model.nPOIs))
    else:
        raise Exception("Invalid value for options.samp.dist. Supported distributions are normal, uniform, exponential, beta, \
        and InverseCDF")  # Raise Exception if invalide distribution is entered
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
