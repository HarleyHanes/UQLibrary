#UQtoolbox
#Module for calculating local sensitivity indices, parameter correlation, and Sobol indices for an arbitrary model
#Authors: Harley Hanes, NCSU, hhanes@ncsu.edu
#Required Modules: numpy, seaborne
#Functions: LSA->GetJacobian
#           GSA->ParamSample, PlotGSA, GetSobol

import numpy as np
import sys
import warnings
import matplotlib.pyplot as plt
#import seaborne as seaborne
###----------------------------------------------------------------------------------------------
###-------------------------------------Class Definitions----------------------------------------
###----------------------------------------------------------------------------------------------

##--------------------------------------uqOptions--------------------------------------------------
#Define class "uqOptions", this will be the class used to collect algorithm options for functions
#   -Subclasses: lsaOptions, plotOptions, gsaOptions
#--------------------------------------lsaOptions------------------------------------------------
class lsaOptions:
    def __init__(self,run=True,  xDelta=10**(-12), method='complex', scale='y'):
        self.run=run                              #Whether to run lsa (True or False)
        self.xDelta=xDelta                        #Input perturbation for calculating jacobian
        self.scale=scale                          #scale can be y, n, or both for outputing scaled, unscaled, or both
        self.method=method                        #method used for approximating derivatives
        if not self.scale.lower() in ('y','n','both'):
            raise Exception('Error! Unrecgonized scaling output, please enter y, n, or both')
        if not self.method.lower() in ('complex','finite'):
            raise Exception('Error! unrecognized derivative approx method. Use complex or finite')
        if self.xDelta<0 or not isinstance(self.xDelta,float):
            raise Exception('Error! Non-compatibale xDelta, please use a positive floating point number')
    pass
#--------------------------------------gsaOptions------------------------------------------------
class gsaOptions:
    def __init__(self, run=True, nSamp=100000):
        self.run=run                            #Whether to run GSA (True or False)
        self.nSamp = nSamp                      #Number of samples to be generated for GSA
        pass
#--------------------------------------plotOptions------------------------------------------------
class plotOptions:
    def __init__(self,run=True,nPoints=100):
        self.run=run
        self.nPoints=nPoints
        pass
#--------------------------------------uqOptions------------------------------------------------
#   Class holding the above options subclasses
class uqOptions:
    def __init__(self,lsa=lsaOptions(),plot=plotOptions(),samp=gsaOptions()):
        self.lsa=lsa
        self.plot=plot
        self.samp=samp
    pass

##-------------------------------------model------------------------------------------------------
#Define class "model", this will be the class used to collect input information for all functions
class model:
    #Model sets should be initialized with base parameter settings, covariance Matrix, and eval function that
    #   takes in a vector of POIs and outputs a vector of QOIs
    def __init__(self,basePOIs=np.empty(0), POInames = np.empty(0), QOInames= np. empty(0), cov=np.empty(0), \
                 evalFcn=np.empty(0), dist='unif',distParms='null'):
        self.basePOIs=basePOIs
        if not isinstance(self.basePOIs,np.ndarray):                    #Confirm that basePOIs is a numpy array
            warnings.warn("model.basePOIs is not a numpy array")
        if np.ndim(self.basePOIs)>1:                                    #Check to see if basePOIs is a vector
            self.basePOIs=np.squeeze(self.basePOIs)                     #Make a vector if an array with 1 dim greater than 1
            if np.ndim(self.basePOIs)!=1:                               #Issue an error if basePOIs is a matrix or tensor
                raise Exception("Error! More than one dimension of size 1 detected for model.basePOIs, model.basePOIs must be dimension 1")
            else:                                                       #Issue a warning if dimensions were squeezed out of base POIs
                warnings.warn("model.basePOIs was reduced a dimension 1 array. No entries were deleted.")
        self.nPOIs=self.basePOIs.size
        #Assign POInames
        self.POInames = POInames                                            #Assign POInames called
        if (self.POInames.size != self.nPOIs) & (self.POInames.size !=0):   #Check that correct size if given
            warnings.warn("POInames entered but the number of names does not match the number of POIs. Ignoring names.")
            self.POInames=np.empty(0)
        if self.POInames.size==0:                                           #If not given or incorrect size, number POIs
            POInumbers=np.arange(0,self.nPOIs)
            self.POInames=np.char.add('POI',POInumbers.astype('U'))
        #Assign evaluation function and compute baseQOIs
        self.evalFcn=evalFcn
        self.baseQOIs=evalFcn(basePOIs)
        if not isinstance(self.baseQOIs,np.ndarray):                    #Confirm that baseQOIs is a numpy array
            warnings.warn("model.baseQOIs is not a numpy array")
        self.nQOIs=len(self.baseQOIs)
        #Assign QOI names
        self.QOInames = QOInames
        if (self.QOInames.size !=self.nQOIs) & (self.QOInames.size !=0):    #Check names if given match number of QOIs
            warnings.warn("QOInames entered but the number of names does not match the number of QOIs. Ignoring names.")
            self.QOInames = np.empty(0)
        if self.QOInames.size==0:                                 #If not given or incorrect size, number QOIs
            QOInumbers = np.arange(0, self.nQOIs)
            self.QOInames = np.char.add('QOI', QOInumbers.astype('U'))
        #Assign covariance matrix
        self.cov=cov
        if self.cov.size!=0 and np.shape(self.cov)!=(self.nPOIs,self.nPOIs): #Check correct sizing
            raise Exception("Error! model.cov is not an nPOI x nPOI array")
        #Assign distributions
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
    def __init__(self,sobolBase=np.empty, sobolTot=np.empty, fD=np.empty, fAB=np.empty, sampD=np.empty):
        self.sobolBase=sobolBase
        self.sobolTot=sobolTot
        self.fD=fD
        self.fAB=fAB
        self.sampD=sampD
    pass
##------------------------------------results-----------------------------------------------------
# Define class "results" which holds a gsaResults object and lsaResults object

class results:
    def __init__(self,lsa=lsaResults(), gsa=gsaResults()):
        self.lsa=lsa
        self.gsa=gsa
    pass


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
    if options.lsa.run:
        results.lsa = LSA(model, options)

    #Run Global Sensitivity Analysis
    if options.gsa.run:
        results.gsa = GSA(model, options)

    #Print Results
    PrintResults(results,model,options)

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
    jacRaw=GetJacobian(model, options.lsa, scale=False)
    # Calculate relative sensitivity index (RSI)
    jacRSI=GetJacobian(model, options.lsa, scale=True)
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
    model=GetSampDist(model, options.gsa)
    #Make Distribution Samples and Calculate model results
    [fA, fB, fAB, fD, sampD] = GetSamples(model, options.gsa)
    #Plot Sample Statistics
    if options.plot.run:
        PlotGSA(model, sampD, fD, options.plot)
    #Calculate Sobol Indices
    [sobolBase, sobolTot]=CalculateSobol(fA, fB, fAB, fD)
    return gsaResults(fD=fD, fAB=fAB, sampD= sampD, sobolBase=sobolBase, sobolTot=sobolTot)

def PrintResults(results,model,options):
    # Print Results
    if options.lsa.run:
        print("Base Parameters: " + str(model.basePOIs))
        print("Base values: " + str(model.baseQOIs))
        print("Jacobian: " + str(results.lsa.jac))
        print("Relative Sensitivities: " + str(results.lsa.rsi))
        print("Fisher Matrix: " + str(results.lsa.fisher))
    if options.gsa.run:
        print("1st Order Sobol Indices: " + str(results.gsa.sobolBase))
        print("Total Sobol Indices: " + str(results.gsa.sobolTot))

###----------------------------------------------------------------------------------------------
###-------------------------------------Support Functions----------------------------------------
###----------------------------------------------------------------------------------------------

##--------------------------------------GetJacobian-----------------------------------------------------
def GetJacobian(model, lsaOptions, **kwargs):
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

    #Load options parameters for increased readibility
    xBase=model.basePOIs
    xDelta=lsaOptions.xDelta

    #Initialize base QOI value, the number of POIs, and number of QOIs
    yBase=model.evalFcn(xBase)                                              # Get base QOI values
    nPOIs = model.nPOIs                                                     # Get number of parameters (nPOIs)
    nQOIs = model.nQOIs                                                     # Get number of outputs (nQOIs)

    jac = np.empty(shape=(nQOIs, nPOIs), dtype=float)                       # Define Empty Jacobian Matrix

    for iPOI in range(0, nPOIs):                                            # Loop through POIs
        # Isolate Parameters
        if lsaOptions.method.lower()== 'complex':
            xPert = xBase + np.zeros(shape=xBase.shape)*1j                  # Initialize Complex Perturbed input value
            xPert[iPOI] += xDelta * 1j                                      # Add complex Step in input
        elif lsaOptions.method.lower() == 'finite':
            xPert=xBase*(1+xDelta)
        yPert = model.evalFcn(xPert)                                        # Calculate perturbed output
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


##--------------------------------------GetSobol----------------------------------------------------
# GSA Component Functions

def GetSamples(model,gsaOptions):
    nSamp = gsaOptions.nSamp
    sampDist = model.sampDist
    evalFcn = model.evalFcn
    # Make 2 POI sample matrices with nSamp samples each
    sampA = sampDist(nSamp)
    sampB = sampDist(nSamp)
    # Calculate matrices of QOI values for each POI sample matrix
    fA = evalFcn(sampA).reshape([nSamp, model.nQOIs])  # nSamp x nQOI out matrix from A
    fB = evalFcn(sampB).reshape([nSamp, model.nQOIs])  # nSamp x nQOI out matrix from B
    # Stack the output matrices into a single matrix
    fD = np.concatenate((fA.copy(), fB.copy()), axis=0)

    # Initialize combined QOI sample matrices
    if model.nQOIs == 1:
        fAB = np.empty([nSamp, model.nPOIs])
    else:
        fAB = np.empty([nSamp, model.nPOIs, model.nQOIs])
    for iParams in range(0, model.nPOIs):
        # Define sampC to be A with the ith parameter in B
        sampAB = sampA.copy()
        sampAB[:, iParams] = sampB[:, iParams].copy()
        if model.nQOIs == 1:
            fAB[:, iParams] = evalFcn(sampAB)
        else:
            fAB[:, iParams, :] = evalFcn(sampAB)  # nSamp x nPOI x nQOI tensor
        del sampAB
    return fA, fB, fAB, fD, np.concatenate((sampA.copy(), sampB.copy()), axis=0)

def CalculateSobol(fA, fB, fAB, fD):
    #Calculates calculates sobol indices using satelli approximation method
    #Inputs: model object (with evalFcn, sampDist, and nParams)
    #        sobolOptions object
    #Determing number of samples, QOIs, and POIs based on inputs
    nSamp=fAB.shape[0]
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
        raise(Exception('fAB has greater than 3 dimensions, make sure fAB is the squeezed form of nSamp x nPOI x nQOI'))
    #QOI variance
    fDvar=np.var(fD, axis=0)

    sobolBase=np.empty((nQOIs, nPOIs))
    sobolTot=np.empty((nQOIs, nPOIs))
    if nQOIs==1:
        #Calculate 1st order parameter effects
        sobolBase=np.sum(fB*(fAB-fA), axis=0)/(nSamp*fDvar)

        #Caclulate 2nd order parameter effects
        sobolTot=np.sum((fA-fAB)**2, axis=0)/(2*nSamp*fDvar)

    else:
        for iQOI in range(0,nQOIs):
            #Calculate 1st order parameter effects
            sobolBase[iQOI,:]=1/(nSamp)*(np.sum(fAB[:,:,iQOI]*fA[:, [iQOI]],axis=0)-np.sum(fA[:,iQOI]*fB[:,iQOI],axis=0))/(nSamp*fDvar[iQOI])
            #Caclulate 2nd order parameter effects
            sobolTot[iQOI,:]=(np.sum(fA[:,iQOI]**2,axis=0)-2*np.sum(fA[:, [iQOI]]*fAB[:,:,iQOI],axis=0)\
                                          +np.sum(fAB[:,:,iQOI]**2,axis=0))/(2*nSamp*fDvar[iQOI])


    return sobolBase, sobolTot

##--------------------------------------GetSampDist----------------------------------------------------
def GetSampDist(model, gsaOptions):
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
        sampDist = lambda nSamp: gsaOptions.fInverseCDF(np.random.rand(nSamp,model.nPOIs))
    else:
        raise Exception("Invalid value for options.gsa.dist. Supported distributions are normal, uniform, exponential, beta, \
        and InverseCDF")  # Raise Exception if invalide distribution is entered
    model.sampDist=sampDist
    return model


#
#
def PlotGSA(model, sampleMat, evalMat, plotOptions):
    #Reduce Sample number
    plotPoints=range(0,int(sampleMat.shape[0]), int(sampleMat.shape[0]/plotOptions.nPoints))
    #Plot POI-POI correlation and distributions
    figure, axes=plt.subplots(nrows=model.nPOIs, ncols= model.nPOIs, squeeze=False)
    for iPOI in range(0,model.nPOIs):
        for jPOI in range(0,iPOI+1):
            if iPOI==jPOI:
                n, bins, patches = axes[iPOI, jPOI].hist(sampleMat[:,iPOI])
            else:
                axes[iPOI, jPOI].plot(sampleMat[plotPoints,iPOI], sampleMat[plotPoints,jPOI],'b*')
            if jPOI==0:
                axes[iPOI,jPOI].set_ylabel(model.POInames[iPOI])
            if iPOI==model.nPOIs-1:
                axes[iPOI,jPOI].set_xlabel(model.POInames[jPOI])
            if model.nPOIs==1:
                axes[iPOI,jPOI].set_ylabel('Instances')
    figure.tight_layout()
    #Plot QOI-QOI correlationa and distributions
    figure, axes=plt.subplots(nrows=model.nQOIs, ncols= model.nQOIs, squeeze=False)
    for iQOI in range(0,model.nQOIs):
        for jQOI in range(0,iQOI+1):
            if iQOI==jQOI:
                axes[iQOI, jQOI].hist([evalMat[:,iQOI]])
            else:
                axes[iQOI, jQOI].plot(evalMat[plotPoints,iQOI], evalMat[plotPoints,jQOI],'b*')
            if jQOI==0:
                axes[iQOI,jQOI].set_ylabel(model.QOInames[iQOI])
            if iQOI==model.nQOIs-1:
                axes[iQOI,jQOI].set_xlabel(model.QOInames[jQOI])
            if model.nQOIs==1:
                axes[iQOI,jQOI].set_ylabel('Instances')
    figure.tight_layout()

    #Plot POI-QOI correlation
    figure, axes=plt.subplots(nrows=model.nQOIs, ncols= model.nPOIs, squeeze=False)
    for iQOI in range(0,model.nQOIs):
        for jPOI in range(0, model.nPOIs):
            axes[iQOI, jPOI].plot(sampleMat[plotPoints,jPOI], evalMat[plotPoints,iQOI],'b*')
            if jPOI==0:
                axes[iQOI,jPOI].set_ylabel(model.QOInames[iQOI])
            if iQOI==model.nQOIs-1:
                axes[iQOI,jPOI].set_xlabel(model.POInames[jPOI])
    #Display all figures
    plt.show()

