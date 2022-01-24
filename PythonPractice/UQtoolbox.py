#UQtoolbox
#Module for calculating local sensitivity indices, parameter correlation, and Sobol indices for an arbitrary model
#Authors: Harley Hanes, NCSU, hhanes@ncsu.edu
#Required Modules: numpy, seaborne
#Functions: LSA->GetJacobian
#           GSA->ParamSample, PlotGSA, GetSobol

#3rd party Modules
import numpy as np
import sys
import warnings
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from tabulate import tabulate                       #Used for printing tables to terminal
#import sobol                                        #Used for generating sobol sequences
import SALib.sample.saltelli as sobol
import scipy.stats as sct

#Package Modules
import lsa
import gsa
#import seaborne as seaborne
###----------------------------------------------------------------------------------------------
###-------------------------------------Class Definitions----------------------------------------
###----------------------------------------------------------------------------------------------

##--------------------------------------uqOptions--------------------------------------------------
#Define class "uqOptions", this will be the class used to collect algorithm options for functions
#   -Subclasses: lsaOptions, plotOptions, gsaOptions
#--------------------------------------lsaOptions------------------------------------------------

#--------------------------------------gsaOptions------------------------------------------------

#--------------------------------------plotOptions------------------------------------------------
class plotOptions:
    def __init__(self,run=True,nPoints=400,path=False):
        self.run=run
        self.nPoints=nPoints
        self.path=path
        pass
#--------------------------------------uqOptions------------------------------------------------
#   Class holding the above options subclasses
class uqOptions:
    def __init__(self,lsa=lsa.options(),plot=plotOptions(),gsa=gsa.options(), \
                 display=True, save=False, path='..'):
        self.lsa=lsa
        self.plot=plot
        self.gsa=gsa
        self.display=display                       #Whether to print results to terminal
        self.save=save                             #Whether to save results to files
        self.path=path                             #Where to save files
        if self.save and not self.path:
            warnings.warn("Save marked as true but no path given, saving files to current folder.")
            path=''
    pass

##-------------------------------------model------------------------------------------------------
#Define class "model", this will be the class used to collect input information for all functions
class model:
    #Model sets should be initialized with base parameter settings, covariance Matrix, and eval function that
    #   takes in a vector of POIs and outputs a vector of QOIs
    def __init__(self,basePOIs=np.empty(0), POInames = np.empty(0), \
                 QOInames= np. empty(0), cov=np.empty(0), \
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
        print(self.baseQOIs)
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
            if self.dist.lower()=='uniform':
                self.distParms=[[.8],[1.2]]*np.ones((2,self.nPOIs))*self.basePOIs
            elif self.dist.lower()=='normal':
                if cov.size()==0:
                    self.distParms=[[1],[.2]]*np.ones((2,self.nPOIs))*self.basePOIs
                else:
                    self.distParms=[self.basePOIs, np.diag(self.cov,k=0)]
            elif distParms.lower() == 'null':
                self.distParms = distParms
            else:
                raise Exception("Unrecognized entry for distParms: " + str(distParms))

        else:
            self.distParms=distParms
    pass
    def copy(self):
        return model(basePOIs=self.basePOIs, POInames = self.POInames, QOInames= self.QOInames, cov=self.cov, \
                 evalFcn=self.evalFcn, dist=self.dist,distParms=self.distParms)

##------------------------------------results-----------------------------------------------------
#-------------------------------------lsaResults--------------------------------------------------
# Define class "lsa", this will be the used to collect relative sensitivity analysis outputs

#-------------------------------------gsaResults--------------------------------------------------
# Define class "gsaResults" which holds sobol analysis results

##------------------------------------results-----------------------------------------------------
# Define class "results" which holds a gsaResults object and lsaResults object

class results:
    def __init__(self,lsa=lsa.results(), gsa=gsa.results()):
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
        results.lsa = lsa.RunLSA(model, options)

    #Run Global Sensitivity Analysis
    # if options.gsa.run:
        # if options.lsa.run:
        #     #Use a reduced model if it was caluclated
        #     results.gsa=GSA(results.lsa.reducedModel, options)
        # else:
    if options.gsa.run:
        results.gsa = gsa.RunGSA(model, options)

    #Print Results
    if options.display:
        PrintResults(results,model,options)                     #Print results to standard output path

    if options.save:
        original_stdout = sys.stdout                            #Save normal output path
        sys.stdout=open(options.path + 'Results.txt', 'a+')            #Change output path to results file
        PrintResults(results,model,options)                     #Print results to file
        sys.stdout=original_stdout                              #Revert normal output path

    #Plot Samples
    if options.gsa.runSobol & options.gsa.run:
        PlotGSA(model, results.gsa.sampD, results.gsa.fD, options)

    return results



def PrintResults(results,model,options):
    # Print Results
    #Results Header
    print('Sensitivity results for nSampSobol=' + str(options.gsa.nSampSobol))
    #Local Sensitivity Analysis
    if options.lsa.run:
        print('\n Base POI Values')
        print(tabulate([model.basePOIs], headers=model.POInames))
        print('\n Base QOI Values')
        print(tabulate([model.baseQOIs], headers=model.QOInames))
        print('\n Sensitivity Indices')
        print(tabulate(np.concatenate((model.POInames.reshape(model.nPOIs,1),np.transpose(results.lsa.jac)),1),
              headers= np.append("",model.QOInames)))
        print('\n Relative Sensitivity Indices')
        print(tabulate(np.concatenate((model.POInames.reshape(model.nPOIs,1),np.transpose(results.lsa.rsi)),1),
              headers= np.append("",model.QOInames)))
        #print("Fisher Matrix: " + str(results.lsa.fisher))
        #Active Subsapce Analysis
        print('\n Active Supspace')
        print(results.lsa.activeSpace)
        print('\n Inactive Supspace')
        print(results.lsa.inactiveSpace)
    if options.gsa.run: 
        if options.gsa.runSobol:
            if model.nQOIs==1:
                print('\n Sobol Indices for ' + model.QOInames[0])
                print(tabulate(np.concatenate((model.POInames.reshape(model.nPOIs,1), results.gsa.sobolBase.reshape(model.nPOIs,1), \
                                               results.gsa.sobolTot.reshape(model.nPOIs,1)), 1),
                               headers=["", "1st Order", "Total Sensitivity"]))
            else:
                for iQOI in range(0,model.nQOIs):
                    print('\n Sobol Indices for '+ model.QOInames[iQOI])
                    print(tabulate(np.concatenate((model.POInames.reshape(model.nPOIs,1),results.gsa.sobolBase[[iQOI],:].reshape(model.nPOIs,1), \
                        results.gsa.sobolTot[[iQOI],:].reshape(model.nPOIs,1)),1), headers = ["", "1st Order", "Total Sensitivity"]))
    
        if options.gsa.runMorris:
            if model.nQOIs==1:
                print('\n Morris Screening Results for' + model.QOInames[0])
                print(tabulate(np.concatenate((model.POInames.reshape(model.nPOIs, 1), results.gsa.muStar.reshape(model.nPOIs, 1), \
                                               results.gsa.sigma2.reshape(model.nPOIs, 1)), 1),
                    headers=["", "muStar", "sigma2"]))
            else:
                print('\n Morris Screening Results for' + model.QOInames[iQOI])
                print(tabulate(np.concatenate(
                    (model.POInames.reshape(model.nPOIs, 1), results.gsa.muStar[[iQOI], :].reshape(model.nPOIs, 1), \
                     results.gsa.sigma2[[iQOI], :].reshape(model.nPOIs, 1)), 1),
                    headers=["", "muStar", "sigma2"]))

###----------------------------------------------------------------------------------------------
###-------------------------------------Support Functions----------------------------------------
###----------------------------------------------------------------------------------------------


##--------------------------------------GetSobol------------------------------------------------------
# GSA Component Functions


#
#
def PlotGSA(model, sampleMat, evalMat, options):
    #Reduce Sample number
    #plotPoints=range(0,int(sampleMat.shape[0]), int(sampleMat.shape[0]/plotOptions.nPoints))
    #Make the number of sample points to survey
    plotPoints=np.linspace(start=0, stop=sampleMat.shape[0]-1, num=options.plot.nPoints, dtype=int)
    #Plot POI-POI correlation and distributions
    figure, axes=plt.subplots(nrows=model.nPOIs, ncols= model.nPOIs, squeeze=False)
    for iPOI in range(0,model.nPOIs):
        for jPOI in range(0,iPOI+1):
            if iPOI==jPOI:
                n, bins, patches = axes[iPOI, jPOI].hist(sampleMat[:,iPOI], bins=41)
            else:
                axes[iPOI, jPOI].plot(sampleMat[plotPoints,iPOI], sampleMat[plotPoints,jPOI],'b*')
            if jPOI==0:
                axes[iPOI,jPOI].set_ylabel(model.POInames[iPOI])
            if iPOI==model.nPOIs-1:
                axes[iPOI,jPOI].set_xlabel(model.POInames[jPOI])
            if model.nPOIs==1:
                axes[iPOI,jPOI].set_ylabel('Instances')
    figure.tight_layout()
    if options.path:
        plt.savefig(options.path+"POIcorrelation.png")

    #Plot QOI-QOI correlationa and distributions
    figure, axes=plt.subplots(nrows=model.nQOIs, ncols= model.nQOIs, squeeze=False)
    for iQOI in range(0,model.nQOIs):
        for jQOI in range(0,iQOI+1):
            if iQOI==jQOI:
                axes[iQOI, jQOI].hist([evalMat[:,iQOI]], bins=41)
            else:
                axes[iQOI, jQOI].plot(evalMat[plotPoints,iQOI], evalMat[plotPoints,jQOI],'b*')
            if jQOI==0:
                axes[iQOI,jQOI].set_ylabel(model.QOInames[iQOI])
            if iQOI==model.nQOIs-1:
                axes[iQOI,jQOI].set_xlabel(model.QOInames[jQOI])
            if model.nQOIs==1:
                axes[iQOI,jQOI].set_ylabel('Instances')
    figure.tight_layout()
    if options.path:
        plt.savefig(options.path+"QOIcorrelation.png")

    #Plot POI-QOI correlation
    figure, axes=plt.subplots(nrows=model.nQOIs, ncols= model.nPOIs, squeeze=False)
    for iQOI in range(0,model.nQOIs):
        for jPOI in range(0, model.nPOIs):
            axes[iQOI, jPOI].plot(sampleMat[plotPoints,jPOI], evalMat[plotPoints,iQOI],'b*')
            if jPOI==0:
                axes[iQOI,jPOI].set_ylabel(model.QOInames[iQOI])
            if iQOI==model.nQOIs-1:
                axes[iQOI,jPOI].set_xlabel(model.POInames[jPOI])
    if options.path:
        plt.savefig(options.path+"POI_QOIcorrelation.png")
    #Display all figures
    if options.display:
        plt.show()

