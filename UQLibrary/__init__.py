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
from tabulate import tabulate   

import mpi4py.MPI as MPI

#Package Modules
from . import lsa
from . import gsa
from . import examples
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
class PlotOptions:
    def __init__(self,run=True,n_points=400,path=False):
        self.run=run
        self.n_points=n_points
        self.path=path
        pass
#--------------------------------------uqOptions------------------------------------------------
#   Class holding the above options subclasses
class Options:
    def __init__(self,lsa=lsa.LsaOptions(),plot=PlotOptions(),gsa=gsa.GsaOptions(), \
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
class Model:
    #Model sets should be initialized with base parameter settings, covariance Matrix, and eval function that
    #   takes in a vector of POIs and outputs a vector of QOIs
    def __init__(self,base_poi=np.empty(0), name_poi = "auto", \
                 name_qoi= "auto", cov=np.empty(0), \
                 eval_fcn=np.empty(0), dist_type='uniform', dist_param="auto"):
        #------------------------base_poi, n_poi, name_poi---------------------
        #Assign base_poi and n_poi
        if not isinstance(base_poi,np.ndarray):                    #Confirm that base_poi is a numpy array
            raise Exception("model.base_poi is not a numpy array")
        if np.ndim(base_poi)>1:                                    #Check to see if base_poi is a vector
            base_poi=np.squeeze(base_poi)                     #Make a vector if an array with 1 dim greater than 1
            if np.ndim(base_poi)!=1:                               #Issue an error if base_poi is a matrix or tensor
                raise Exception("Error! More than one dimension of size 1 detected for model.base_poi, model.base_poi must be dimension 1")
            else:                                                       #Issue a warning if dimensions were squeezed out of base POIs
                warnings.warn("model.base_poi was reduced a dimension 1 array. No entries were deleted.")
        self.base_poi=base_poi
        del base_poi
        self.n_poi=self.base_poi.size
        
        #Assign name_poi----------------UNFINISHED VALUE CHECKING
        
        POInumbers=np.arange(0,self.n_poi)
        name_poi_auto=np.char.add('POI',POInumbers.astype('U'))
        #Check name_poi is string
        if type(name_poi)==np.ndarray:
            #Check data type
            if name_poi.size!= self.n_poi:
                raise Exception("Incorrect number of entries in name_poi")
        elif type(name_poi)==list:
            #Check data type
            if len(name_poi)!= self.n_poi:
                raise Exception("Incorrect number of entries in name_poi")
            name_poi=np.array(name_poi)
        elif type(name_poi)==str and name_poi.lower()!="auto":
            if self.n_poi!=1:
                raise Exception("Only one qoi name entered for >1 pois")
        else :
            if name_poi.lower()!= "auto":
                warnings.warn("Unrecognized name_poi entry, using automatic values")
            name_poi = name_poi_auto
        if (name_poi.size != self.n_poi) & (name_poi.size !=0):   #Check that correct size if given
            warnings.warn("name_poi entered but the number of names does not match the number of POIs. Ignoring names.")
            name_poi=np.empty(0)
        if name_poi.size==0:                       
            name_poi = name_poi_auto
        self.name_poi = name_poi
        del name_poi
        #-----------------eval_fcn, base_qoi, n_qoi, name_qoi------------------
        #Assign evaluation function and compute base_qoi
        self.eval_fcn=eval_fcn
        del eval_fcn
        self.base_qoi=self.eval_fcn(self.base_poi)
        if not isinstance(self.base_qoi,np.ndarray):                    #Confirm that base_qoi is a numpy array
            warnings.warn("model.base_qoi is not a numpy array")
        self.n_qoi=len(self.base_qoi)
        
        #Assign name_qoi----------------UNFINISHED VALUE CHECKING
        #Formulate automatic names so they can be referenced in each case
        QOInumbers=np.arange(0,self.n_qoi)
        name_qoi_auto=np.char.add('POI',QOInumbers.astype('U'))
        #Check name_qoi is string
        if type(name_qoi)==np.ndarray:
            #Check data type
            if name_qoi.size!= self.n_qoi:
                raise Exception("Incorrect number of entries in name_qoi")
        elif type(name_qoi)==list:
            #Check data type
            if len(name_qoi)!= self.n_qoi:
                raise Exception("Incorrect number of entries in name_qoi")
            name_qoi=np.array(name_qoi)
        elif type(name_qoi)==str and name_qoi.lower()!="auto":
            if self.n_qoi!=1:
                raise Exception("Only one qoi name entered for >1 qois")
        else :
            if name_qoi.lower()!= "auto":
                warnings.warn("Unrecognized name_qoi entry, using automatic values")
            name_qoi= name_qoi_auto
        if (name_qoi.size !=self.n_qoi) & (name_qoi.size !=0):    #Check names if given match number of QOIs
            warnings.warn("name_qoi entered but the number of names does not match the number of QOIs. Ignoring names.")
            name_qoi = name_qoi_auto
        if name_qoi.size==0:                                 #If not given or incorrect size, number QOIs
            name_qoi = name_qoi_auto
        self.name_qoi = name_qoi
        del name_qoi
            
            
            
            
        #------------------------------covariance matrix-----------------------
        self.cov=cov
        if self.cov.size!=0 and np.shape(self.cov)!=(self.n_poi,self.n_poi): #Check correct sizing
            raise Exception("Error! model.cov is not an nPOI x nPOI array")
            
        #--------------------------------dist_type-----------------------------
        #Only allow distributions that are currently fully implemented
        valid_distribution =np.array(["uniform", "saltelli uniform", "normal", \
                                      "saltelli normal"])
        # valid_distribution =np.array(["uniform", "normal", "exponential", \
        #                           "saltelli normal", "beta", "InverseCDF"])
        #If distribution type is valid, save its value
        if np.all(dist_type!= valid_distribution):
            raise Exception(str(dist_type) + " is an invalid distribution. Valid" +\
                            " distributions are" + str(valid_distribution))
        else:
            self.dist_type=dist_type
            del dist_type
                
        #--------------------------------dist_param----------------------------
        # Apply automatic distribution parameter settings
        if str(dist_param).lower() == 'auto':
            if (self.dist_type == "uniform" or self.dist_type == "saltelli uniform"):
                self.dist_param=[[.8],[1.2]]*np.ones((2,self.n_poi))*self.base_poi
                del dist_param
            elif (self.dist_type == "normal" or self.dist_type == "saltelli normal"):
                if cov.size()==0:
                    self.dist_param=[[1],[.2]]*np.ones((2,self.n_poi))*self.base_poi
                    del dist_param
        # Apply manual distribution settings
        elif type(dist_param) == np.ndarray:
            #Check dimensions of numpy array are correct
            if dist_param.shape[1] == self.n_poi:
                # Correct number of parameters for each distribution
                if (self.dist_type == "uniform" or self.dist_type == "saltelli uniform")\
                    and dist_param.shape[0]!=2:
                    raise Exception("2 parameters per POI required for uniform")
                elif (self.dist_type == "normal" or self.dist_type == "saltelli normal")\
                    and dist_param.shape[0]!=2:
                    raise Exception("2 parameters per POI required for normal")
                # Assign dist_param if conditions met
                else :
                    self.dist_param=dist_param
                    del dist_param
            else:
                raise Exception("Incorrect shape of dist_param. Given shape: "\
                                + dist_param.shape + ", desired shape: ... x n_poi") 
        elif dist_param.lower() == "cov":
            if np.any(self.dist_type.lower()==["normal", "saltelli normal"]):
                self.dist_param=[self.base_poi, np.diag(self.cov,k=0)]   
                del dist_param 
            else :
                raise Exception("Covariance based sampling only implemented for"\
                                +"normal or saltelli normal distributions")
        else:
            raise Exception("Incorrect data-type for dist_param, use ndarray, 'auto', or 'cov'")
            
        #Construct Distribution function
        self.sample_fcn = gsa.get_samp_dist(self.dist_type, self.dist_param, self.n_poi)
    
    pass
    def copy(self):
        return Model(base_poi=self.base_poi, name_poi = self.name_poi, name_qoi= self.name_qoi, cov=self.cov, \
                 eval_fcn=self.eval_fcn, dist_type=self.dist_type,dist_param=self.dist_param)

##------------------------------------results-----------------------------------------------------
# Define class "results" which holds a gsaResults object and lsaResults object

class Results:
    def __init__(self,lsa=lsa.LsaResults(), gsa=gsa.GsaResults()):
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
def run_uq(model, options, logging = False):
    """Runs both the local and global sensitivity, and result printing, saving, and plotting.
    
    Parameters
    ----------
    model : Model
        Object of class Model holding run information.
    options : Options
        Object of class Options holding run settings.
    logging : bool or int
        Holds level 
        
    Returns
    -------
    Results 
        Object of class Results holding all run results.
    """
    mpi_comm = MPI.COMM_WORLD
    mpi_rank = mpi_comm.Get_rank()
    mpi_size = mpi_comm.Get_size()
    
    if logging : 
        print("Starting UQ with precessor " + str(mpi_rank) + " of "+ str(mpi_size) + ".")
    
    #Only hold results in base processor
    results = Results()
        
    #Run Local Sensitivity Analysis only on base processor
    if mpi_rank == 0 and options.lsa.run:
        if logging: 
            print("Starting LSA")
        results.lsa = lsa.run_lsa(model, options.lsa)
        #---------------Broadcast results.lsa to other threads-----------

    #Run Global Sensitivity Analysis
    # if options.gsa.run:
        # if options.lsa.run:
        #     #Use a reduced model if it was caluclated
        #     results.gsa=GSA(results.lsa.reducedModel, options)
        # else:
    if options.gsa.run:
        if logging: 
            print("Starting GSA")
        results.gsa = gsa.run_gsa(model, options.gsa, logging = logging)

    #Print, save, and plot Results only on thread 0
    if options.display and mpi_rank ==0:
        if logging: 
            print("Printing Results")
        print_results(results,model,options)                     #Print results to standard output path

    if options.save and mpi_rank == 0:
        if logging: 
            print("Saving Results")
        original_stdout = sys.stdout                            #Save normal output path
        sys.stdout=open(options.path + 'Results.txt', 'a+')            #Change output path to results file
        print_results(results,model,options)                     #Print results to file
        sys.stdout=original_stdout                              #Revert normal output path
        if options.gsa.run_morris:
            np.savez(options.path + "morris_indices.npz",\
                     morris_mean_abs = results.gsa.morris_mean_abs,
                     morris_mean = results.gsa.morris_mean,
                     morris_std = results.gsa.morris_std,
                     base_response = model.base_qoi)

    #Plot Samples
    if options.gsa.run_sobol and options.gsa.run and options.plot and mpi_rank == 0:
        if logging: 
            print("Plotting Results")
        plot_gsa(model, results.gsa.samp_d, results.gsa.f_d, options)

    return results



def print_results(results,model,options):
    """Prints Results object to console or document.
    
    Parameters
    ----------
    results : Results
        Object of class Model holding run information.
    model : Model
        Object of class Model holding run information.
    options : Options
        Object of class Options holding run settings.
    """
    # Print Results
    #Results Header
    #print('Sensitivity results for nSampSobol=' + str(options.gsa.n_samp_sobol))
    #Local Sensitivity Analysis
    if options.lsa.run:
        print('\n Base POI Values')
        print(tabulate([model.base_poi], headers=model.name_poi))
        print('\n Base QOI Values')
        print(tabulate([model.base_qoi], headers=model.name_qoi))
        print('\n Sensitivity Indices')
        print(tabulate(np.concatenate((model.name_poi.reshape(model.n_poi,1),np.transpose(results.lsa.jac)),1),
              headers= np.append("",model.name_qoi)))
        print('\n Relative Sensitivity Indices')
        print(tabulate(np.concatenate((model.name_poi.reshape(model.n_poi,1),np.transpose(results.lsa.rsi)),1),
              headers= np.append("",model.name_qoi)))
        #print("Fisher Matrix: " + str(results.lsa.fisher))
        #Active Subsapce Analysis
        if options.lsa.run_param_subset:
            print('\n Active Supspace')
            print(results.lsa.active_set)
            print('\n Inactive Supspace')
            print(results.lsa.inactive_set)
    if options.gsa.run: 
        if options.gsa.run_sobol:
            if model.n_qoi==1:
                print('\n Sobol Indices for ' + model.name_qoi[0])
                print(tabulate(np.concatenate((model.name_poi.reshape(model.n_poi,1), results.gsa.sobol_base.reshape(model.n_poi,1), \
                                               results.gsa.sobol_tot.reshape(model.n_poi,1)), 1),
                               headers=["", "1st Order", "Total Sensitivity"]))
            else:
                for i_qoi in range(0,model.n_qoi):
                    print('\n Sobol Indices for '+ model.name_qoi[i_qoi])
                    print(tabulate(np.concatenate((model.name_poi.reshape(model.n_poi,1),results.gsa.sobol_base[[i_qoi],:].reshape(model.n_poi,1), \
                        results.gsa.sobol_tot[[i_qoi],:].reshape(model.n_poi,1)),1), headers = ["", "1st Order", "Total Sensitivity"]))
    
        if options.gsa.run_morris:
            if model.n_qoi==1:
                print('\n Morris Screening Results for ' + model.name_qoi[0])
                print(tabulate(np.concatenate((model.name_poi.reshape(model.n_poi, 1), results.gsa.morris_mean_abs.reshape(model.n_poi, 1), \
                                               results.gsa.morris_std.reshape(model.n_poi, 1)), 1),
                    headers=["", "mu_star", "sigma"]))
            else:
                for i_qoi in range(model.n_qoi):
                    print('\n Morris Screening Results for ' + model.name_qoi[i_qoi])
                    print(tabulate(np.concatenate(
                        (model.name_poi.reshape(model.n_poi, 1), results.gsa.morris_mean_abs[:,[i_qoi]].reshape(model.n_poi, 1), \
                     results.gsa.morris_std[:,[i_qoi]].reshape(model.n_poi, 1)), 1),
                    headers=["", "mu_star", "sigma"]))

###----------------------------------------------------------------------------------------------
###-------------------------------------Support Functions----------------------------------------
###----------------------------------------------------------------------------------------------


##--------------------------------------GetSobol------------------------------------------------------
# GSA Component Functions


#
#
def plot_gsa(model, sample_mat, eval_mat, options):
    """Plots Sobol Sampling results from gsa module.
    
    Parameters
    ----------
    model : Model
        Object of class Model holding run information.
    sample_mat: np.ndarray
        n_samp x n_poi array holding each parameter sample
    eval_mat : np.ndarray
        n_samp x n_qoi array holding each function evaluation
    options : Options
        Object of class Options holding run settings.
    """
    #Reduce Sample number
    #plotPoints=range(0,int(sample_mat.shape[0]), int(sample_mat.shape[0]/plotOptions.n_points))
    #Make the number of sample points to survey
    plotPoints=np.linspace(start=0, stop=sample_mat.shape[0]-1, num=options.plot.n_points, dtype=int)
    #Plot POI-POI correlation and distributions
    figure, axes=plt.subplots(nrows=model.n_poi, ncols= model.n_poi, squeeze=False)
    for iPOI in range(0,model.n_poi):
        for jPOI in range(0,iPOI+1):
            if iPOI==jPOI:
                n, bins, patches = axes[iPOI, jPOI].hist(sample_mat[:,iPOI], bins=41)
            else:
                axes[iPOI, jPOI].plot(sample_mat[plotPoints,iPOI], sample_mat[plotPoints,jPOI],'b*')
            if jPOI==0:
                axes[iPOI,jPOI].set_ylabel(model.name_poi[iPOI])
            if iPOI==model.n_poi-1:
                axes[iPOI,jPOI].set_xlabel(model.name_poi[jPOI])
            if model.n_poi==1:
                axes[iPOI,jPOI].set_ylabel('Instances')
    figure.tight_layout()
    if options.path:
        plt.savefig(options.path+"POIcorrelation.png")

    #Plot QOI-QOI correlationa and distributions
    figure, axes=plt.subplots(nrows=model.n_qoi, ncols= model.n_qoi, squeeze=False)
    for i_qoi in range(0,model.n_qoi):
        for j_qoi in range(0,i_qoi+1):
            if i_qoi==j_qoi:
                axes[i_qoi, j_qoi].hist([eval_mat[:,i_qoi]], bins=41)
            else:
                axes[i_qoi, j_qoi].plot(eval_mat[plotPoints,i_qoi], eval_mat[plotPoints,j_qoi],'b*')
            if j_qoi==0:
                axes[i_qoi,j_qoi].set_ylabel(model.name_qoi[i_qoi])
            if i_qoi==model.n_qoi-1:
                axes[i_qoi,j_qoi].set_xlabel(model.name_qoi[j_qoi])
            if model.n_qoi==1:
                axes[i_qoi,j_qoi].set_ylabel('Instances')
    figure.tight_layout()
    if options.path:
        plt.savefig(options.path+"QOIcorrelation.png")

    #Plot POI-QOI correlation
    figure, axes=plt.subplots(nrows=model.n_qoi, ncols= model.n_poi, squeeze=False)
    for i_qoi in range(0,model.n_qoi):
        for jPOI in range(0, model.n_poi):
            axes[i_qoi, jPOI].plot(sample_mat[plotPoints,jPOI], eval_mat[plotPoints,i_qoi],'b*')
            if jPOI==0:
                axes[i_qoi,jPOI].set_ylabel(model.name_qoi[i_qoi])
            if i_qoi==model.n_qoi-1:
                axes[i_qoi,jPOI].set_xlabel(model.name_poi[jPOI])
    if options.path:
        plt.savefig(options.path+"POI_QOIcorrelation.png")
    #Display all figures
    if options.display:
        plt.show()

