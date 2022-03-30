# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 14:03:35 2022

@author: USER
"""

#3rd party Modules
import numpy as np
#import sys
import warnings
import mpi4py.MPI as MPI
from .sampling import parallel_eval


class GsaOptions:
    __slots__=["run", "run_sobol", "run_morris", "n_samp_sobol", "n_samp_morris",\
               "l_morris"]
    def __init__(self, run = True, run_sobol=True, run_morris=True, n_samp_sobol=100000, \
                 n_samp_morris=4, l_morris=3):
        self.run = run
        if self.run == False:
            self.run_sobol = False
            self.run_morris = False
        else:
            self.run_sobol=run_sobol                            #Whether to run Sobol (True or False)
            self.run_morris=run_morris                          #Whether to run Morris (True or False)
        self.n_samp_sobol = n_samp_sobol                      #Number of samples to be generated for GSA
        self.n_samp_morris = n_samp_morris
        self.l_morris=l_morris
        pass

class GsaResults:
    #
    def __init__(self,sobol_base=np.nan, sobol_tot=np.nan, f_a=np.nan, f_b=np.nan, f_d=np.nan, f_ab=np.nan, \
                 samp_d=np.nan, morris_std=np.nan, morris_mean_abs=np.nan, morris_mean=np.nan):
        self.sobol_base=sobol_base
        self.sobol_tot=sobol_tot
        self.f_a=f_a
        self.f_b=f_b
        self.f_d=f_d
        self.f_ab=f_ab
        self.samp_d=samp_d
        self.morris_mean_abs=morris_mean_abs
        self.morris_mean = morris_mean
        self.morris_std=morris_std
    pass



##--------------------------------------GSA-----------------------------------------------------
def run_gsa(model, gsa_options, logging = False):
    """Implements global sensitivity analysis using Morris or Sobol analysis.
    
    Parameters
    ----------
    model : Model
        Contaings simulation information.
    gsa_options : GSAOptions
        Contains run settings
        
    Returns
    -------
    GsaResults 
        Holds all run results
    """
    #GSA implements the following local sensitivity analysis methods on "model" object
        # 1) Gets sampling distribution (used only for internal calculations)
        # 2) Calculates Sobol Indices
        # 3) Performs Morris Screenings (not yet implemented)
        # 4) Produces histogram plots for QOI values (not yet implemented)
    # Required Inputs: Object of class "model" and object of class "options"
    # Outputs: Object of class gsa with fisher and sobol elements
    
    #Load mpi details to keep track of thread number
    mpi_comm = MPI.COMM_WORLD
    mpi_rank = mpi_comm.Get_rank()
    mpi_size = mpi_comm.Get_size()
    
    # Initialize gsa_results in all threads
    gsa_results = GsaResults()
    
    #Morris Screening
    if gsa_options.run_morris:
        #Set non-biased perturbation distance for even l
        #Source: Smith, R. 2011. Uncertainty Quanitification. p.333
        pert_distance = gsa_options.l_morris/ (2*(gsa_options.l_morris-1))
        
        #Create parameter sample only on thread 0 since it need not be parallelized
        # initialize memory location on all threads
        morris_samp = np.zeros((gsa_options.n_samp_morris*(model.n_poi+1), model.n_poi),dtype = float)
        if logging > 1:
            print("initialized morris_samp of size: " + str(morris_samp.shape))
        if mpi_rank == 0:
            if logging:
                print("Generating Morris Sample")
            morris_samp = get_morris_poi_sample(model.sample_fcn, gsa_options.n_samp_morris,\
                                                model.n_poi, pert_distance)
        mpi_comm.Bcast([morris_samp,MPI.DOUBLE], root = 0)
                
            
        morris_mean_abs, morris_mean, morris_std = calculate_morris(\
                                             model.eval_fcn, morris_samp, \
                                             pert_distance, logging = logging)
        gsa_results.morris_mean_abs=morris_mean_abs
        gsa_results.morris_mean = morris_mean
        gsa_results.morris_std=morris_std

    #Sobol Analysis Un parallelized for now
    if gsa_options.run_sobol and mpi_rank == 0:
        if logging:
            print("Generating Sobol Sample")
        #Make Distribution Samples and Calculate model results
        [f_a, f_b, f_ab, f_d, samp_d] = get_sobol_sample(model, gsa_options)
        #Calculate Sobol Indices
        if logging:
            print("Calculating Sobol Sample")
        [sobol_base, sobol_tot]=calculate_sobol(f_a, f_b, f_ab, f_d)
        gsa_results.f_d=f_d
        gsa_results.f_a=f_a
        gsa_results.f_b=f_b
        gsa_results.f_ab=f_ab
        gsa_results.samp_d=samp_d
        gsa_results.sobol_base=sobol_base
        gsa_results.sobol_tot=sobol_tot
        #------------broadcast gsa results to other threads--------------------
        
    return gsa_results


###----------------------------------------------------------------------------------------------
###-------------------------------------Support Functions----------------------------------------
###----------------------------------------------------------------------------------------------


def get_sobol_sample(model,gsa_options):
    """Constructs and evaluates sobol samples using predefined sampling distributions.
        Currently only function for uniform or saltelli normal
    
    Parameters
    ----------
    model : Model
        Contaings simulation information.
    gsa_options : GSAOptions
        Contains run settings
        
    Returns
    -------
    np.ndarray
        n_samp_sobol x n_qoi array of evaluations of Sobol sample part a
    np.ndarray
        n_samp_sobol x n_qoi array of evaluations of Sobol sample part b
    np.ndarray
        n_samp_sobol x n_qoi array x n_poi array of evaluations of mixed Sobol sample ab
    np.ndarray
        2*n_samp_sobol x n_qoi array of concatenated evaluations of part a and b
    np.ndarray
        2*n_samp_sobol x n_poi array of concatenated POI samples of part a and b
    """
    n_samp_sobol = gsa_options.n_samp_sobol
    # Make 2 POI sample matrices with n_samp_sobol samples each
    # if np.all(model.dist_type!=np.array(["satelli normal", "satelli uniform"])):
    #           warnings.warn("Non-satelli sampling algorithm used for Sobol analysis."\
    #                         + " Suggested distribution types are satelli normal "+\
    #                             "and satelli uniform.")
    sample_compact = model.sample_fcn(2*n_samp_sobol)
    f_compact = model.eval_fcn(sample_compact)
    # Seperate sample into a and b for algorithm
    samp_a = sample_compact[:n_samp_sobol]
    samp_b = sample_compact[n_samp_sobol:]
    f_a = f_compact[:n_samp_sobol]
    f_b = f_compact[n_samp_sobol:] # n_samp_sobol x nQOI out matrix from B
    # Stack the output matrices into a single matrix
    f_d = np.concatenate((f_a.copy(), f_b.copy()), axis=0)

    # Initialize combined QOI sample matrices
    if model.n_qoi == 1:
        f_ab = np.empty([n_samp_sobol, model.n_poi])
    else:
        f_ab = np.empty([n_samp_sobol, model.n_poi, model.n_qoi])
    for i_param in range(0, model.n_poi):
        # Define sampC to be A with the ith parameter in B
        samp_ab = samp_a.copy()
        samp_ab[:, i_param] = samp_b[:, i_param].copy()
        if model.n_qoi == 1:
            f_ab[:, i_param] = model.eval_fcn(samp_ab).squeeze()
        else:
            f_ab[:, i_param, :] = model.eval_fcn(samp_ab)  # n_samp_sobol x nPOI x nQOI tensor
        del samp_ab
    return f_a, f_b, f_ab, f_d, sample_compact

def calculate_sobol(f_a, f_b, f_ab, f_d):
    """Calculates 1st order and total sobol indices using Saltelli approximation formula.
    
    Parameters
    ----------
    f_a : np.ndarray
        n_samp_sobol x n_qoi array of evaluations of Sobol sample part a
    f_b : np.ndarray
        n_samp_sobol x n_qoi array of evaluations of Sobol sample part b
    f_ab : np.ndarray
        n_samp_sobol x n_qoi array x n_poi array of evaluations of mixed Sobol sample ab
    f_d : np.ndarray
        2*n_samp_sobol x n_qoi array of concatenated evaluations of part a and b
    
        
    Returns
    -------
    np.ndarray
        n_qoi x n_poi array of 1st order Sobol indices
    np.ndarray
        n_qoi x n_poi array of total Sobol indices
    """
    #Calculates calculates sobol indices using satelli approximation method
    #Inputs: model object (with eval_fcn, sample, and nParams)
    #        sobolOptions object
    #Determing number of samples, QOIs, and POIs based on inputs
    if f_ab.ndim==1:
        n_qoi=1
        n_poi=1
    elif f_ab.ndim==2:
        n_qoi=1
        n_poi=f_ab.shape[1]
    elif f_ab.ndim==3:
        n_poi=f_ab.shape[1]
        n_qoi=f_ab.shape[2]
    else:
        raise(Exception('f_ab has greater than 3 dimensions, make sure f_ab is' \
                        'the squeezed form of n_samp_sobol x nPOI x nQOI'))
    #QOI variance
    fDvar=np.var(f_d, axis=0)

    sobol_base=np.empty((n_qoi, n_poi))
    sobol_tot=np.empty((n_qoi, n_poi))
    if n_qoi==1:
        #Calculate 1st order parameter effects
        sobol_base=np.mean(f_b*(f_ab-f_a), axis=0)/(fDvar)

        #Caclulate 2nd order parameter effects
        sobol_tot=np.mean((f_a-f_ab)**2, axis=0)/(2*fDvar)

    else:
        for iQOI in range(0,n_qoi):
            #Calculate 1st order parameter effects
            sobol_base[iQOI,:]=np.mean(f_b[:,[iQOI]]*(f_ab[:,:,iQOI]-f_a[:,[iQOI]]),axis=0)/fDvar[iQOI]
            #Caclulate 2nd order parameter effects
            sobol_tot[iQOI,:]= np.mean((f_a[:,[iQOI]]-f_ab[:,:,iQOI])**2,axis=0)/(2*fDvar[iQOI])


    return sobol_base, sobol_tot
#==============================================================================
#----------------------------------Morris Sampling-----------------------------
#==============================================================================


##--------------------------------calculate_morris-----------------------------
def calculate_morris(eval_fcn, morris_samp, pert_distance, logging = False):
    """Calculates morris samples using information from Model and GsaOptions objects.
    
    Parameters
    ----------
    model : Model
        Contaings simulation information.
    gsa_options : GSAOptions
        Contains run settings
        
    Returns
    -------
    np.ndarray
        n_qoi x n_poi array of morris sensitivity mean indices
    np.ndarray
        n_qoi x n_poi array of morris sensitivity variance indices
    """
    #Evaluate Sample
    #Load mpi details to keep track of thread number
    mpi_comm = MPI.COMM_WORLD
    mpi_rank = mpi_comm.Get_rank()
    mpi_size = mpi_comm.Get_size()
    if logging and mpi_rank == 0:
        print("Evaulating Morris Sample")
        
    if mpi_size == 1:
        f_eval_compact = eval_fcn(morris_samp)
    else:
        f_eval_compact = parallel_eval(eval_fcn, morris_samp, logging = logging)
        
    #Make sure all threads finish collecting f_eval_compact before continuing
    mpi_comm.Barrier()
    if logging > 1 and mpi_rank == 0:
        print("f_eval_compact: " + str(f_eval_compact))
        
    # Initialize Morris indices so that the memory is reserved when broadcasting
    if f_eval_compact.ndim == 2:
        morris_mean_abs = np.zeros((morris_samp.shape[1], f_eval_compact.shape[1]), dtype = float) # n_poi x n_qoi
        morris_mean = np.zeros(morris_mean_abs.shape, dtype = float)
        morris_std = np.zeros(morris_mean_abs.shape, dtype = float) # n_poi x n_qoi
    else :
        morris_mean_abs = np.zeros((morris_samp.shape[1]), dtype = float) # n_poi x n_qoi
        morris_mean = np.zeros(morris_mean_abs.shape, dtype = float)
        morris_std = np.zeros(morris_mean_abs.shape, dtype = float) # n_poi x n_qoi
    
    
    # Perform morris calculation only on base thread
    if mpi_rank == 0:
        #Compute # of pois, qois and samples to ensure consitency
        if morris_samp.ndim == 1:
            n_poi = 1
        elif morris_samp.ndim == 2:
            n_poi = morris_samp.shape[1]
        else:
            raise Exception("More than 2 dimensions in morris_samp")
        #Convert to int so it can be used in indexing
        n_samp = int(morris_samp.shape[0]/(n_poi+1))
        if f_eval_compact.ndim == 2:
            n_qoi = f_eval_compact.shape[1]
        elif f_eval_compact.ndim ==1:
            n_qoi = 1
        else:
            raise Exception("More than 2 dimensions in f_eval")
            
        #Uncompact Samples
        f_eval_seperated = morris_seperate(f_eval_compact, n_samp, n_poi, n_qoi)
        morris_samp_seperated = morris_seperate(morris_samp, n_samp, n_poi, n_poi)
        if logging > 1:
            print("morris samp seperated: " + str(morris_samp_seperated))
        
        #Get which sample perturbs which poi
        poi_pert_location = get_poi_pert_location(morris_samp_seperated)
        if logging > 1:
            print("poi_pert_location: " + str(poi_pert_location))
            
            
        #initialize data storage arrays with 1 dimension lower if n_qoi =1
        if n_qoi > 1:
            deriv_approx = np.empty((n_samp, n_poi, n_qoi))  # n_samp x n_poi x n_qoi
        else: 
            deriv_approx = np.empty((n_samp, n_poi))  # n_samp x n_poi
        if logging >1:
            print("QOIs : " + str(f_eval_seperated))
        
        #Apply finite difference formula 
        #Source: Smith, R. 2011, Uncertainty Quanitification. p.333
        if logging > 0:
            print("Calculating Morris indices")
        for i_samp in range(n_samp):
            for i_pert in range(n_poi): 
                i_poi = poi_pert_location[i_samp, i_pert]
                deriv_approx[i_samp,i_poi] = (f_eval_seperated[i_samp,i_pert+1] - \
                                              f_eval_seperated[i_samp,i_pert])/ pert_distance
        # for i_poi in range(n_poi):
        #     deriv_approx[:,i_poi] = f_eval_seperated[:,i_poi+1] - f_eval_seperated[:,i_poi]
        if logging > 1:
            print("deriv approx: " + str(deriv_approx))
            
        #Apply Morris Index formulas
        #Source: Smith, R. 2011, Uncertainty Quanitification. p.332
        morris_mean_abs = np.mean(np.abs(deriv_approx),axis = 0) # n_poi x n_qoi
        morris_mean = np.mean(deriv_approx, axis = 0)
        morris_std=np.sqrt(np.var(deriv_approx, axis=0)) # n_poi x n_qoi
        
        if logging > 1:
            print("morris mean abs: " + str(morris_mean_abs))
            print("morris mean abs: " + str(morris_mean))
            print("morris st: " + str(morris_std))
        if logging  and mpi_size > 1: 
            print("Broadcasting Morris Indices")
    #Send out finished morris indices to all threads
    mpi_comm.Bcast([morris_mean_abs, MPI.DOUBLE], root = 0)
    mpi_comm.Bcast([morris_mean, MPI.DOUBLE], root = 0)
    mpi_comm.Bcast([morris_std, MPI.DOUBLE], root = 0)
    if logging:
        print("Finished broadcasting (thread " + str(mpi_rank) + ")")

    return morris_mean_abs, morris_mean, morris_std

def morris_seperate(qoi_compact, n_samp, n_poi, n_qoi):
    if n_qoi > 1:
        qoi_seperated = np.empty((n_samp, n_poi+1, n_qoi))
    else: 
        qoi_seperated = np.empty((n_samp, n_poi+1))
    #Seperate each parameter search for ease of computation
    for i_samp in range(n_samp):
        qoi_seperated[i_samp] = qoi_compact[i_samp*(n_poi+1):(i_samp+1)*(n_poi+1)].squeeze()
        
    return qoi_seperated

def get_poi_pert_location(morris_samp_seperate):
    n_samp = morris_samp_seperate.shape[0]
    n_poi = morris_samp_seperate.shape[2]
    poi_pert_location = np.empty((n_samp, n_poi))
    for i_samp in range(n_samp):
        for i_poi in range(n_poi):
            poi_pert_location[i_samp,i_poi] = np.argmax(np.abs(morris_samp_seperate[i_samp,i_poi+1] \
                                          - morris_samp_seperate[i_samp,i_poi]))
    return poi_pert_location.astype(int)
##---------------------------get_morris_poi_sample-----------------------------

def get_morris_poi_sample(param_dist, n_samp, n_poi, pert_distance, random = False):
    #Use sobol distributions for low discrepancy
    #Generate n_samp_morris samples
    random_samp =  param_dist(n_samp)
    #Define Sampling matrices that are constant
    J=np.ones((n_poi+1,n_poi))
    B = (np.tril(np.ones(J.shape), -1))
    morris_samp_compact = np.empty((n_samp*(n_poi+1), n_poi))
    for i_samp in range(n_samp):
        jTheta=random_samp[i_samp,]*J
        #Calculate Morris Sample matrix
        #Source: Smith, R. 2011. Uncertainty Quantification. p.334
        if random == True:  
            #Define Random Sampling matrices
            #D=np.diag(np.random.choice(np.array([1,-1]), size=(n_poi,)))
            #NOTE: using non-random step direction to keep denominator in deriv approx
            #   equal to delta rather than -delta for some samples. Random form is
            #   kept above in comments 
            D=np.diag(np.random.choice(np.array([1,1]), size=(n_poi,)))
            P=np.identity(n_poi)
            np.random.shuffle(P)
            samp_mat = np.matmul(jTheta+pert_distance/2*(np.matmul((2*B-J),D)+J),P)
        elif random == False:
            #Define non-random Sampling matrices
            D=np.diag(np.random.choice(np.array([1,1]), size=(n_poi,)))
            P=np.identity(n_poi)
            np.random.shuffle(P)
            # Only use non-random formulations for testing matrix generation
            samp_mat = jTheta+pert_distance/2*(np.matmul((2*B-J),D)+J)
        #Stack each grid seach so that a single eval_fcn call is required
        morris_samp_compact[i_samp*(n_poi+1):(i_samp+1)*(n_poi+1),:] = samp_mat
    return morris_samp_compact


            

