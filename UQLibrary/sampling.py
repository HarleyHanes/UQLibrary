# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 19:26:38 2022

@author: USER
"""
import numpy as np
from scipy.stats import qmc
import scipy.stats as sct

import mpi4py.MPI as MPI

#======================================================================================================
#-----------------------------------Sampling-----------------------------------------------------------
#======================================================================================================
       

##--------------------------------------GetSampDist----------------------------------------------------
def get_samp_dist(dist_type, dist_param, n_poi, fcn_inverse_cdf = np.nan):
    """Adds sampling function sample to model for drawing of low-discrepency
        from given distribution type.
    
    Parameters
    ----------
    model : Model
        Contaings simulation information.
    gsa_options : GSAOptions
        Contains run settings
        
    Returns
    -------
    Model
        model object with added sample function
    """
    # Determine Sample Function- Currently only 1 distribution type can be defined for all parameters
    if dist_type == 'normal':  # Normal Distribution
        sample_fcn = lambda n_samp_sobol: np.random.randn(n_samp_sobol, n_poi)*\
            np.sqrt(dist_param[[1], :]) + dist_param[[0], :]
    elif dist_type == 'saltelli normal':
        sample_fcn = lambda n_samp_sobol: saltelli_normal(n_samp_sobol, dist_param)
    elif dist_type == 'uniform':  # uniform distribution
        # doubleParms=np.concatenate(model.dist_param, model.dist_param, axis=1)
        sample_fcn = lambda n_samp_sobol: np.random.rand(n_samp_sobol, n_poi)*\
            (dist_param[[1], :]-dist_param[[0],:]) + dist_param[[0], :]
    elif dist_type == 'saltelli uniform':  # uniform distribution
        # doubleParms=np.concatenate(model.dist_param, model.dist_param, axis=1)
        sample_fcn = lambda n_samp_sobol: saltelli_uniform(n_samp_sobol, dist_param)
    elif dist_type == 'exponential': # exponential distribution
        sample_fcn = lambda n_samp_sobol: np.random.exponential(dist_param,size=(n_samp_sobol, n_poi))
    elif dist_type == 'beta': # beta distribution
        sample_fcn = lambda n_samp_sobol:np.random.beta(dist_param[[0],:], dist_param[[1],:],\
                                               size=(n_samp_sobol, n_poi))
    elif dist_type == 'InverseCDF': #Arbitrary distribution given by inverse cdf
        if fcn_inverse_cdf == np.nan:
            raise Exception("InverseCDF distribution selected but no function provided.")
        sample_fcn = lambda n_samp_sobol: fcn_inverse_cdf(np.random.rand(n_samp_sobol, n_poi))
    else:
        raise Exception("Invalid value for model.dist_type. Supported distributions are normal, uniform, exponential, beta, \
        and InverseCDF")  # Raise Exception if invalide distribution is entered
    
    return sample_fcn



def saltelli_sample(n_samp, n_poi):
    """Constructs a uniform [0,1] low discrepency saltelli sample for use in Sobol
        index approximation
    
    Parameters
    ----------
    n_samp_sobol : int
        Number of samples to take
    dist_param : np.ndarray
        2 x n_poi array of min and max sampling bounds for each parameter
        
    Returns
    -------
    np.ndarray
        Low discrepancy POI sample of uniform distribution on [0,1] constructed 
        using satelli's alrogrithm
    """
    
    #Add .5 to n_samp/2 so that if n_samp is odd, an extra sample is generated
    sampler = qmc.Sobol(d= n_poi*2, scramble = True)
    #Use the smallest log2 sample size at least as large as n_samp to keep
    #   quadrature balance 
    #   (see https://scipy.github.io/devdocs/reference/generated/scipy.stats.qmc.Sobol.html )
    base_sample = sampler.random_base2(m=int(np.ceil(np.log2(n_samp/2))))
    
    #Add .5 to n_samp/2 so that if n_samp is odd, an extra sample is generated
    base_sample=base_sample[:int(n_samp/2+.5),:]
    
    sample = np.empty((n_samp, n_poi))
    
    #Seperate and stack half the samples in the 2nd dimension for saltelli's 
    # algorithm
    if n_samp%2==0:
        sample[:int((n_samp)/2),:]=base_sample[:,0:n_poi]
        sample[int((n_samp)/2):,:]=base_sample[:,n_poi:]
    else :
        sample[:int((n_samp+.5)/2),:] = base_sample[:,0:n_poi]
        sample[int((n_samp+.5)/2):-1,:] = base_sample[:,n_poi:]
    return sample


def saltelli_uniform(n_samp, dist_param):
    """Constructs a uniform low discrepency saltelli sample for use in Sobol
        index approximation
    
    Parameters
    ----------
    n_samp: int
        Number of samples to take
    dist_param : np.ndarray
        2 x n_poi array of mean and variance for each parameter
        
    Returns
    -------
    np.ndarray
        Low discrepancy POI sample of uniform distribution constructed using 
        satelli's alrogrithm
    """
    n_poi=dist_param.shape[1]
    
    sample_base = saltelli_sample(n_samp,n_poi)
    
    sample_transformed = dist_param[[0],:]+(dist_param[[1],:]-dist_param[[0],:])*sample_base
    return sample_transformed


def saltelli_normal(n_samp, dist_param):
    """Constructs a normal low discrepency saltelli sample for use in Sobol
        index approximation
    
    Parameters
    ----------
    n_samp: int
        Number of samples to take
    dist_param : np.ndarray
        2 x n_poi array of mean and variance for each parameter
        
    Returns
    -------
    np.ndarray
        Low discrepancy POI sample of normal distribution constructed using 
        satelli's alrogrithm
    """
    
    n_poi=dist_param.shape[1]
    
    sample_base = saltelli_sample(n_samp,n_poi)
    sample_transform=sct.norm.ppf(sample_base)*np.sqrt(dist_param[[1], :]) \
        + dist_param[[0], :]
    return sample_transform


#======================================================================================================
#----------------------------Parallelization Support---------------------------------------------------
#======================================================================================================

def parallel_eval(eval_fcn, poi_sample, logging = False):
    """ Seperates samples and parallelizes model computations

    Parameters
    ----------
    eval_fcn : function
        User defined function that maps POIs to QOIs
    poi_sample : np.ndarray
        n_samp x n_poi array of POI samples

    Returns
    -------
    qoi_samp : np.ndarray
        n_samp x n_qoi array of QOIs from each POI sample
    """
    mpi_comm = MPI.COMM_WORLD
    mpi_rank = mpi_comm.Get_rank()
    mpi_size = mpi_comm.Get_size()
     
    # Seperate poi samples into subsample for each thread
    if mpi_rank == 0:
        if logging > 1:
            print("poi_sample in thread " + str(mpi_rank) + ": " + str(poi_sample))
        for i_rank in range(mpi_size):
            if mpi_rank == 0:
                samp_per_subsample = int(np.round(poi_sample.shape[0]/mpi_size))
                if i_rank == 0:
                    data = poi_sample[0:samp_per_subsample]
                else: 
                    if i_rank == mpi_size-1:
                        data_broadcast = poi_sample[(i_rank*samp_per_subsample):]
                    else:
                        data_broadcast = poi_sample[(i_rank*samp_per_subsample):((i_rank+1)*samp_per_subsample)]
                    mpi_comm.send(data_broadcast.shape, dest = i_rank, tag = 0)
                    mpi_comm.Send([data_broadcast,MPI.DOUBLE],dest = i_rank, tag = 1)
                    #print("poi_subsample sent to thread " + str(i_rank) + ": " + str(data_broadcast))
    else:
        data_shape = mpi_comm.recv(source = 0, tag = 0)
        data = np.empty(data_shape)
        mpi_comm.Recv(data,source=0, tag=1)
                
    
    # Evaluate each subsamples
    qoi_subsample = eval_fcn(data)
    mpi_comm.Barrier()
    if qoi_subsample.ndim == 2:
        qoi_sample = np.zeros((poi_sample.shape[0], qoi_subsample.shape[1]), dtype = float)
    else:
        qoi_sample = np.zeros((poi_sample.shape[0]), dtype = float)
    #print(poi_reconstructed)

    if mpi_rank > 0:
        mpi_comm.send(qoi_subsample.shape, dest = 0, tag = 0)
        mpi_comm.Send([qoi_subsample, MPI.DOUBLE], dest = 0, tag =1)
        #print("sending data from thread " + str(mpi_rank) + ": " + str(data))
    elif mpi_rank ==0 :
        total_samp=0
        for i_rank in range(mpi_size):
            if i_rank > 0:
                subsample_shape = mpi_comm.recv(source = i_rank, tag = 0)
                #print("receiving data from thread " + str(i_rank) + " of shape: " + str(data_shape))
            else :
                subsample_shape = qoi_subsample.shape
            n_samp = subsample_shape[0]
            if i_rank > 0:
                #print("poi_reconstructed before receiving: " + str(poi_reconstructed))
                mpi_comm.Recv(qoi_sample[total_samp:(total_samp+n_samp)], source = i_rank, tag=1)
            else :
                qoi_sample[total_samp:(total_samp+n_samp)] = qoi_subsample
            if logging > 1:
                print("qoi_reconstructed after receiving thread " + str(i_rank) + ": " + str(qoi_sample))
            total_samp += n_samp 
            
    # Send back out qoi_sample so all threads have a return
    mpi_comm.Bcast([qoi_sample, MPI.DOUBLE], root = 0)
    
    
    return qoi_sample
