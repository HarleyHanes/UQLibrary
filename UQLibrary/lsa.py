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

class LsaOptions:
    def __init__(self,run=True, run_param_subset=True, x_delta=10**(-12),\
                 method='complex', scale='y', subset_rel_tol=.001):
        self.run=run                              #Whether to run lsa (True or False)
        self.x_delta=x_delta                        #Input perturbation for calculating jacobian
        self.scale=scale                          #scale can be y, n, or both for outputing scaled, unscaled, or both
        self.method=method                        #method used for approximating derivatives
        if self.run == False:
            self.run_param_subset = False
        else:
            self.run_param_subset=run_param_subset
        self.subset_rel_tol=subset_rel_tol
        if not self.scale.lower() in ('y','n','both'):
            raise Exception('Error! Unrecgonized scaling output, please enter y, n, or both')
        if not self.method.lower() in ('complex','finite'):
            raise Exception('Error! unrecognized derivative approx method. Use complex or finite')
        if self.x_delta<0 or not isinstance(self.x_delta,float):
            raise Exception('Error! Non-compatibale x_delta, please use a positive floating point number')
        if self.subset_rel_tol<0 or self.subset_rel_tol>1 or not isinstance(self.x_delta,float):
            raise Exception('Error! Non-compatibale x_delta, please use a positive floating point number less than 1')
    pass


##--------------------------------------LSA-----------------------------------------------------
# Local Sensitivity Analysis main
class LsaResults:
    def __init__(self,jacobian=np.empty, rsi=np.empty, fisher=np.empty, reduced_model=np.empty, active_set="", inactive_set=""):
        self.jac=jacobian
        self.rsi=rsi
        self.fisher=fisher
        self.reduced_model=reduced_model
        self.active_set=active_set
        self.inactive_set=inactive_set
    pass


def run_lsa(model, lsa_options):
    """Implements local sensitivity analysis using LSI, RSI, and parameter subset reduction.
    
    Parameters
    ----------
    model : Model
        Object of class Model holding run information.
    options : Options
        Object of class Options holding run settings.
        
    Returns
    -------
    LsaResults 
        Object of class LsaResults holding all run results.
    """
    # LSA implements the following local sensitivity analysis methods on system specified by "model" object
        # 1) Jacobian
        # 2) Scaled Jacobian for Relative Sensitivity Index (RSI)
        # 3) Fisher Information matrix
    # Required Inputs: object of class "model" and object of class "options"
    # Outputs: Object of class lsa with Jacobian, RSI, and Fisher information matrix

    # Calculate Jacobian
    jac_raw=get_jacobian(model.eval_fcn, model.base_poi, lsa_options.x_delta,\
                         lsa_options.method, scale=False, y_base=model.base_qoi)
    # Calculate relative sensitivity index (RSI)
    jac_rsi=get_jacobian(model.eval_fcn, model.base_poi, lsa_options.x_delta,\
                         lsa_options.method, scale=True, y_base=model.base_qoi)
    # Calculate Fisher Information Matrix from jacobian
    fisher_mat=np.dot(np.transpose(jac_raw), jac_raw)

    #Active Subspace Analysis
    if lsa_options.run_param_subset:
        reduced_model, active_set, inactive_set = get_active_subset(model, lsa_options)
        #Collect Outputs and return as an lsa object
        return LsaResults(jacobian=jac_raw, rsi=jac_rsi, fisher=fisher_mat,\
                          reduced_model=reduced_model, active_set=active_set,\
                          inactive_set=inactive_set)
    else:
        return LsaResults(jacobian=jac_raw, rsi=jac_rsi, fisher=fisher_mat)
    
###----------------------------------------------------------------------------------------------
###-------------------------------------Support Functions----------------------------------------
###----------------------------------------------------------------------------------------------

  
    
  
##--------------------------------------GetJacobian-----------------------------------------------------
def get_jacobian(eval_fcn, x_base, x_delta, method, **kwargs):
    """Calculates scaled or unscaled jacobian using different derivative approximation methods.
    
    Parameters
    ----------
    eval_fcn : 
        Holds run information.
    x_base : np.ndarray
        POI values at which to calculate Jacobian
    lsa_options : Lsa_Options
        Holds run options
    **scale : bool
        Whether or not to apply relative scaling of POI and QOI
    **y_base : np.ndarray
        QOI values used in finite difference approximation, saves a function evaluation
        
    Returns
    -------
    np.ndarray 
        Scaled or unscaled jacobian
    """
    if 'scale' in kwargs:                                                   # Determine whether to scale derivatives
                                                                            #   (for use in relative sensitivity indices)
        scale = kwargs["scale"]
        if not isinstance(scale, bool):                                     # Check scale value is boolean
            raise Exception("Non-boolean value provided for 'scale' ")      # Stop compiling if not
    else:
        scale = False                                                       # Function defaults to no scaling
    if 'y_base' in kwargs:
        y_base = eval_fcn(x_base)
        #y_base = kwargs["y_base"]
        # Make sure x_base is int/ float and convert to numpy array
        if type(y_base)==int or type(y_base)==float:
            y_base = np.array([y_base])
        elif type(x_base)== list:
            y_list = y_base
            y_base = np.empty(len(y_list))
            for i_poi in len(y_list):
                if type(y_list[i_poi])==int or type(y_list[i_poi])==float:
                   y_base[i_poi] = y_list[i_poi]
                else:
                    raise Exception(str(i_poi) + "th y_base value is of type:  " + str(type(y_list[i_poi])))
        elif type(y_base)!= np.ndarray:
            raise Exception("y_base of type " + str(type(y_base)) + ". Accepted" \
                            " types are int/ float and list or numpy arrays of ints/ floats")
    else:
        y_base = eval_fcn(x_base)

    # Make sure x_base is int/ float and convert to numpy array
    if type(x_base)==int or type(x_base)==float:
        x_base = np.array([x_base])
    elif type(x_base)== list:
        x_list = x_base
        x_base = np.empty(len(x_list))
        for i_poi in len(x_list):
            if type(x_list[i_poi])==int or type(x_list[i_poi])==float:
               x_base[i_poi] = x_list[i_poi]
            else:
                raise Exception(str(i_poi) + "th x_base value is of type:  " + str(type(x_list[i_poi])))
    elif type(x_base)!= np.ndarray:
        raise Exception("x_base of type " + str(type(x_base)) + ". Accepted" \
                        " types are int/ float and list or numpy arrays of ints/ floats")

    #Initialize base QOI value, the number of POIs, and number of QOIs
    n_poi = np.size(x_base)
    n_qoi = np.size(y_base)

    jac = np.empty(shape=(n_qoi, n_poi), dtype=float)                       # Define Empty Jacobian Matrix

    for i_poi in range(0, n_poi):                                            # Loop through POIs
        # Isolate Parameters
        if method.lower()== 'complex':
            xPert = x_base + np.zeros(shape=x_base.shape)*1j                  # Initialize Complex Perturbed input value
            xPert[i_poi] += x_delta * 1j                                      # Add complex Step in input
        elif method.lower() == 'finite':
            xPert = x_base.copy()
            xPert[i_poi] += x_delta
        yPert = eval_fcn(xPert)                                        # Calculate perturbed output
        for i_qoi in range(0, n_qoi):                                        # Loop through QOIs
            if method.lower()== 'complex':
                jac[i_qoi, i_poi] = np.imag(yPert[i_qoi] / x_delta)                 # Estimate Derivative w/ 2nd order complex
            elif method.lower() == 'finite':
                jac[i_qoi, i_poi] = (yPert[i_qoi]-y_base[i_qoi]) / x_delta
            #Only Scale Jacobian if 'scale' value is passed True in function call
            if scale:
                jac[i_qoi, i_poi] *= x_base[i_poi] * np.sign(y_base[i_qoi]) / (sys.float_info.epsilon + y_base[i_qoi])
                                                                            # Scale jacobian for relative sensitivity
        del xPert, yPert, i_poi, i_qoi                                        # Clear intermediate variables
    return jac                                                              # Return Jacobian




##--------------------------------------------Parameter dimension reduction------------------------------------------------------
def get_active_subset(model,lsa_options):
    """Calculates active and inactive parameter subsets.
        --Not fully function, reduced model is still full model
    
    Parameters
    ----------
    model : Model
        Holds run information.
    lsa_options : Lsa_Options
        Holds run options
        
    Returns
    -------
    Model 
        New model using reduced parameters
    np.ndarray
        Data type string of active parameters
    np.ndarray
        Data type string of inactive parameters
    """
    eliminate=True
    inactive_index=np.zeros(model.n_poi)
    #Calculate Jacobian
    jac=get_jacobian(model.eval_fcn, model.base_poi, lsa_options.x_delta,\
                         lsa_options.method, scale=False, y_base=model.base_qoi)
    while eliminate:
        #Caclulate Fisher
        fisher_mat=np.dot(np.transpose(jac), jac)
        #Perform Eigendecomp
        eigen_values, eigen_vectors =np.linalg.eig(fisher_mat)
        #Eliminate dimension/ terminate
        if np.min(eigen_values) < lsa_options.subset_rel_tol * np.max(eigen_values):
            #Get inactive parameter
            inactive_param_reduced_index=np.argmax(np.absolute(eigen_vectors[:, np.argmin(np.absolute(eigen_values))]))
            inactive_param=inactive_param_reduced_index+np.sum(inactive_index[0:(inactive_param_reduced_index+1)]).astype(int)
                #This indexing may seem odd but its because we're keeping the full model parameter numbering while trying
                # to index within the reduced model so we have to add to the index the previously removed params
            #Record inactive param in inactive space
            inactive_index[inactive_param]=1
            #Remove inactive elements of jacobian
            jac=np.delete(jac,inactive_param_reduced_index,1)
        else:
            #Terminate Active Subspace if singular values within tolerance
            eliminate=False
            
    #Define active and inactive spaces
    active_set=model.name_poi[inactive_index == False]
    inactive_set=model.name_poi[inactive_index == True]
    
    reduced_model = model_reduction(model, inactive_param)
    # reduced_model.base_poi=reduced_model.base_poi[inactive_index == False]
    # reduced_model.name_poi=reduced_model.name_poi[inactive_index == False]
    # reduced_model.eval_fcn = lambda reduced_poi: model.eval_fcn(
    #     np.array([x for x, y in zip(reduced_poi,model.base_poi) if inactive_index== True]))
    # #reduced_model.eval_fcn=lambda reduced_poi: model.eval_fcn(np.where(inactive_index==False, reduced_poi, model.base_poi))
    # reduced_model.base_qoi=reduced_model.eval_fcn(reduced_model.base_poi)
    return reduced_model, active_set, inactive_set

def model_reduction(model,inactive_param):
    """Computes a new Model object using only active parameter set"
        -Not fully function, reduced model is still full model
        
    Parameters
    ----------
    model : Model
        Original run information
    inactive_param : Lsa_Options
        Holds run options
    
        
    Returns
    -------
    Model 
        New model using reduced parameters
    """
    reduced_model = model
    # #Record Index of reduced param
    # inactive_index=np.where(reduced_model.name_poi==inactive_param)[0]
    # #confirm exactly parameter matches
    # if len(inactive_index)!=1:
    #     raise Exception("More than one or no POIs were found matching that name.")
    # #Remove relevant data elements
    # reduced_model.base_poi=np.delete(reduced_model.base_poi, inactive_index)
    # reduced_model.name_poi=np.delete(reduced_model.name_poi, inactive_index)
    # reduced_model.eval_fcn=lambda reduced_poi: model.eval_fcn(np.where(inactive_index==True,reduced_poi,model.base_poi))
    # print('made eval_fcn')
    # print(reduced_model.eval_fcn(reduced_model.base_poi))
    return reduced_model

def get_reduced_pois(reduced_poi,dropped_indices,model):
    """Maps from the space of reduced pois to the full poi set.
        
    Parameters
    ----------
    reduced_poi : np.ndarray
        Set of reduced parameter set values
    dropped_indices : np.ndarray
        Indices of dropped parameters
    model : Model
        Original run information
    
        
    Returns
    -------
    np.ndarray 
        New model using reduced parameters
    """
    #Load in full parameter set to start output
    full_poi=model.base_poi
    #Use a counter to keep track of the index in full_pois based on index in
    # reduced_pois
    reduced_counter=0
    for i_poi in np.arange(0,model.n_poi):
        if dropped_indices==i_poi:
            full_poi[i_poi]=reduced_poi[reduced_counter]
            reduced_counter=reduced_counter+1
            
            
    return full_poi