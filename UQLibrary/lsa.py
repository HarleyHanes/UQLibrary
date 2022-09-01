# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 14:03:22 2022

@author: USER
"""
#3rd party Modules
import numpy as np
import sys
import scipy as scp
import warnings
#import matplotlib.pyplot as plt
#import scipy.integrate as integrate
#from tabulate import tabulate                       #Used for printing tables to terminal
#import sobol                                        #Used for generating sobol sequences
#import SALib.sample as sample
import scipy.stats as sct
from .sampling import parallel_eval
import mpi4py.MPI as MPI
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

class LsaOptions:
    __slots__=["run", "run_lsa", "run_pss", "x_delta", "deriv_method", "scale",\
               "pss_algorithm", "pss_rel_tol", "pss_decomp_method"]
    def __init__(self,run=True, run_lsa = True, run_pss=True, x_delta=10**(-12),\
                 deriv_method='complex', scale='y', pss_algorithm = "rrqr",\
                 pss_rel_tol=1e-8, pss_decomp_method = "svd"):
        #----------------------------Run Selections----------------------------
        self.run=run                              #Whether to run lsa (True or False)
        if self.run == False:
            self.run_pss = False
            self.run_lsa= False
        else:
            self.run_pss=run_pss
            self.run_lsa = run_lsa
        #-------------------------Derivative Calculations----------------------
        self.x_delta=x_delta                      #Input perturbation for calculating jacobian
        self.deriv_method=deriv_method                        #method used for approximating derivatives
        if not self.deriv_method.lower() in ('complex','finite'):
            raise Exception('Error! unrecognized derivative approx method. Use complex or finite')
        if self.x_delta<0 or not isinstance(self.x_delta,float):
            raise Exception('Error! Non-compatibale x_delta, please use a positive floating point number')
        #-----------------------Parameter Subset Selection---------------------
        #pss_rel_tol
        if type(pss_rel_tol)!= float:
            warnings.warn("Invalid data type for pss_rel_tol: " + str(type(pss_rel_tol)) \
                          +". Defaulting to 1e-8")
            self.pss_rel_tol = 1e-8
        elif pss_rel_tol<0 or pss_rel_tol>1 :
            warnings.warn("Invalid pss_rel_tol: " + str(pss_rel_tol)+". Defaulting to 1e-8")
            self.pss_rel_tol = 1e-8
        else :
            self.pss_rel_tol=pss_rel_tol
            
        #pss_decomp_method
        if type(pss_decomp_method)!= str:
            warnings.warn("Non-string value for pss_decomp_method detected. Defaulting to SVD.")
            self.pss_decomp_method = "svd"
        elif pss_decomp_method.lower() != "svd" and pss_decomp_method.lower() != "eigen":
            warnings.warn("Unknown pss_decomp_method: "+str(pss_decomp_method)+". Defaulting to SVD.")
            self.pss_decomp_method = "svd"
        else:
            self.pss_decomp_method = pss_decomp_method
        
        #pss_algorithm
        if type(pss_algorithm)!= str:
            warnings.warn("Non-string value for pss_algorithm detected. Defaulting to RRQR.")
            self.pss_algorithm = "rrqr"
        elif pss_algorithm.lower() != "rrqr" and pss_algorithm.lower() != "rrqr":
            warnings.warn("Unknown pss_algorithm: "+str(pss_algorithm)+". Defaulting to RRQR.")
            self.pss_algorithm = "rrqr"
        else:
            self.pss_algorithm = pss_algorithm
            
    pass


##--------------------------------------LSA-----------------------------------------------------
# Local Sensitivity Analysis main
class LsaResults:
    def __init__(self,jacobian=np.empty, rsi=np.empty, fisher=np.empty,
                 active_set="", inactive_set="", ident_values = np.empty,\
                 ident_vectors=np.empty):
        self.jac=jacobian
        self.rsi=rsi
        self.fisher=fisher
        self.active_set=active_set
        self.inactive_set=inactive_set
        self.ident_values = ident_values
        self.ident_vectors = ident_vectors
    pass


def run_lsa(model, lsa_options, logging = 0):
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
    if lsa_options.run_lsa:
        jac_raw=get_jacobian(model.eval_fcn, model.base_poi, lsa_options.x_delta,\
                             lsa_options.deriv_method, scale=False, y_base=model.base_qoi)
        # Calculate relative sensitivity index (RSI)
        jac_rsi=get_jacobian(model.eval_fcn, model.base_poi, lsa_options.x_delta,\
                             lsa_options.deriv_method, scale=True, y_base=model.base_qoi)
        # Calculate Fisher Information Matrix from jacobian
        fisher_mat=np.dot(np.transpose(jac_raw), jac_raw)

    #Active Subspace Analysis
    if lsa_options.run_pss:
        active_set, inactive_set, ident_values, ident_vectors= \
            get_active_subset(model, lsa_options,logging = logging)
        #Collect Outputs and return as an lsa object
    if lsa_options.run_lsa and lsa_options.run_pss:
        return LsaResults(jacobian=jac_raw, rsi=jac_rsi, fisher=fisher_mat,\
                          active_set=active_set, inactive_set=inactive_set,\
                          ident_values = ident_values, ident_vectors = ident_vectors)
    elif lsa_options.run_pss and not lsa_options.run_lsa:
        return LsaResults(active_set=active_set, inactive_set=inactive_set,\
                          ident_values = ident_values, ident_vectors = ident_vectors)
    elif lsa_options.run_lsa and not lsa_options.run_pss:
        return LsaResults(jacobian=jac_raw, rsi=jac_rsi, fisher=fisher_mat)
    
###----------------------------------------------------------------------------------------------
###-------------------------------------Support Functions----------------------------------------
###----------------------------------------------------------------------------------------------

  
    
  
##--------------------------------------GetJacobian-----------------------------------------------------
def get_jacobian(eval_fcn, x_base, x_delta, deriv_method, **kwargs):
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
        if deriv_method.lower()== 'complex':
            xPert = x_base + np.zeros(shape=x_base.shape)*1j                  # Initialize Complex Perturbed input value
            xPert[i_poi] += x_delta * 1j                                      # Add complex Step in input
        elif deriv_method.lower() == 'finite':
            xPert = x_base.copy()
            xPert[i_poi] += x_delta
        yPert = eval_fcn(xPert)                                        # Calculate perturbed output
        for i_qoi in range(0, n_qoi):                                        # Loop through QOIs
            if deriv_method.lower()== 'complex':
                jac[i_qoi, i_poi] = np.imag(yPert[i_qoi] / x_delta)                 # Estimate Derivative w/ 2nd order complex
            elif deriv_method.lower() == 'finite':
                jac[i_qoi, i_poi] = (yPert[i_qoi]-y_base[i_qoi]) / x_delta
            #Only Scale Jacobian if 'scale' value is passed True in function call
            if scale:
                jac[i_qoi, i_poi] *= x_base[i_poi] * np.sign(y_base[i_qoi]) / (sys.float_info.epsilon + y_base[i_qoi])
                                                                            # Scale jacobian for relative sensitivity
        del xPert, yPert, i_poi, i_qoi                                        # Clear intermediate variables
    return jac                                                              # Return Jacobian




##--------------------------------------------Parameter dimension reduction------------------------------------------------------

def get_active_subset(model, lsa_options,logging=0):
    """ Selects active subset algorithm and returns results.
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
    
    if lsa_options.pss_algorithm.lower()=="smith":
        if logging:
            print("Running Smith Textbook Subset Selection Algoirthm")
        active_set, inactive_set, ident_values, ident_vectors= \
            pss_smith(model.eval_fcn, model.base_poi, model.base_qoi, \
                              model.name_poi, model. name_qoi, lsa_options.pss_decomp_method,\
                              lsa_options.pss_rel_tol, lsa_options.x_delta, \
                              lsa_options.deriv_method, logging = logging)
    elif lsa_options.pss_algorithm.lower() == "rrqr":
        if logging:
            print("Running RRQR Parameter Subset Selection Algoirthm")
        active_set, inactive_set, ident_values, ident_vectors = \
            pss_rrqr(model.eval_fcn, model.base_poi, model.base_qoi, \
                              model.name_poi, model. name_qoi, lsa_options.pss_decomp_method,\
                              lsa_options.pss_rel_tol, lsa_options.x_delta, \
                              lsa_options.deriv_method, logging = logging)
    else :
        raise Exception("Unrecognized pss_algorith: " + str(lsa_options.pss_algorith))
    return active_set, inactive_set, ident_values, ident_vectors
        
def pss_rrqr(eval_fcn, base_poi, base_qoi, name_poi, name_qoi,\
                      decomp_method, subset_rel_tol, x_delta, deriv_method,
                      logging =0):
    """Calculates active and inactive parameter subsets according to algorithm B1 of
        [requires reference]. Algorithm outline is as follows.
        1) Compute Sensitivity Matrix
        2) Compute svd of sensitivity matrix or eigendecomposition of Fisher
        3) Identify unidentifiable parameters
        4) Check error of reduced model
    
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
    n_poi = base_poi.size
    unidentifiable_indexes = np.array([], dtype = int)
    identifiable_indexes = np.arange(0,n_poi,dtype = int)
    #Step 1) Calculate Sensitivity matrix
    sens = get_jacobian(eval_fcn, base_poi, x_delta,\
                         deriv_method, scale=False, y_base=base_qoi)
    #Step 2) Compute eigen/ singular value decomposition
    #Perform Eigendecomp
    if decomp_method.lower() == "eigen":
        #Caclulate Fisher
        fisher_mat=np.dot(np.transpose(sens), sens)
        ident_values, ident_vectors =np.linalg.eig(fisher_mat)
    elif decomp_method.lower() == "svd":
        #Perform QR Decomposition to improve efficiency
        Q, R =np.linalg.qr(sens, mode='reduced')
        u, ident_values, ident_vectors = np.linalg.svd(R, full_matrices = False)
    if logging >1 : 
        print("Initial Identifiability Values: " + str(ident_values))
        print("Initial Identifiability Vectors: "  + str(ident_vectors))
    #Step 3) Identify unidentifiable parameters
    unidentifiable_singular_values = np.nonzero((ident_values/ident_values[0])<subset_rel_tol)
    num_unidentifiable = np.size(unidentifiable_singular_values)
    num_identifiable = n_poi - num_unidentifiable
    if logging > 1:
        print("Num unidentifiable parameter: " + str(num_unidentifiable))
    if num_unidentifiable >0:
        #Get identifiability vector
        unidentifiable_vec = ident_vectors[:,-1]
        max_ind = np.argmax(np.abs(unidentifiable_vec))
        
        #Move largest magnitude element to bottom of vector
        P_tild = np.arange(0,n_poi)
        P_tild[[max_ind, -1]] = P_tild[[-1, max_ind]]
        
        #Compute QR of R*P_tild
        B = R[:, P_tild]
        Q_tild, R_tild = np.linalg.qr(B)
        
        Q = np.matmul(Q,Q_tild)
        R = R_tild
        P = P_tild
        if logging > 2:
            print("Q shape: "+ str(Q.shape))
            print("R: "+ str(R))
        del max_ind 
        for iteration in range(num_unidentifiable-1):
            #l is max index of remaining parameters to be reduced, so if n_poi =4
            # and n_unidentifiable = 2, then l=2
            l = n_poi - iteration-1
            R_11 = R[0:l,0:l]
            R_12 = R[0:l, (1+l):]
            R_22 = R[(1+l):, (1+l):]
            null1, null2, V_l = np.linalg.svd(R_11)
            v_l = V_l[:,-1]
            
            max_ind = np.argmax(np.abs(v_l))
            P_tild_loop = np.arange(0,v_l.shape[0])
            P_tild_loop[[-1, max_ind]] = P_tild_loop[[max_ind, -1]]
            
            R_11P = R_11[:, P_tild_loop]
            Q_tild_smlr, R_11_tild = np.linalg.qr(R_11P)
            if logging > 2:
                print("R_11P: "+ str(R_11P))
                print("Q_tild_smlr: "+ str(Q_tild_smlr))
                print(scp.linalg.block_diag(Q_tild_smlr, np.eye(n_poi-l)))
            Q = np.matmul(Q, scp.linalg.block_diag(Q_tild_smlr, np.eye(n_poi-l)))
            P[0:l] = P[P_tild_loop]
            R = scp.linalg.block_diag(R_11_tild, R_22)
            R[0:l, (l+1):] = np.matmul(np.transpose(Q_tild_smlr),R_12)
        identifiable_indexes = P[0:num_identifiable]
        unidentifiable_indexes = P[num_identifiable:]
    #Define active and inactive spaces
    active_set=name_poi[identifiable_indexes]
    inactive_set=name_poi[unidentifiable_indexes]
    
    # reduced_model = model_reduction(model, inactive_param)
    # reduced_model.base_poi=reduced_model.base_poi[inactive_index == False]
    # reduced_model.name_poi=reduced_model.name_poi[inactive_index == False]
    # reduced_model.eval_fcn = lambda reduced_poi: model.eval_fcn(
    #     np.array([x for x, y in zip(reduced_poi,model.base_poi) if inactive_index== True]))
    # #reduced_model.eval_fcn=lambda reduced_poi: model.eval_fcn(np.where(inactive_index==False, reduced_poi, model.base_poi))
    # reduced_model.base_qoi=reduced_model.eval_fcn(reduced_model.base_poi)
    return active_set, inactive_set, ident_values, ident_vectors

def pss_smith(eval_fcn, base_poi, base_qoi, name_poi, name_qoi,\
                      decomp_method, subset_rel_tol, x_delta, deriv_method,
                      logging =0):
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
    n_poi = base_poi.size
    inactive_index=np.zeros(n_poi)
    #Calculate Jacobian
    jac=get_jacobian(eval_fcn, base_poi, x_delta,\
                         deriv_method, scale=False, y_base=base_qoi)
    #Inititalize lists that hold the values and vectors at each iteration
    ident_values_stored = []
    ident_vectors_stored = []
    reduction_order = []
    unidentifiable_poi = 0
    while eliminate:
        #Perform Eigendecomp
        if decomp_method.lower() == "eigen":
            #Caclulate Fisher
            fisher_mat=np.dot(np.transpose(jac), jac)
            ident_values, ident_vectors =np.linalg.eig(fisher_mat)
        elif decomp_method.lower() == "svd":
            u, ident_values, ident_vectors = np.linalg.svd(jac, full_matrices = False)
            # if logging >1 : 
            #     print("Identifiability Values: " + str(ident_values))
            #     print("Identifiability Vectors: "  + str(ident_vectors))
            
        ident_values_stored.append(ident_values)
        ident_vectors_stored.append(ident_vectors)
            
        
        #Eliminate dimension/ terminate
        if ident_values[-unidentifiable_poi-1] < subset_rel_tol * np.max(ident_values):
            unidentifiable_poi+=1
            #Get inactive parameter
            inactive_param = np.argmax(np.absolute(ident_vectors[:, -unidentifiable_poi]))
                #This indexing may seem odd but its because we're keeping the full model parameter numbering while trying
                # to index within the reduced model so we have to add to the index the previously removed params
            #Record inactive param in inactive space
            if logging > 1 :
                print("Unidentifiable Param Index: " + str(inactive_param))
                print("Corresponding Vector: " + str(ident_vectors[:, -unidentifiable_poi]))
                # if np.all(ident_vectors[:, np.argmin(np.absolute(ident_values))] == ident_vectors[:,-1]):
                #     print("Unidentifiable vector is last vector")
                # else :
                #     print("Unidentifiable vector is NOT last vector")
            inactive_index[inactive_param]=1
            reduction_order.append(name_poi[inactive_param])
            #Zero out inactive elements of jacobian
            jac[:,inactive_param] = 0
        else:
            #Terminate Active Subspace if singular values within tolerance
            eliminate=False
            
    #Define active and inactive spaces
    active_set=name_poi[inactive_index == False]
    inactive_set=name_poi[inactive_index == True]

# reduced_model = model_reduction(model, inactive_param)
# reduced_model.base_poi=reduced_model.base_poi[inactive_index == False]
# reduced_model.name_poi=reduced_model.name_poi[inactive_index == False]
# reduced_model.eval_fcn = lambda reduced_poi: model.eval_fcn(
#     np.array([x for x, y in zip(reduced_poi,model.base_poi) if inactive_index== True]))
# #reduced_model.eval_fcn=lambda reduced_poi: model.eval_fcn(np.where(inactive_index==False, reduced_poi, model.base_poi))
# reduced_model.base_qoi=reduced_model.eval_fcn(reduced_model.base_poi)
    return active_set, inactive_set, ident_values_stored, ident_vectors_stored


def test_model_reduction(model, inactive_pois, n_samp, save_location,pss_tol, \
                         logging = 0):
    mpi_comm = MPI.COMM_WORLD
    mpi_rank = mpi_comm.Get_rank()
    mpi_size = mpi_comm.Get_size()
    # Get inactive indices
    inactive_indices = np.where(np.isin(model.name_poi, inactive_pois))
    if logging:
        print("Inactive Indices:" + str(inactive_indices))
        print("Inactive POIs:" + str(model.name_poi[inactive_indices]))
    # Generate Parameter Sample
    poi_samp_full = model.sample_fcn(n_samp)
    poi_samp_reduced = np.copy(poi_samp_full)
    # Fix Inactive Parameters
    poi_samp_reduced[:,inactive_indices] = model.base_poi[inactive_indices]

    print("poi_samp_full:" +str(poi_samp_full))
    print("poi_samp_reduced:" +str(poi_samp_reduced))
    # Compute model evaluations
    if mpi_size == 0: 
        qoi_samp_full = model.eval_fcn(poi_samp_full)
        qoi_samp_reduced = model.eval_fcn(poi_samp_reduced)
    else : 
        qoi_samp_full = parallel_eval(model.eval_fcn, poi_samp_full, logging = logging)
        qoi_samp_reduced = parallel_eval(model.eval_fcn, poi_samp_reduced, logging =logging)
    mpi_comm.Barrier()
    
    print("qoi_samp_full:" +str(qoi_samp_full))
    print("qoi_samp_reduced:" +str(qoi_samp_reduced))
    if mpi_rank==0:
        #Save results
        np.savez(save_location + "data.npz",\
                 qoi_samp_full = qoi_samp_full, qoi_samp_reduced = qoi_samp_reduced,\
                 poi_samp_reduced = poi_samp_reduced, poi_samp_full = poi_samp_full)
        for i_qoi in range(model.n_qoi):
            
            # Compute KDE approximations
            kernel_full = sct.gaussian_kde(qoi_samp_full[:,i_qoi])
            kernel_reduced = sct.gaussian_kde(qoi_samp_reduced[:,i_qoi])
            
            #Plot KDEs
            plot_kde(kernel_full, kernel_reduced, \
                     [np.min(qoi_samp_full[:,i_qoi]), np.max(qoi_samp_full[:,i_qoi])],\
                     model.name_qoi[i_qoi], 
                     save_location + "kde_"+ model.name_qoi[i_qoi] + ".png",\
                     pss_tol)
                
def test_model_reduction_precomputed(name_qoi, qoi_samp_full, qoi_samp_reduced, n_samp, save_location,pss_tol, \
                         logging = 0, fontsize = 14, linewidth = 2.5):
    mpi_comm = MPI.COMM_WORLD
    mpi_rank = mpi_comm.Get_rank()
    mpi_size = mpi_comm.Get_size()
    if mpi_rank==0:
        for i_qoi in range(name_qoi.size):
            
            # Compute KDE approximations
            kernel_full = sct.gaussian_kde(qoi_samp_full[:,i_qoi])
            kernel_reduced = sct.gaussian_kde(qoi_samp_reduced[:,i_qoi])
            
            #Plot KDEs
            plot_kde(kernel_full, kernel_reduced, \
                     [np.min(qoi_samp_full[:,i_qoi]), np.max(qoi_samp_full[:,i_qoi])],\
                     name_qoi[i_qoi], 
                     save_location + "kde_"+  name_qoi[i_qoi] + ".png",\
                     pss_tol,
                     fontsize = fontsize,
                     linewidth = linewidth)


        
def plot_kde(kernel_full, kernel_reduced, qoi_bounds, qoi_name, save_location, pss_tol,
             fontsize = 14, linewidth = 2.5):
    #Generate qoi evaluation values
    qoi_spacing = np.linspace(qoi_bounds[0], qoi_bounds[1],200)
    #Evaluate kernels at qoi values
    qois_full = kernel_full.pdf(qoi_spacing)
    qois_reduced = kernel_reduced.pdf(qoi_spacing)
    #Plot kernels
    fig = plt.figure()
    plt.rc('font', size= fontsize) 
    plt.plot(qoi_spacing, qois_full, linewidth = linewidth ,label="Full Model")
    plt.plot(qoi_spacing, qois_reduced, linewidth = linewidth, label="Reduced Model", linestyle='dashdot')
    plt.xlabel(qoi_name)
    plt.title("KDE of " + qoi_name +  " (PSS tolerance=" + str(pss_tol) + ")")
    plt.gca().xaxis.set_major_locator(MaxNLocator(nbins = 4))
    plt.gca().yaxis.set_major_locator(MaxNLocator(nbins = 4))
    plt.legend()
    #fig.tight_layout()
    plt.savefig(save_location)
    


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