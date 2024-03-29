# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 12:54:21 2022

@author: USER
"""


import numpy as np
import sys

#Load functions to be tested
sys.path.insert(0, '../../')
import UQLibrary as uq

#==============================================================================
#-------------------------------LSA integrated tests---------------------------
#==============================================================================

#----------------------------------Jacobian ----------------------------------
def test_linear_jac_finite():
    (model, options) = uq.examples.GetExample("linear")
    #Run only jacobian approximation with finite approx
    options.gsa.run = False
    options.lsa.run_pss = False
    options.lsa.x_delta = 1e-3
    options.lsa.deriv_method = "finite"
    options.display = False
    
    model.base_poi = np.array([.5, .5])  #np.random.uniform(size=2)
    
    results = uq.run_uq(model, options)
    
    true_jac = np.array([1, 2])
    
    assert np.allclose(results.lsa.jac, true_jac)
    
def test_linear_jac_complex():
    (model, options) = uq.examples.GetExample("linear")
    #Run only jacobian approximation with finite approx
    options.gsa.run = False
    options.lsa.run_pss = False
    options.lsa.x_delta = 1e-16
    options.lsa.deriv_method = "complex"
    options.display = False
    
    model.base_poi = np.random.uniform(size=2)
    
    results = uq.run_uq(model, options)
    
    true_jac = np.array([1, 2])
    assert np.allclose(results.lsa.jac, true_jac)
    
#==============================================================================
#------------------------------Identifiability tests---------------------------
#==============================================================================

# Heated Rod
    
# Helmholtz, RRQR Algorithm
def test_helmholtz_rrqr_ident_subset_complex():
     (model, options) = uq.examples.GetExample("helmholtz (identifiable)")
     #Run only lsa with finite approx
     options.gsa.run = False
     options.lsa.run_lsa = False
     options.lsa.run_pss =True
     options.lsa.x_delta = 1e-16
     options.lsa.deriv_method = "complex"
     options.lsa.pss_algorithm = "RRQR"
     options.lsa.pss_rel_tol = 1e-6
     options.lsa.pss_decomp_method = "svd"
     options.display = True
     options.plot = False
   
     results = uq.run_uq(model, options)
     true_subset = np.array(["alpha1", "alpha11", "alpha111"])
     assert np.all(true_subset==results.lsa.active_set)
     
def test_helmholtz_rrqr_single_unident_subset_complex():
     (model, options) = uq.examples.GetExample("helmholtz (unidentifiable)")
     #Run only lsa with finite approx
     options.gsa.run = False
     options.lsa.run_lsa = False
     options.lsa.run_pss =True
     options.lsa.x_delta = 1e-16
     options.lsa.deriv_method = "complex"
     options.lsa.pss_algorithm = "RRQR"
     options.lsa.pss_rel_tol = 1e-6
     options.lsa.pss_decomp_method = "svd"
     options.display = True
     options.plot = False
   
     results = uq.run_uq(model, options)
     true_subset = np.array(["alpha1", "alpha11"])
     
     assert np.all(true_subset==results.lsa.active_set)
     
def test_helmholtz_rrqr_double_unident_subset_complex():
     (model, options) = uq.examples.GetExample("helmholtz (double unidentifiable)")
     #Run only lsa with finite approx
     options.gsa.run = False
     options.lsa.run_lsa = False
     options.lsa.run_pss =True
     options.lsa.x_delta = 1e-16
     options.lsa.deriv_method = "complex"
     options.lsa.pss_algorithm = "RRQR"
     options.lsa.pss_rel_tol = 1e-6
     options.lsa.pss_decomp_method = "svd"
     options.display = True
     options.plot = False
   
     results = uq.run_uq(model, options)
     true_ident = np.array(["alpha1", "alpha11"])
     true_unident = np.array(["alpha111", "fake parameter"])
     
     assert np.all(true_ident==results.lsa.active_set) and \
         np.all(true_unident==results.lsa.inactive_set)
         
 
def test_helmholtz_smith_ident_subset_complex():
     (model, options) = uq.examples.GetExample("helmholtz (identifiable)")
     #Run only lsa with finite approx
     options.gsa.run = False
     options.lsa.run_lsa = False
     options.lsa.run_pss =True
     options.lsa.x_delta = 1e-16
     options.lsa.deriv_method = "complex"
     options.lsa.pss_algorithm = "Smith"
     options.lsa.pss_rel_tol = 1e-6
     options.lsa.pss_decomp_method = "svd"
     options.display = True
     options.plot = False
   
     results = uq.run_uq(model, options)
     true_subset = np.array(["alpha1", "alpha11", "alpha111"])
     assert np.all(true_subset==results.lsa.active_set)
     
def test_helmholtz_smith_single_unident_subset_complex():
     (model, options) = uq.examples.GetExample("helmholtz (unidentifiable)")
     #Run only lsa with finite approx
     options.gsa.run = False
     options.lsa.run_lsa = False
     options.lsa.run_pss =True
     options.lsa.x_delta = 1e-16
     options.lsa.deriv_method = "complex"
     options.lsa.pss_algorithm = "Smith"
     options.lsa.pss_rel_tol = 1e-6
     options.lsa.pss_decomp_method = "svd"
     options.display = True
     options.plot = False
   
     results = uq.run_uq(model, options)
     true_subset = np.array(["alpha1", "alpha11"])
     
     assert np.all(true_subset==results.lsa.active_set)
     
def test_helmholtz_smith_double_unident_subset_complex():
     (model, options) = uq.examples.GetExample("helmholtz (double unidentifiable)")
     #Run only lsa with finite approx
     options.gsa.run = False
     options.lsa.run_lsa = False
     options.lsa.run_pss =True
     options.lsa.x_delta = 1e-16
     options.lsa.deriv_method = "complex"
     options.lsa.pss_algorithm = "Smith"
     options.lsa.pss_rel_tol = 1e-6
     options.lsa.pss_decomp_method = "svd"
     options.display = True
     options.plot = False
   
     results = uq.run_uq(model, options)
     true_ident = np.array(["alpha1", "alpha11"])
     true_unident = np.array(["alpha111", "fake parameter"])
     
     assert np.all(true_ident==results.lsa.active_set) and \
         np.all(true_unident==results.lsa.inactive_set)

#==============================================================================
#----------------------------Morris integrated tests---------------------------
#==============================================================================

def test_linear_portfolio():
    (model, options) = uq.examples.GetExample("portfolio (normal)")
    options.gsa.run = True
    options.lsa.run = False
    options.gsa.run_sobol = False
    options.gsa.run_morris = True
    options.gsa.l_morris = 1/40
    options.gsa.n_samp_morris = 100
    options.plot = False
    
    results = uq.run_uq(model, options)
    assert np.allclose(results.gsa.morris_mean_abs, np.array([2, 1]))
    
# def test_sir_4param():
#     (model, options) = uq.examples.GetExample("sir 4 param")
#     options.gsa.run = True
#     options.lsa.run = False
#     options.gsa.run_sobol = False
#     options.gsa.run_morris = True
#     options.gsa.l_morris = 1/40
#     options.gsa.n_samp_morris = 100
    
#     results = uq.run_uq(model, options)
#     assert np.allclose(results.gsa.morris_mean_abs, np.array([.1448,.2422, 1.0257, 1.0012])*10**4, rtol = 10**(-2))
    