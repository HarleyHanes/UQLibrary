# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 10:44:53 2022

@author: USER
"""

import numpy as np
import sys

#Load Load functions to be tested
sys.path.insert(0, '../')
from lsa import get_jacobian
    
#----------------------------------Support Functions---------------------------
    

def third_order_uni_poly(x,c):
    p_x = c[0]+c[1]*x+c[2]*x**2 + c[3]* x**3
    return  np.array([p_x])

def third_order_uni_poly_deriv(x,c):
    dpdx = c[1]+2*c[2]*x + 3* c[3]*x**2
    return dpdx

def third_order_multi_poly(x,c):
    p_x = np.sum(c[0,:]+c[1,:]*x+c[2,:]*x**2 + c[3,:]* x**3)
    return  np.array([p_x])

def third_order_multi_poly_grad(x,c):
    gradf = c[1,:] + 2*c[2,:]*x + 3*c[3,:]*x**2
    return gradf

#==============================================================================
#------------------------------------Jacobian Tests----------------------------
#==============================================================================

# -------------------------------Interior Point Tests--------------------------
#Finite difference
#1) R1-> R1
def test_finite_diff_uni_poly():
    #Test random points
    c_rand = np.random.uniform(size = (4))
    x_rand = np.random.uniform()
    
    fcn = lambda x: third_order_uni_poly(x,c_rand) 
    deriv_approx = get_jacobian(fcn, x_rand, 1e-8, "finite")
    deriv = third_order_uni_poly_deriv(x_rand, c_rand)
    assert np.allclose(deriv_approx,deriv)
    
#2) R3-> R1
def test_finite_diff_multi_poly():
    c_rand=np.random.uniform(size=(4,3))
    x_rand = np.random.uniform(size= (3))
    
    fcn = lambda x: third_order_multi_poly(x, c_rand)
    grad_approx = get_jacobian(fcn, x_rand, 1e-8, "finite")
    grad = third_order_multi_poly_grad(x_rand, c_rand)
    assert np.allclose(grad_approx,grad)
    
    
#Complex Step
#1) R1 -> R1
def test_complex_uni_poly():
    #Test random points
    c_rand = np.random.uniform(size = (4))
    x_rand = np.random.uniform()
    
    fcn = lambda x: third_order_uni_poly(x,c_rand) 
    deriv_approx = get_jacobian(fcn, x_rand, 1e-16, "complex")
    deriv = third_order_uni_poly_deriv(x_rand, c_rand)
    assert np.allclose(deriv_approx,deriv)
    
#2) R3 -> R1
def test_complex_multi_poly():
    c_rand=np.random.uniform(size=(4,3))
    x_rand = np.random.uniform(size= (3))
    
    fcn = lambda x: third_order_multi_poly(x, c_rand)
    grad_approx = get_jacobian(fcn, x_rand, 1e-16, "complex")
    grad = third_order_multi_poly_grad(x_rand, c_rand)
    assert np.allclose(grad_approx,grad)
    
#--------------------------------Boundary Point Tests--------------------------