# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

# Import modules
import numpy as np
import SensitivityAnalysis as uq

# Companion Functions
def linear_function(x, params):
    return params[0] + x * params[1]

def quadratic_function(x, params):
    return params[0] + x * params[1] + x*params[2]

# Main Script


# Define assessing points and parameters
evalPoints = np.array([0, .5, 1, 2])                                    # Currently requires 1xn or nx1 ordering


# Select eval function
evalFcn= lambda params:quadratic_function(evalPoints,params)            #Quadratic Function

#Select base parameter values
params=np.array([1, 1, 1])

#Create model object
model=uq.model(baseParams=params, evalFcn=evalFcn)

#Run LSA to get jacobian, RSI, and fisher information matrix
lsa=uq.LSA(model,h=1e-6,xBase=model.basePOIs)

#jacobian=uq.GetJacobian(model, h=1e-6, scale=True, xBase=np.array([1, 1, 1]))
#uq.GSA(params, evalFcn)


print("Evaluation Points: " + str(evalPoints))
print("Base Parameters: " + str(params))
print("Base values: " + str(model.baseQOIs))
print("Jacobian: " + str(lsa.jac))
print("Scaled Jacobian: " + str(lsa.rsi))
print("Fisher Matrix: " + str(lsa.fisher))


# Estimate Jacobian
# h=.01
# xEval=x
# yBase=y
# for iOutput in range(0,len(y.transpose)):
#     for iDim in range(0, len(x)): #Cycle through observation dimension
#         xEval=x[i, :]
#         yBase=y[i, :]
#         xLeft=xEval-h
#         xRight=xEval+h
#         yLeft=eval_function(xLeft)
#         yRight=eval_function(xRight)
#         # Jac[i]=(yRight+yLeft-2*yBase)/(h**2)
#         Jac[i]=(yRight-yLeft)/(2*h)
#
# print("Jacobian: " + str(Jac))

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
