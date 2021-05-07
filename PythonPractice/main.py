# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

# Import modules
# import numpy as np
import UQtoolbox as uq
import UQtoolbox_examples as uqExamples

# Main Script


# Define assessing points and parameters


# Get model and options object from Example set
# [model, options]=uqExamples.GetExample('linear')
[model, options] = uqExamples.GetExample('linear')

# Run UQ package
results = uq.RunUQ(model, options)

# print("Evaluation Points: " + str(evalPoints))
# print("Base Parameters: " + str(model.basePOIs))
# print("Base values: " + str(model.baseQOIs))
# print("Jacobian: " + str(results.lsa.jac))
# print("Scaled Jacobian: " + str(results.lsa.rsi))
# print("Fisher Matrix: " + str(results.lsa.fisher))
# print("1st Order Sobol Indices: " + str(results.gsa.sobolBase))
# print("1st Order Sobol Indices: " + str(results.gsa.sobolTot))

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

# See PyCharm help at https://www.jetbrains.com/help/pycha