

from SALib.sample import saltelli
from SALib.analyze import sobol
from SALib.test_functions import Ishigami
import UQtoolbox as uq
import UQtoolbox_examples as uqExamples
import numpy as np


#Set seed for reporducibility
np.random.seed(10)
#
# # Define the model inputs
problem = {
    'num_vars': 3,
    'names': ['x1', 'x2', 'x3'],
    'bounds': [[-3.14159265359, 3.14159265359],
               [-3.14159265359, 3.14159265359],
               [-3.14159265359, 3.14159265359]]
}

# Generate samples
param_values = saltelli.sample(problem, 512*2)

# Run model (example)
Y = Ishigami.evaluate(param_values)

# Perform analysis
Si = sobol.analyze(problem, Y, print_to_console=True)

# Print the first-order sensitivity indices
print(Si['S1'])

#
# # Main Script
#
#
# # Define assessing points and parameters
# #
#
# # Get model and options object from Example set
# # [model, options] = uqExamples.GetExample('integrated helmholtz')
#
# # [model, options] = uqExamples.GetExample('linear product')
#
#

#
[model, options] = uqExamples.GetExample('ishigami')
#
# # [model, options] = uqExamples.GetExample('linear')
#
# # [model, options] = uqExamples.GetExample('trial function')
#
# # Run UQ package
results = uq.RunUQ(model, options)

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
