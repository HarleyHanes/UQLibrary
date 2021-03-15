# UQtoolbox_examples
# Module of example problems setup for running in UQtoolbox
#   Each example outputs a model and options object with all required fields

import UQtoolbox as uq
import numpy as np


def GetExample(example):
    # Master function for selecting an example using a corresponding string
    # Inputs: example- string that corresponds to the desired model
    # Outputs: model and options objects corresponding to the desired model
    if example.lower() == 'linear':
        evalPoints = np.array([[0], [.5], [1], [2]])  # Currently requires 1xn or nx1 ordering
        model = uq.model(evalFcn=lambda params: linear_function(evalPoints, params),
                         baseParams=np.array([1, 1]),
                         covMat=np.array([[1, 0], [0, 1]]))
        jacOptions = uq.jacOptions()  # Load base jacobian setttings
        sampOptions = uq.sampOptions(nSamp=100)  # Keep normal sampling but reduce sample
        # size to 100
        plotOptions = uq.plotOptions()  # Load base plot options
        options = uq.uqOptions(jac=jacOptions, plot=plotOptions, samp=sampOptions)  # Combine options objects
    elif example.lower()=='quadratic':
        evalPoints = np.array([[0], [.5], [1], [2]])  # Currently requires 1xn or nx1 ordering
        model = uq.model(evalFcn=lambda params: quadratic_function(evalPoints, params),
                         baseParams=np.array([1, 1, 1]),
                         covMat=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
        jacOptions = uq.jacOptions()  # Load base jacobian setttings
        sampOptions = uq.sampOptions(nSamp=100)  # Keep normal sampling but reduce sample
        # size to 100
        plotOptions = uq.plotOptions()  # Load base plot options
        options = uq.uqOptions(jac=jacOptions, plot=plotOptions, samp=sampOptions)  # Combine options objects

    else:
        raise Exception("Unrecognized Example Type")

    return model, options


def linear_function(x, params):
    return params[0] + x * params[1]

def quadratic_function(x, params):
    return params[0] + x * params[1] + x*params[2]
