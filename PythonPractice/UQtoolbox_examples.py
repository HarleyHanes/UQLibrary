# UQtoolbox_examples
# Module of example problems setup for running in UQtoolbox
#   Each example outputs a model and options object with all required fields

import UQtoolbox as uq
import numpy as np


def GetExample(example, **kwargs):
    # Master function for selecting an example using a corresponding string
    # Inputs: example- string that corresponds to the desired model
    # Outputs: model and options objects corresponding to the desired model

    # Select Example model
    if example.lower() == 'linear':
        baseEvalPoints = np.array([[0], [.5], [1], [2]])  # Currently requires 1xn or nx1 ordering
        model = uq.model(evalFcn=lambda params: linear_function(baseEvalPoints, params),
                         basePOIs=np.array([1, 1]),
                         covMat=np.array([[1, 0], [0, 1]]))
        jacOptions = uq.jacOptions()  # Load base jacobian setttings
        sampOptions = uq.sampOptions(nSamp=100)  # Keep normal sampling but reduce sample
        # size to 100
        plotOptions = uq.plotOptions()  # Load base plot options
    elif example.lower() == 'quadratic':
        baseEvalPoints = np.array([[0], [.5], [1], [2]])  # Currently requires 1xn or nx1 ordering
        model = uq.model(evalFcn=lambda params: quadratic_function(baseEvalPoints, params),
                         basePOIs=np.array([1, 1, 1]),
                         covMat=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
        jacOptions = uq.jacOptions()  # Load base jacobian setttings
        sampOptions = uq.sampOptions(nSamp=100)  # Keep normal sampling but reduce sample size to 100
        plotOptions = uq.plotOptions()  # Load base plot options
    elif example.lower() == 'helmholtz':
        baseEvalPoints = np.transpose(np.arange(0, 1, .02))
        print(baseEvalPoints)
        model = uq.model(evalFcn=lambda params: HelmholtzEnergy(baseEvalPoints, params),
                         basePOIs=np.array([-392.66, 770.1, 57.61]),
                         covMat=np.array([[22.55, -110.062, 123.47],  # Covaraince matrix calculated by OLS
                                          [110.062, 585.06, -690.28],  # at baseParams and basEvalPoints
                                          [123.47, -690.28, 842.21]]))
        jacOptions = uq.jacOptions()  # Load base jacobian setttings
        sampOptions = uq.sampOptions(nSamp=100)  # Keep normal sampling but reduce sample size to 100
        plotOptions = uq.plotOptions()  # Load base plot options
    else:
        raise Exception("Unrecognized Example Type")

    # Apply optional inputs
    if 'basePOI' in kwargs:  # Change base parameter values to input
        model.basePOIs = kwargs['basePOI']
    if 'evalPoints' in kwargs:  # Determine eval points
        model.evalPoints = kwargs['evalPoints']

    # Combine options objects
    options = uq.uqOptions(jac=jacOptions, plot=plotOptions, samp=sampOptions)

    return model, options


def linear_function(x, params):
    return params[0] + x * params[1]


def quadratic_function(x, params):
    return params[0] + x * params[1] + x * params[2]


def HelmholtzEnergy(x, params):
    return params[0] * (x ** 2) + params[1] * (x ** 4) + params[2] * (x ** 6)
