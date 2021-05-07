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
        baseEvalPoints = np.array([0, .5, 1, 2])  #Requires 1xnQOIs indexing
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
                         covMat=np.array([[0.0990, - 0.4078, 0.4021],  # Covaraince matrix calculated by DRAMs
                                          [-0.4078, 2.0952, -2.4078],  # at baseParams and basEvalPoints
                                          [0.4021, -2.4078, 3.0493]]) * (10 ** 3))
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
    if params.ndim==1:
        return params[0] + x * params[1]
    if params.ndim==2:
        return np.outer(params[:,0],np.ones(x.size)) + np.outer(params[:,1],x)


def quadratic_function(x, params):
    if params.ndim==1:
        return params[0] + x * params[:,1] + x**2 * params[:,2]
    if params.ndim==2:
        return params[:,0] + x * params[:,1] + x**2 * params[:,2]


def HelmholtzEnergy(x, params):
    if params.ndim == 1:
        return params[0] * (x ** 2) + params[1] * (x ** 4) + params[2] * (x ** 6)
    elif params.ndim == 2:
        return params[:, 0] * (x.transpose() ** 2) + params[:, 1] * (x.transpose() ** 4) + params[:, 2] * (x**6)
           