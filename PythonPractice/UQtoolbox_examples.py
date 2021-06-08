# UQtoolbox_examples
# Module of example problems setup for running in UQtoolbox
#   Each example outputs a model and options object with all required fields

import UQtoolbox as uq
import numpy as np
import math as math


def GetExample(example, **kwargs):
    # Master function for selecting an example using a corresponding string
    # Inputs: example- string that corresponds to the desired model
    # Outputs: model and options objects corresponding to the desired model

    # Initialize options object
    options = uq.uqOptions()
    # Select Example model
    if example.lower() == 'linear':
        #baseEvalPoints = np.array([0, .5, 1, 2])  # Requires 1xnQOIs indexing
        baseEvalPoints= np.array([1])
        model = uq.model(evalFcn=lambda params: linear_function(baseEvalPoints, params),
                         basePOIs=np.array([1, 1]),
                         cov=np.array([[1, 0], [0, 1]]),
                         dist='unif',
                         distParms=np.array([[0], [1]])*np.array([1, 1])
                         #distParms=np.array([[.9999999999], [1.0000000001]])*np.array([1, 1])
                         )
        options.gsa = uq.gsaOptions() # Keep normal sampling scheme
    elif example.lower() == 'quadratic':
        baseEvalPoints = np.array([0, .5, 1, 2])  # Currently requires 1xn or nx1 ordering
        model = uq.model(evalFcn=lambda params: quadratic_function(baseEvalPoints, params),
                         basePOIs=np.array([1, 1, 1]),
                         cov=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
        options.gsa = uq.gsaOptions(nSamp=100)  # Keep normal sampling but reduce sample size to 100
    elif example.lower() == 'helmholtz':
        baseEvalPoints = np.arange(0, 1, .02)
        model = uq.model(evalFcn=lambda params: HelmholtzEnergy(baseEvalPoints, params),
                         basePOIs=np.array([-392.66, 770.1, 57.61]),
                         cov=np.array([[0.0990, - 0.4078, 0.4021],  # Covaraince matrix calculated by DRAMs
                                       [-0.4078, 2.0952, -2.4078],  # at baseParams and basEvalPoints
                                       [0.4021, -2.4078, 3.0493]]) * (10 ** 3),
                         dist='unif')  # Use uniform sampling of +-20% nominal value
        model.distParms = np.array([[.8, .8, .8], [1.2, 1.2, 1.2]]) * model.basePOIs
        options.gsa = uq.gsaOptions(nSamp=10000)  # Keep normal sampling but reduce sample size to 100
    elif example.lower() == 'integrated helmholtz':
        model = uq.model(evalFcn=lambda params: IntegratedHelmholtzEnergy(np.array([.8, .80001]), params),
                         basePOIs=np.array([-389.4, 761.3, 61.5]),
                         cov=np.array([[0.0990, - 0.4078, 0.4021],  # Covaraince matrix calculated by DRAMs
                                       [-0.4078, 2.0952, -2.4078],  # at baseParams and basEvalPoints
                                       [0.4021, -2.4078, 3.0493]]) * (10 ** 3),
                         dist="unif")  # Use uniform sampling of +-20% nominal value
        model.distParms = np.array([[.8, .8, .8], [1.2, 1.2, 1.2]]) * model.basePOIs
        #model.distParms = np.array([[.999999, .999999, .999999], [1.000001, 1.000001, 1.000001]]) * model.basePOIs
        options.gsa = uq.gsaOptions()  # Use default number of samples
    elif example.lower() == 'linear product':  # Linear product example taken from Homma1996
        model = uq.model(evalFcn=LinearProd,
                         basePOIs=np.array([.5, .5, .5, .5, .5]),
                         dist="unif",
                         distParms=np.array([[0, 0, 0, 0, 0], [1, 1, 1, 1, 1]]))
        options.gsa = uq.gsaOptions()  # Use default number of samples
    elif example.lower() == 'ishigami':
        model = uq.model(evalFcn=Ishigami,
                         basePOIs=np.array([0, 0, 0]),
                         dist="unif",
                         distParms=np.array([[-math.pi, -math.pi, -math.pi], [math.pi, math.pi, math.pi]]))
        options.lsa=uq.lsaOptions(method='finite', xDelta=10**(-6))
        options.gsa=uq.gsaOptions()   # Use default number of samples
    elif example.lower() == 'trial function':
        model = uq.model(evalFcn=TrialFunction,
                         basePOIs=np.array([1, 1, 1]),
                         dist="unif",
                         distParms=np.array([[1, 1, 1], [1000, 100, 10]])
                         )
        options.gsa=uq.gsaOptions()

    else:
        raise Exception("Unrecognized Example Type")

    # Apply optional inputs
    if 'basePOI' in kwargs:  # Change base parameter values to input
        model.basePOIs = kwargs['basePOI']
    if 'evalPoints' in kwargs:  # Determine eval points
        model.evalPoints = kwargs['evalPoints']

    return model, options


def linear_function(x, params):
    if params.ndim == 1:
        return params[0] + (x * params[1])
    if params.ndim == 2:
        return params[:, 0] + (params[:, 1] * x)


def quadratic_function(x, params):
    if params.ndim == 1:
        return params[0] + (x * params[1]) + ((x ** 2) * params[2])
    if params.ndim == 2:
        return np.outer(params[:, 0], np.ones(x.shape)) + np.outer(params[:, 1], x) + np.outer(params[:, 2], x**2)


def HelmholtzEnergy(x, params):
    if params.ndim == 1:
        return params[0] * (x ** 2) + params[1] * (x ** 4) + params[2] * (x ** 6)
    elif params.ndim == 2:
        return params[:, 0] * (x ** 2) + params[:, 1] * (x ** 4) + params[:, 2] * (x ** 6)


def IntegratedHelmholtzEnergy(x, params):
    if params.ndim == 1:
        return params[0] * (x ** 3) / 3 + params[1] * (x ** 5) / 5 + params[2] * (x ** 7) / 7
    elif params.ndim == 2:
        return np.outer(params[:, 0], (x ** 3)) / 3 + np.outer(params[:, 1], (x ** 5)) / 5 + np.outer(params[:, 2], (x ** 7)) / 7



def LinearProd(params):
    if params.ndim == 1:
        return np.array([np.prod(2 * params + 1) / (2 ** (len(np.transpose(params))))])
    elif params.ndim == 2:
        return np.prod(2 * params + 1, axis=1) / (2 ** (len(np.transpose(params)) + 1))
def Ishigami(params):
    if params.ndim == 1:
        return np.array([np.sin(params[0])+np.sin(params[1])**2+(params[2]**4)*np.sin(params[0])])
    elif params.ndim == 2:
        return np.array([np.sin(params[:, 0])+np.sin(params[:, 1])**2+(params[:, 2]**4)*np.sin(params[:, 0])])
def TrialFunction(params):
    if params.ndim == 1:
        return np.array([params[0]+params[1]*(params[2]**2)])
    elif params.ndim == 2:
        return params[:, 0]+params[:, 1]*(params[:, 2]**2)