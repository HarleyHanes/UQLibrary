# UQtoolbox_examples
# Module of example problems setup for running in UQtoolbox
#   Each example outputs a model and options object with all required fields

import UQLibrary as uq
import numpy as np
import math as math
import scipy.integrate as integrate

def GetExample(example, **kwargs):
    # Master function for selecting an example using a corresponding string
    # Inputs: example- string that corresponds to the desired model
    # Outputs: model and options objects corresponding to the desired model

    # Initialize options object
    options = uq.Options()
    # Select Example model
    if example.lower() == 'linear':
        #baseEvalPoints = np.array([0, .5, 1, 2])  # Requires 1xnQOIs indexing
        baseEvalPoints= np.array([2])
        model = uq.Model(eval_fcn=lambda params: linear_function(baseEvalPoints, params),
                         base_poi=np.array([1, 1]),
                         cov=np.array([[1, 0], [0, 1]]),
                         dist_type='uniform',
                         dist_param=np.array([[0], [1]])*np.array([1, 1])
                         #dist_param=np.array([[.9999999999], [1.0000000001]])*np.array([1, 1])
                         )
    
    elif example.lower() == 'quadratic':
        baseEvalPoints = np.array([0, .5, 1, 2])  # Currently requires 1xn or nx1 ordering
        model = uq.Model(eval_fcn=lambda params: quadratic_function(baseEvalPoints, params),
                         base_poi=np.array([1, 1, 1]),
                         cov=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
        options.gsa.nSampSobol=100               # Keep normal sampling but reduce sample size to 100
    
    elif example.lower() == 'helmholtz':
        baseEvalPoints = np.arange(0, .801, .05)
        model = uq.Model(eval_fcn=lambda params: HelmholtzEnergy(baseEvalPoints, params),
                         base_poi=np.array([-392.66, 770.1, 57.61]),
                         cov=np.array([[0.0990, - 0.4078, 0.4021],  # Covaraince matrix calculated by DRAMs
                                       [-0.4078, 2.0952, -2.4078],  # at baseParams and basEvalPoints
                                       [0.4021, -2.4078, 3.0493]]) * (10 ** 3),
                         poi_name = np.array(["alpha1", "alpha11", "alpha111"]),
                         dist_type='uniform')  # Use uniform sampling of +-20% nominal value
        model.dist_param = np.array([[.8, .8, .8], [1.2, 1.2, 1.2]]) * model.base_poi
        options.gsa.nSampSobol=10000              # Keep normal sampling but reduce sample size to 100
    
    elif example.lower() == 'integrated helmholtz':
        model = uq.Model(eval_fcn=lambda params: IntegratedHelmholtzEnergy(np.arange(0,.8,.06), params),
                         base_poi=np.array([-389.4, 761.3, 61.5]),
                         cov=np.array([[0.0990, - 0.4078, 0.4021],  # Covaraince matrix calculated by DRAMs
                                       [-0.4078, 2.0952, -2.4078],  # at baseParams and basEvalPoints
                                       [0.4021, -2.4078, 3.0493]]) * (10 ** 3),
                         name_poi=np.array(["alpha_1", "alpha11", "alpha111"]),
                         name_qoi=np.array(["x=.8", "x=.80001"]),
                         dist_type="uniform")  # Use uniform sampling of +-20% nominal value
        model.dist_param = np.array([[.8, .8, .8], [1.2, 1.2, 1.2]]) * model.base_poi
        #model.dist_param = np.array([[.999999, .999999, .999999], [1.000001, 1.000001, 1.000001]]) * model.base_poi
    
    elif example.lower() == 'linear product':  # Linear product example taken from Homma1996
        model = uq.Model(eval_fcn=LinearProd,
                         base_poi=np.array([.5, .5, .5, .5, .5]),
                         dist_type="uniform",
                         dist_param=np.array([[0, 0, 0, 0, 0], [1, 1, 1, 1, 1]]))
        
    elif example.lower() == 'ishigami (uniform)':
        model = uq.Model(eval_fcn=Ishigami,
                         base_poi=np.array([0, 0, 0]),
                         dist_type="saltelli uniform",
                         dist_param=np.array([[-math.pi, -math.pi, -math.pi], [math.pi, math.pi, math.pi]]))
        options.lsa.method = 'finite' 
        options.lsa.xDelta = 10**(-6)
        options.gsa.nSampSobol = 500000          # Use default number of samples
        
    elif example.lower() == 'ishigami (normal)':
        model = uq.Model(eval_fcn=Ishigami,
                         base_poi=np.array([0, 0, 0]),
                         dist_type="saltelli normal",
                         dist_param=np.array([[0, 0, 0], [(2*math.pi)**2/12, (2*math.pi)**2/12, (2*math.pi)**2/12]]))
        options.lsa.method = 'finite' 
        options.lsa.xDelta = 10**(-6)
        options.gsa.nSampSobol= 500000 
        
    elif example.lower() == 'trial function':
        model = uq.Model(eval_fcn=TrialFunction,
                         base_poi=np.array([1, 1, 1]),
                         dist_type="uniform",
                         dist_param=np.array([[1, 1, 1], [1000, 100, 10]])
                         )
    #---------------------------------Portfolio Models------------------------------------------------------------
    # See p.353 of Smith, 2015, Uncertainty Quantification
    elif example.lower() == 'portfolio (normal)':
        model=uq.Model(eval_fcn=lambda params: Portfolio(params, np.array([2, 1])),
                       base_poi=np.array([0, 0]),
                       dist_type="normal",
                       dist_param=np.array([[0, 0], [1, 9]]))
        options.gsa.nSampSobol = 100000
        
    elif example.lower() == 'portfolio (uniform)':
        model=uq.Model(eval_fcn=lambda params: Portfolio(params, np.array([2, 1])),
                       base_poi=np.array([0, 0]),
                       dist_type="uniform",
                       dist_param=np.array([[-np.sqrt(12)/2, -3*np.sqrt(3)], [np.sqrt(12)/2, 3*np.sqrt(3)]]))
        options.path = '..\\Figures\\Portfolio(Uniform)'
        options.gsa.nSampSobol = 2**12
    #--------------------------------Heated Rod Models------------------------------------------------------------
    elif example.lower() == 'aluminum rod (uniform)':
        model = uq.Model(eval_fcn=lambda params: HeatRod(params, np.array([55])),
                         base_poi=np.array([-18.4, .00191]),
                         dist_type="uniform",
                         dist_param=np.array([[-18.4-(.1450*np.sqrt(3)), .00191-(1.4482*(10**(-5))*np.sqrt(3))],\
                                             [-18.4+(.1450*np.sqrt(3)), .00191+(1.4482*(10**(-5))*np.sqrt(3))]]),
                         name_poi=np.array(['Phi', 'h']),
                         name_qoi=np.array(['T(x=55)']))
        options.path = '..\\Figures\\AluminumRod(Uniform, x=55)'
        options.gsa.nSampSobol = 500000
        
    elif example.lower() == 'aluminum rod (normal)':
        model = uq.Model(eval_fcn=lambda params: HeatRod(params, np.array([15, 25, 35, 45, 55])),
                         base_poi=np.array([-18.4, .00191]),
                         dist_type='normal',
                         dist_param=np.array([[-18.4, .00191], [.1450**2, (1.4482*10**(-5))**2]]),
                         name_poi=np.array(['Phi', 'h']),
                         name_qoi=np.array(["x=15", "x=25", "x=35", "x=45", "x=55"])
                         #name_qoi=np.array(['T(x=15)','T(x=55)'])
                         )
        options.path = '..\\Figures\\AluminumRod(Normal)'
        options.gsa.nSampSobol = 100000000
        
    elif example.lower() == 'aluminum rod (saltelli normal)':
        model = uq.Model(eval_fcn=lambda params: HeatRod(params, np.array([55])),
                         base_poi=np.array([-18.4, .00191]),
                         dist_type="saltelli normal",
                         dist_param=np.array([[-18.4, .00191], [.1450**2, (1.4482*10**(-5))**2]]),
                         name_poi=np.array(['Phi', 'h']),
                         name_qoi=np.array(['T(x=55)']))
        options.path = '..\\Figures\\AluminumRod(saltelli_normal, x=55)'
        options.gsa.nSampSobol = 200000
    #------------------------------------SIR Models----------------------------------------------------------------
        
    elif example.lower() == 'sir infected':
        model = uq.Model(eval_fcn = lambda params: SolveSIRinfected(params, np.array([960, 40, 0]), np.array([0, 1, 3, 5, 6])),
                         base_poi=np.array([8, 1.5]),
                         name_poi=np.array(['beta', 'gamma']),
                         dist_type='uniform',
                         dist_param=np.array([[0, 0], [1, 1]])
                         )
        options.lsa.method='finite'
        options.lsa.xDelta=.00001
    elif example.lower() == 'sir enedmic':
            model = uq.Model(eval_fcn = lambda params: SIR_endemic_integrated(params, np.array([900, 100, 0]), 20),
                             base_poi=np.array([8, 1.5]),
                             name_poi=np.array(['gamma', 'k', 'r', 'delta']),
                             dist_type='uniform',
                             dist_param=np.array([[0, 0], [1, 1]])
                             )
            options.lsa.method='finite'
            options.lsa.xDelta=.00001
        
    elif example.lower() == 'sobol test function':
        model = uq.Model(eval_fcn= lambda params: SobolTestFunction(params,np.array([78, 12, .5, 2, 97, 33])),
                         base_poi= np.array([.5, .5, .5, .5, .5, .5]),
                         dist_type='uniform',
                         dist_param=np.array([[0,0,0,0,0,0], [1,1,1,1,1,1]]))
    else:
        raise Exception("Unrecognized Example Type")

    # Apply optional inputs
    if 'basePOI' in kwargs:  # Change base parameter values to input
        model.base_poi = kwargs['basePOI']
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
        return np.array([np.sin(params[0])+7*np.sin(params[1])**2+.1*(params[2]**4)*np.sin(params[0])])
    elif params.ndim == 2:
        return np.sin(params[:, [0]])+7*np.sin(params[:, [1]])**2+.1*(params[:, [2]]**4)*np.sin(params[:, [0]])
def TrialFunction(params):
    if params.ndim == 1:
        return np.array([params[0]+params[1]*(params[2]**2)])
    elif params.ndim == 2:
        return params[:, [0]]+params[:, [1]]*(params[:, [2]]**2)
def Portfolio(params,c):
    if params.ndim == 1:
        return np.array([c[0]*params[0]+c[1]*params[1]])
    elif params.ndim == 2:
        return c[0]*params[:, 0]+c[1]*params[:, 1]
def HeatRod(params,x):
    #Set or load paramters
    Tamb=21.19
    a=.95
    b=.95
    L=70
    k=2.37
    if params.ndim==1:
        Phi=params[0]
        h=params[1]
    elif params.ndim==2:
        Phi=params[:,0]
        h=params[:,1]
    #Compute intermediates
    gamma=np.sqrt(2*(a+b)*h/(a*b*k))
    c1=-Phi/(k*gamma)*(np.exp(gamma*L)*(h+k*gamma)/(np.exp(-gamma*L)*(h-k*gamma)+np.exp(gamma*L)*(h+k*gamma)))
    c2=Phi/(k*gamma)+c1
    #Compute temperature
    gx = np.outer(gamma, x)
    if (gx.shape[0] != 1 and gx.shape[1] == 1) or (gx.shape[1] != 1 and gx.shape[0] == 1):
        gx = gx.squeeze()
    if gx.ndim == 2 and c1.ndim == 1:
        c1 = c1[:, np.newaxis]
        c2 = c2[:, np.newaxis]
    T = c1*np.exp(-gx)+c2*np.exp(gx)+Tamb
    return T
def SolveSIRinfected(params,y0,tEval):
    if params.ndim==1:
        sol=integrate.solve_ivp(lambda t,y: SIRdydt(params,t,y), np.array([0, np.max(tEval)]),y0,t_eval=tEval)
        infected=sol.y[1,:]
    else:
        infected=np.empty((params.shape[0],tEval.size))
        for i in np.arange(0,params.shape[0]):
            sol = integrate.solve_ivp(lambda t, y: SIRdydt(params[i,], t, y), np.array([0, np.max(tEval)]), y0,
                                      t_eval=tEval)
            infected[i,:]=sol.y[1,:]
    return infected


def SIR_endemic_integrated(params,y0,tEval):
    if params.ndim==1:
        sol=integrate.solve_ivp(lambda t,y: SIRdydt_endemic(params,t,y), np.array([0, np.max(tEval)]),y0,t_eval=tEval)
        recovered=sol.y[2,:]
        tdiff = sol.t[1:] - sol.t[:-1]
        rhr = np.sum(recovered[1:]*tdiff)
        lhr = np.sum(recovered[:-1]*tdiff)
        total_infected = (rhr+lhr)/2
    else:
        infected=np.empty((params.shape[0]))
        for i in np.arange(0,params.shape[0]):
            sol = integrate.solve_ivp(lambda t, y: SIRdydt_endemic(params[i,], t, y), np.array([0, np.max(tEval)]), y0,
                                      t_eval=tEval)
            recovered=sol.y[2,:]
            tdiff = sol.t[1:] - sol.t[:-1]
            rhr = np.sum(recovered[1:]*tdiff)
            lhr = np.sum(recovered[:-1]*tdiff)
            total_infected[i,:] = (rhr+lhr)/2
    return infected

def SIRdydt_endemic(params,t,y):
    dydt=np.empty(3)
    dydt[0] = params[3]*(y[1]+y[2])-params[0]*params[1]*y[0]*y[1]
    dydt[1] = params[0]*params[1]*y[0]*y[1] - (params[3]+params[2])*y[1]
    dydt[2] = params[2]*y[1]-params[3]*y[2]
    return dydt

def SIRdydt(params,t,y):
    dydt=np.empty(3)
    dydt[0]=-params[0]*y[1]/(np.sum(y))*y[0]
    dydt[1]=params[0]*y[1]/(np.sum(y))*y[0]-params[1]*y[1]
    dydt[2]=params[1]*y[1]
    return dydt

def SobolTestFunction(theta,a):
    if theta.ndim==1:
        return np.array([np.prod((np.abs(4*theta-2)+a)/(1+a))])
    else:
        return np.prod((np.abs(4*theta-2)+a)/(1+a), axis=1)