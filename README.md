# UQLibrary Course Project README
*UQLibrary* is a package for sensitivity and identifiability analysis using methods such as parameter subset, active subspace, Morris, and Sobol analyses.

## Installation
*UQLibrary* can be installed from pip using the command,

```
pip install UQLibrary
```

Required dependencies are *numpy*, *scipy*, *mpi4py*, *matplotlib*, and *tabulate*.

## Running *UQLibrary*

*UQLibrary* is designed to be imported as a module to a script where the required objects can be built to run its sensitivity and identifiability analysis functions. However, an example implementation of *UQLibrary* can be run using the command,

```
python3 -m UQLibrary
```

The source code for this example can be found in the *__main __.py* file of *UQLibrary*.

All methods of *UQLibrary* can be run collectively through the function *UQLibrary.run_uq* which requires as inputs variables of class *UQLibrary.Model* and *UQLibrary.Options*. Constructing a variable of *UQLibrary.Model* requires providing *base_poi*, a  one-dimensional numpy array of nominal parameter vaules, and *eval_fcn*, a function which inputs a *numpy* array of parameter values and outputs a *numpy* array of quantities of interest. For local sensitivity analysis methods, these arrays can be one dimensional, but for global sensitivity analysis methods they must be able to be two-dimensional such that the first dimension is the number of model evaluations to perform and the second dimension is the set of parameters/ outputs at each evaluation. *UQLibrary.Model* also automatically generates arrays of parameter and quantity names along with parameter sampling distributions but these can be manually set using keyword arguments.


*UQLibrary.Options* contains specifications on which sensitivity and identifiability analysis methods to run, required hyper-parameters, and how to display and save results. All components of *UQLibrary.Options* are automatically generated at initialization but can be specified by the user using keyword arguments. Examples of how to formulate both these variables can be found in *UQLibrary.examples*. *UQLibrary.run_uq* also takes the keyword argument *logging* which determines which intermediate messages to print where 0 or False is none, 1 or True is print when each method or function begins running, and integers 2 or greater is to print when methods or function begin running and results of intermediate steps.

## Interpreting *UQLibrary* Results

Running *UQLibrary.run_uq* outputs a variable of class *UQLibrary.Results* which contains *numpy* arrays of all results from methods selected in *UQLibrary.Options* such as local sensitivity indices, active and inactive subspaces, Sobol indices, and Morris indices. These results are grouped under two frields of *UQLibrary.Results*, *lsa* and *gsa*, according to which submodule the corresponding method is in. *UQLibrary.run_uq* can also print tables of the results to either the command line or a "Results.txt." When Sobol analysi is used, *UQLibrary.run_uq* can also construct scatter plots of the parameter samples and their correlation to each output quantity. Whether each of these methods is used to display results is detailed in *UQLibrary.Options*

functions to a project requires formulating a function which inputs a *numpy* array of parameter values and outputs a *numpy* array of quantities of interest. For local sensitivity analysis methods, these arrays can be one dimensional, but for global sensitivity analysis methods they must be able to be two-dimensional such that the first dimension is the number of model evaluations to perform and the second dimension is the set of parameters/ outputs at each evaluation. Examples of how to formulate these functions can be found in *UQLibrary.examples*.


## Example Case: Ishigami function

To showcase running *UQLibrary* we consider local sensitivity and Sobol analysis of the Ishigami function,

f(x) =  sin(x<sub>1</sub>) + a sin(x<sub>2</sub>)<sup>2</sup>+b x<sub>3</sub><sup>4</sup>sin(x<sub>1</sub>).

In *UQLibrary.examples* we wrote a function *Ishigami* which computes f(x) for at a=7 and b=0.1. We additionally assume that that x<sub>1</sub>, x<sub>2</sub>, and x<sub>3</sub> have base values of 0 and are all uniformly distributed in [-pi, pi] and that we sample using Saltelli low-discrepency samples. To construct the corresponding object of class *UQLibrary.Model* we call,

```
model = UQLibrary.Model(eval_fcn=Ishigami,
                         base_poi=np.array([0, 0, 0]),
                         dist_type="saltelli uniform",
                         dist_param=np.array([[-math.pi, -math.pi, -math.pi],\
                              [math.pi, math.pi, math.pi]]))
```

We next aim to set our run settings using the *UQLibrary.Options* class but want to specify that we only use local sensitivity analysis and Sobol analysis. We also want to specify we use finite difference derivative approximations with step-size h=10<sup>-6</sup> and use 100000 samples for Sobol analysis. To construct the corresponding variable we call,

```
options = UQLibrary.Options()
options.lsa.method = 'finite'
options.lsa.xDelta = 10**(-6)
options.gsa.nSampSobol = 100000          # Use default number of samples
options.path='..\\Figures\\Ishigami(uniform)'
```

Now, having constructed our required variables, we run

```
results = UQLibrary.run_uq(model, options, logging =0)
```

which prints to command line and saves in "Results.txt" the following results for sensitivity analysis.

```
Base POI Values
  POI0    POI1    POI2
------  ------  ------
     0       0       0

 Base QOI Values
  POI0
------
     0

 Sensitivity Indices
        POI0
----  ------
POI0       0
POI1       0
POI2       0

 Relative Sensitivity Indices
        POI0
----  ------
POI0       0
POI1       0
POI2       0

 Sobol Indices for POI0
        1st Order    Total Sensitivity
----  -----------  -------------------
POI0    0.126062              0.906936
POI1    0.0862534             0.085604
POI2   -0.0172513             0.776759

 Morris Screening Results for POI0
        mu_star    sigma
----  ---------  -------
POI0    2.56189  2.91683
POI1    3.39627  3.01568
POI2    1.86548  2.53656

 Base POI Values
  POI0    POI1    POI2
------  ------  ------
     0       0       0

 Base QOI Values
  POI0
------
     0

 Sensitivity Indices
        POI0
----  ------
POI0       0
POI1       0
POI2       0

 Relative Sensitivity Indices
        POI0
----  ------
POI0       0
POI1       0
POI2       0

 Sobol Indices for POI0
        1st Order    Total Sensitivity
----  -----------  -------------------
POI0   0.124831              0.923706
POI1   0.0859534             0.0852085
POI2  -0.00552825            0.791928

 Morris Screening Results for POI0
        mu_star     sigma
----  ---------  --------
POI0   0.574754  0.591535
POI1   2.56448   2.97578
POI2   0.188533  0.222611

 Base POI Values
  POI0    POI1    POI2
------  ------  ------
     0       0       0

 Base QOI Values
  POI0
------
     0

 Sensitivity Indices
        POI0
----  ------
POI0       0
POI1       0
POI2       0

 Relative Sensitivity Indices
        POI0
----  ------
POI0       0
POI1       0
POI2       0

 Sobol Indices for POI0
        1st Order    Total Sensitivity
----  -----------  -------------------
POI0    0.124298             0.911996
POI1    0.0838929            0.0844483
POI2   -0.0378661            0.793264

 Morris Screening Results for POI0
        mu_star     sigma
----  ---------  --------
POI0   0.846896  0.666396
POI1   4.69346   5.0475
POI2   0.27566   0.358165

```

<!-- Results show that local sensitivities of all parameters are zero, which is expected since $\sin(x)$ and $x^4$ have zero derivatives at $x=0$. Additionally, Sobol analysis identifies $x_1$ as the most sensitive in both first order and total metrics while Morris screening identifies $x_2$ as the most sensitive parameter. -->



<!--
## Module Guide
UQLibrary is seperated into three modules; *UQtoolbox*, *lsa*, and *gsa*.

### *UQtoolbox*

The *UQtoolbox* model contains functions *run_uq* for running all package modules and the functions *print_results* and *plot_gsa* for printing, saving, and plotting data results in standardize format. The *UQtoolbox* module defines the three primary classes used to interact with UQLibrary; *model, options,* and *results*. An object of the *model* class holds all information about the system UQLibrary is testing. Fields that are always required are; *evalFcn*, the function mapping from the parameters of interest (POIs) to the quantities of interestes (QOIs), and *basePOIs*, the parameters sensitivity analysis is focused on. If using global methods, *paramDist*, the sampling distributions of each parameter, are required but, if no distributions are provided, *UQLibrary* assumes uniform distributions &pm20;% about the *basePOIs* values. The *options* class holds the subclasses *lsaOptions* and *gsaOptions*, both defined in their respective modules, along with whether to display or plot results and the locations to save results to. The *results* class is the output of the *RunUQ* function and holds the subclasses *lsaResults* and *gsaResults*, both defined in their respective modules.

### *lsa*

The *lsa* module -->
