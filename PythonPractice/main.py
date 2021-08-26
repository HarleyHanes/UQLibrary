

from SALib.sample import saltelli
from SALib.analyze import sobol
from SALib.test_functions import Ishigami
import UQtoolbox as uq
import UQtoolbox_examples as uqExamples
import numpy as np
import matplotlib.pyplot as plt

def main():
    # #Set seed for reporducibility
    # #np.random.seed(10)
    # #
    # # Define the model inputs
    # problem = {
    #     'num_vars': 2,
    #     'names': ['Phi', 'h'],
    #     'bounds': [[-18.4-(.1450*np.sqrt(3)), -18.4+(.1450*np.sqrt(3))],\
    #              [.00191-(1.4482*(10**(-5))*np.sqrt(3)), .00191+(1.4482*(10**(-5))*np.sqrt(3))]]
    # }
    #
    # # # Generate samples
    # param_values = saltelli.sample(problem, 2**13)
    # # # Run model (example)
    # Y = uqExamples.HeatRod(param_values, np.array([55]))
    # #
    # # # Perform analysis
    # Si = sobol.analyze(problem, Y, print_to_console=True)
    # #
    # # # Print the first-order sensitivity indices
    # print(Si['S1'])


    # # Get model and options object from Example set
    [model, options] = uqExamples.GetExample('helmholtz')
    options.gsa.run = False
    #
    # # [model, options] = uqExamples.GetExample('linear product')
    #
    # [model, options] = uqExamples.GetExample('aluminum rod (saltelli normal)')
    #options.plot.nPoints=options.gsa.nSamp
    #f
    # [model, options] = uqExamples.GetExample('aluminum rod (normal)')
    #
    # [model, options] = uqExamples.GetExample('trial function')
    #
    # # Run UQ package
    # (baseSobol,totalSobol)= uq.TestAccuracy(model, options, np.arange(start=10000, stop=200000, step=5000))
    results = uq.RunUQ(model, options)
    #
    # plt.plot(results.gsa.sampD[:450,0], results.gsa.sampD[:450,1],'rs')
    # plt.plot(results.gsa.sampD[options.gsa.nSamp:options.gsa.nSamp+450,0], results.gsa.sampD[options.gsa.nSamp:options.gsa.nSamp+450,1],'bo')
    # plt.show()
if __name__ == '__main__':
    main()


