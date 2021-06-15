

#from SALib.sample import saltelli
#from SALib.analyze import sobol
#from SALib.test_functions import Ishigami
import UQtoolbox as uq
import UQtoolbox_examples as uqExamples
import numpy as np

def main():
    #Set seed for reporducibility
    np.random.seed(10)
    #
    # # Define the model inputs
    # problem = {
    #     'num_vars': 3,
    #     'names': ['x1', 'x2', 'x3'],
    #     'bounds': [[-3.14159265359, 3.14159265359],
    #                [-3.14159265359, 3.14159265359],
    #                [-3.14159265359, 3.14159265359]]
    # }
    #
    # # Generate samples
    # param_values = saltelli.sample(problem, 512*2)
    #
    # # Run model (example)
    # Y = Ishigami.evaluate(param_values)
    #
    # # Perform analysis
    # Si = sobol.analyze(problem, Y, print_to_console=True)
    #
    # # Print the first-order sensitivity indices
    # print(Si['S1'])


    # # Get model and options object from Example set
    # # [model, options] = uqExamples.GetExample('ishigami')
    #
    # # [model, options] = uqExamples.GetExample('linear product')
    #
    #[model, options] = uqExamples.GetExample('integrated helmholtz')
    #f
    [model, options] = uqExamples.GetExample('linear')
    #
    # [model, options] = uqExamples.GetExample('trial function')
    #
    # # Run UQ package
    results = uq.RunUQ(model, options)

if __name__ == '__main__':
    main()
