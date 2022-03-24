
# importing sys
import sys
  
# adding Folder_2 to the system path
sys.path.insert(0, '../')
import UQLibrary as uq
#import mpi4py.MPI as MPI

def main():
    # #Set seed for reporducibility
    # #np.random.seed(10)
    
    # # Get model and options object from Example set
    # [model, options] = uqExamples.GetExample('aluminum rod (uniform)')
    #
    # # [model, options] = uqExamples.GetExample('linear product')
    #
    # [model, options] = uqExamples.GetExample('aluminum rod (saltelli normal)')
    #options.plot.nPoints=options.gsa.nSamp
    #f
    # [model, options] = uqExamples.GetExample('aluminum rod (normal)')
    #
    print("Loading Example: Ishigami function with Low Discrepency Normal Distribution")
    # [model, options] = uq.examples.GetExample('ishigami (normal)')
    # options.lsa.run_param_subset = False
    # options.display = True
    # options.save = True
    [model, options] = uq.examples.GetExample('helmholtz')
    options.lsa.subset_rel_tol=.023
    options.lsa.run_lsa = False
    options.lsa.run_param_subset = True
    options.gsa.run = False
    options.display = True
    options.save = True
    #options.plot = True
    #
    # # Run UQ package
    # (baseSobol,totalSobol)= uq.TestAccuracy(model, options, np.arange(start=10000, stop=200000, step=5000))
    results = uq.run_uq(model, options, logging = 0)
    #
    # plt.plot(results.gsa.sampD[:450,0], results.gsa.sampD[:450,1],'rs')
    # plt.plot(results.gsa.sampD[options.gsa.nSamp:options.gsa.nSamp+450,0], results.gsa.sampD[options.gsa.nSamp:options.gsa.nSamp+450,1],'bo')
    # plt.show()
if __name__ == '__main__':
    main()


