import sys
import io
import os
import UQtoolbox as uq
import aerofusion.data.array_conversion as arr_conv
from aerofusion.rom import incompressible_navier_stokes_rom as incrom
import numpy as np
import logging
import argparse
import libconf
from aerofusion.plot.plot_2D import plot_contour
from aerofusion.plot.plot_2D import plot_pcolormesh
import matplotlib.pyplot as plt

def main(argv=None):
    #Variables to load
    num_modes=np.array([2, 3, 4, 5,6,7,8,9,10,15,20,25,30,40,50])
    fileprefix = 'sensitivities/morris_mu'
    distance_fig_name='plots/sens_distance.png'
    #Loop through modes
    sensitivities=()
    for i_modes in range(len(num_modes)):
        #Load sensitivity
        i_sens=np.load(fileprefix + '_m' + str(i_modes)+ '.npz')
        #Initialize sensitivities vector
        #Store sensitivities
        sensitivities.append(i_sens)
    #Caclulate Distances
    sens_distance=np.mean((sensitivities[:]**2)-(sensitivities[-1]**2))/np.mean((sensitivities**2)[-1])
    plt.plot(num_modes, sens_distance)
    plt.xlabel('Number of Modes')
    plt.ylabel('Relative MSE')
    plt.savefig(distance_fig_name)

if __name__ == "__main__":
    sys.exit(main())