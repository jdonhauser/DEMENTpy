import os
import sys

import numpy  as np
import pandas as pd


os.chdir('d:/postdocCopenhagen/VOC_model/DEMENT_VOC_v1/DEMENTpy-iss22-outfile-refactor/src')
sys.path.append('d:/postdocCopenhagen/VOC_model/DEMENT_VOC_v1/DEMENTpy-iss22-outfile-refactor/src')

# make other .py files in src/ available. 
from initialization import initialize_data
from grid import Grid
from output import Output
from utility import export

# original script: passes command line arguments
    # #...Obtain the command line arguments
    # input_folder  = sys.argv[1]   # input folder name; e.g., a study site or a project name
    # output_folder = sys.argv[2]   # output folder name
    # outname       = sys.argv[3]   # output file name and seed of Pseudo-RNG

input_folder  = 'd:/postdocCopenhagen/VOC_model/DEMENT_VOC_v1/DEMENTpy-iss22-outfile-refactor/input'

# input folder name; e.g., a study site or a project name
# create output folder
os.makedirs('d:/postdocCopenhagen/VOC_model/DEMENT_VOC_v1/DEMENTpy-iss22-outfile-refactor/output' , exist_ok=True)
output_folder = 'd:/postdocCopenhagen/VOC_model/DEMENT_VOC_v1/DEMENTpy-iss22-outfile-refactor/output'   # output folder name
outname       = '2026100901'   # prefix for the output file (use date plus index for #of run) at the same time used as rngseed of Pseudo-RNG
site = input_folder       # no forcing and/or parameter switching


    #...Set up the working directory (change to input dir)
os.chdir(input_folder)

    #...seed the pseudorandom number generator
np.random.seed(int(outname[:]))

#...a few system constants
runtime    = pd.read_csv('runtime.txt',header=None,index_col=0,sep='\t')
pulse      = int(runtime.loc['pulse',1])         # number of pulses
cycle      = int(runtime.loc['end_time',1])      # number of time steps in each pulse
interval   = int(runtime.loc['interval',1])      # interval of time step to record outputs
# Export_format = int(runtime.loc['output_CSV',1])      # interval of time step to record outputs
mode       = int(runtime.loc['dispersal',1])     # 0:'default' or 1:'dispersal'

#...Initialize data by calling the Function: Initialize_Data() 
data_initialization = initialize_data(runtime,'.') # starting site
# creates data frames with initial values

#...Prepare for output by creating an instance of the Output class
Output_init = Output(runtime,data_initialization)

#...Create an instance of the Grid class
Ecosystem = Grid(runtime,data_initialization)

# ...Run the model
for p in range(pulse):

    for i in range(cycle):

        # substrates degradation
        Ecosystem.degradation(i)

        # monomers uptake
        Ecosystem.uptake(i)

        # microbial metabolism
        Ecosystem.metabolism(i)

        # microbial death
        Ecosystem.mortality(i)

        # microbial reproduction and dispersal
        Ecosystem.reproduction(i)

        # output basic data using the "output" method in the Output class
        if i == 0:
            Output_init.output(Ecosystem, p, i)  # day 1
        elif i % interval == interval - 1:
            Output_init.output(Ecosystem, p, i)  # interval
        # output microibal allocation data using the "microbes_tradeoff" method
        Output_init.microbes_tradeoff(Ecosystem, p, i)
        # output microbial mass of every iteration using the "microbes_abundance" method
        Output_init.microbes_abundance(Ecosystem, p, i)

    # re-initialize microbial community in each new pulse
    if site == input_folder:
        Ecosystem.reinitialization(data_initialization, data_initialization['Microbes_pp'], Output_init, mode, p)
    else:
        switch = int(runtime.loc['switch', 1])  # the pulse from which inputs to be changed onwards
        data_switch = initialize_data(runtime, site)  # switching site
        if p < switch - 1:
            # stick to the starting site
            Ecosystem.reinitialization(data_initialization, data_initialization['Microbes_pp'], Output_init, mode, p,
                                       switch)
        else:
            # switch to data of another site
            Ecosystem.reinitialization(data_switch, data_initialization['Microbes_pp'], Output_init, mode, p, switch)


#...export the Output_init object to the output_folder using the export() funtion in the utility module 
os.chdir(output_folder) # not needed if site is an absolute path
Output_init.export_to_csv(outname)