# voc.py module holding classes dealing with VOC (volatile organic compound) substrates
# and monomers.
#
# VOCs represent volatile compounds entrapped within the litter that are released as a
# function of temperature and litter cellulose content (see Grid.degradation()), rather
# than through the regular enzyme-mediated substrate degradation pathway used for the
# other substrates. They therefore have their own input table and classes, kept separate
# from the regular Substrate/Monomer classes.

import numpy as np
import pandas as pd


class VOCSubstrate:
    """
    Deals with properties related to VOC substrates (the VOC pool entrapped in litter).
    """

    def __init__(self, runtime, voc_substrates_init):
        """
        The constructor of VOCSubstrate class.

        Parameters:
            runtime:             dataframe; user-specified parameters when running the model
            voc_substrates_init: dataframe; initial VOC substrate pool sizes (C,N,P); e.g. VOC1,VOC2,VOC3
        """

        self.n_VOC = int(runtime.loc["n_VOC", 1])
        self.gridsize = int(runtime.loc["gridsize", 1])
        self.VOC_Start = voc_substrates_init.astype(
            "float32"
        )  # initial VOC substrate pool concentrations


class VOCMonomer:
    """
    Deals with properties related to VOC monomers.
    """

    def __init__(self, runtime):
        """
        The constructor of VOCMonomer class.

        Parameters:
            runtime: dataframe; user-provided model setup parameters
        """

        self.n_VOC = int(runtime.loc["n_VOC", 1])

    def voc_monomer_initialization(self, voc_index):
        """
        Derive the initial pool of VOC monomers.

        VOC monomers start out empty; they are only replenished over time as VOCs are
        released from the VOC substrate pool (see Grid.uptake()).

        Parameters:
            voc_index: index; names of the VOC monomers, matching the VOC substrates they
                       correspond to (e.g. VOC1, VOC2, VOC3)
        Return:
            VOCMonomers_df: dataframe (n_VOC*3); columns: C,N,P
        """

        VOCMonomers_df = pd.DataFrame(
            data=np.zeros((self.n_VOC, 3), dtype="float32"),
            index=voc_index,
            columns=["C", "N", "P"],
            dtype="float32",
        )

        return VOCMonomers_df
