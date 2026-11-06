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

    def __init__(self, runtime, parameters, voc_attributes):
        """
        The constructor of VOCMonomer class.

        Parameters:
            runtime:        dataframe; user-provided model setup parameters
            parameters:     dataframe; model parameters
            voc_attributes: dataframe; molecular properties of each VOC (rows: VOC1,VOC2,...);
                            columns: MW (molecular weight; g mol-1), nC (number of C atoms per molecule)
        """

        self.n_VOC = int(runtime.loc["n_VOC", 1])
        self.MW_ref = parameters.loc[
            "VOC_MW_ref", 1
        ]  # Reference molecular weight at which VOC transporter kinetics are not scaled
        self.beta_Vmax = parameters.loc[
            "VOC_beta_Vmax", 1
        ]  # Exponent of the MW scaling of VOC uptake Vmax (1: constant molar turnover)
        self.beta_Km = parameters.loc[
            "VOC_beta_Km", 1
        ]  # Exponent of the MW scaling of VOC uptake Km (>beta_Vmax: higher affinity of small VOCs)
        self.AE_nC = parameters.loc[
            "VOC_AE_nC", 1
        ]  # AE reduction scaled by 1/nC (transport cost per molecule spread over fewer C)
        self.AE_C1 = parameters.loc[
            "VOC_AE_C1", 1
        ]  # Additional AE reduction for C1 compounds (inefficient C1 assimilation pathways)
        self.Uptake_Km_min = parameters.loc["Uptake_Km_min", 1]  # Minimum uptake Km

        if voc_attributes[["MW", "nC"]].isna().any().any():
            raise ValueError(
                "VOC_attributes.csv must provide MW and nC for every VOC in initial_VOC_substrates.csv"
            )
        if (voc_attributes["MW"] <= 0).any() or (voc_attributes["nC"] < 1).any():
            raise ValueError("VOC_attributes.csv must satisfy MW > 0 and nC >= 1")
        self.MW = voc_attributes["MW"].astype("float32")
        self.nC = voc_attributes["nC"].astype("float32")

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

    def voc_uptake_kinetics(self, Uptake_Vmax0, Uptake_Km0):
        """
        Scale the transporter kinetics of VOC monomers with their molecular weight.

        Vmax and Km are mass-based, so a transporter with constant molar turnover moves less
        mass of a small molecule (Vmax ~ MW^beta_Vmax). Small molecules also diffuse faster to
        the cell, lowering the apparent Km (Km ~ MW^beta_Km). With beta_Km > beta_Vmax, small
        VOCs have a higher specific affinity (Vmax/Km) but a lower mass-based uptake at saturation.

        Parameters:
            Uptake_Vmax0: dataframe; rows: monomers (VOCs last); cols: uptake enzymes
            Uptake_Km0:   dataframe; rows: monomers (VOCs last); cols: uptake enzymes
        Return:
            Uptake_Vmax0, Uptake_Km0: dataframes; VOC rows scaled
        """

        Uptake_Vmax0 = Uptake_Vmax0.copy()
        Uptake_Km0 = Uptake_Km0.copy()
        if self.n_VOC == 0:
            return Uptake_Vmax0, Uptake_Km0

        size_ratio = (self.MW / self.MW_ref).to_numpy(dtype="float32")[:, None]
        Uptake_Vmax0.iloc[-self.n_VOC :] *= size_ratio**self.beta_Vmax
        Km_VOC = Uptake_Km0.iloc[-self.n_VOC :] * size_ratio**self.beta_Km
        Uptake_Km0.iloc[-self.n_VOC :] = Km_VOC.clip(lower=self.Uptake_Km_min)

        return Uptake_Vmax0.astype("float32"), Uptake_Km0.astype("float32")

    def voc_AE_penalty(self, monomer_index):
        """
        Derive the monomer-specific reduction of assimilation efficiency (AE).

        Each VOC loses AE_nC/nC (energy cost of transport per molecule, spread over fewer C atoms
        in small molecules); C1 compounds lose an additional AE_C1 for their inefficient
        assimilation pathways. Non-VOC monomers are not penalized.

        Parameters:
            monomer_index: index; names of all monomers (incl. VOCs)
        Return:
            AE_penalty: series; index: monomers
        """

        AE_penalty = pd.Series(data=0, index=monomer_index, dtype="float32")
        penalty_VOC = self.AE_nC / self.nC + self.AE_C1 * (self.nC == 1)
        AE_penalty.loc[self.MW.index] = penalty_VOC.to_numpy(dtype="float32")

        return AE_penalty
