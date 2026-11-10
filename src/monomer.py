# This module, monomer.py, handles monomer-related properties in a class, Monomer().
# and leaching of monomers with a function, monomer_leaching()

import warnings

import numpy as np
import pandas as pd


class Monomer:
    """
    This class deals with all calculations closely related to monomers.

    Methods include:
      1.monomer_initialization():   determine the initalized monomoers in the system.
      2.monomer_input_rate():       specify input rates of monomers during simulation.
      3.monomer_uptake_reqenzyme(): derive the monomer-required uptake enzymes/transporters.
      4.monomer_MW():               assign the (conceptual) molecular weight of each monomer.
      5.monomer_uptake_kinetics():  MW-based scaling of uptake Vmax and Km (MW_mode).
      6.monomer_AE_penalty():       MW-based reduction of assimilation efficiency (MW_mode, C1_penalty_mode).
    """

    def __init__(self, runtime, parameters):
        """
        The constructor for Monomoer class.

        Parameters:
             runtime:    dataframe; user-provided model setup parameters
             parameters: dataframe; model parameters
        """

        self.n_VOC       = int(runtime.loc['n_VOC',1])                             # Number of VOC monomer types (tracked by the VOCMonomer class)
        self.n_monomers  = int(runtime.loc['n_substrates',1]) + 2 + self.n_VOC     # Number of monomers within the system (organic + NH4 + PO4 + VOCs)
        self.n_uptake    = int(runtime.loc['n_uptake',1])                          # Number of uptake transporters for each taxon
        self.Monomer_Substrate_Ratio = runtime.loc['Monomer_Substrate_Ratio',1]    # amount of initial monomer relative to substrate per grid box; default:0
        self.Uptake_per_monomer      = int(runtime.loc['Uptake_per_monomer',1])    # number of transporters per monomer;default:1
        self.Init_NH4  = runtime.loc['Init_NH4',1]                                 # Initial NH4
        self.Init_PO4  = runtime.loc['Init_PO4',1]                                 # Initial PO4
        self.Input_NH4 = runtime.loc['Input_NH4',1]                                # Input of NH4
        self.Input_PO4 = runtime.loc['Input_PO4',1]                                # Input of PO4
        self.MW_mode   = int(runtime.loc['MW_mode',1])                             # 1: MW-based uptake kinetics and AE penalty for all monomers
        self.C1_penalty_mode = int(runtime.loc['C1_penalty_mode',1])               # 1: only the C1 AE penalty (MW < MW_C1); no other MW adjustment
        if self.MW_mode == 1 and self.C1_penalty_mode == 1:
            warnings.warn("runtime: MW_mode and C1_penalty_mode both 1; using MW_mode = 1, C1_penalty_mode = 0")
            self.C1_penalty_mode = 0
        self.MW_ref    = parameters.loc['MW_ref',1]                                # Base molecular weight: no adjustment of uptake kinetics
        self.beta_Vmax = parameters.loc['MW_beta_Vmax',1]                          # Exponent of the MW scaling of uptake Vmax (0: no scaling)
        self.beta_Km   = parameters.loc['MW_beta_Km',1]                            # Exponent of the MW scaling of uptake Km (1: molar concentration)
        self.AE_MW     = parameters.loc['AE_MW_penalty',1]                         # AE reduction at MW_ref, scaled by MW_ref/MW for all monomers
        self.MW_C1     = parameters.loc['MW_C1',1]                                 # MW below which a monomer counts as a C1 compound
        self.AE_C1     = parameters.loc['AE_C1_penalty',1]                         # Additional AE reduction for C1 compounds (inefficient C1 assimilation pathways)
        self.MW        = None                                                      # Molecular weight of each monomer; set by monomer_MW()


    def monomer_initialization(self,substrates_init):
        """
        Derive the initial pool of (non-VOC) monomers.

        NOTE: VOC monomers are initialized separately by the VOCMonomer class and are
        expected to be concatenated onto the dataframe returned here (see initialization.py).

        Parameters:
            substrates_init: dataframe; initial substrates pool (without VOCs); from the substrate module
        Return:
            Monomers_df: dataframe (shape: (n_substrates+2)*3)
        """

        # Monomer pool sizes for all elements
        Monomers_array = np.concatenate((np.stack([[0,self.Init_NH4,0],[0,0,self.Init_PO4]]),substrates_init.values*self.Monomer_Substrate_Ratio),axis=0)    
        n_organic = len(substrates_init.index)
        index = ["NH4","PO4","DeadMic","DeadEnz"] + ["Mon" + str(i) for i in range(3,n_organic + 1)]  #NOTE: index starts from 3
        Monomers_df = pd.DataFrame(data=Monomers_array, index=index, columns=["C","N","P"],dtype='float32')
        
        return Monomers_df

    def monomer_ratios(self, Monomers_df):
        """
        Initialization of monomer_ratio.

        Parameter:
            Monomers_df:   initialized pool of monomers; derived from the above method
        Return:
            Monomer_ratios: dataframe; row: monomers; column: C,N,P
        '"""

        is_NH4 = Monomers_df.index == "NH4"
        is_PO4 = Monomers_df.index == "PO4"
        Monomer_ratios = Monomers_df.copy(deep=True)
        Monomer_ratios[:] = 0.0
        Monomer_ratios.loc[is_NH4, "N"] = 1.0
        Monomer_ratios.loc[is_PO4, "P"] = 1.0

        return Monomer_ratios

    def monomer_input_rate(self, sub_mon_input):
        """
        Derive the monomer input rates.

        Parameters:
          sub_mon_input: dataframe; loaded as an input file
        Return:
          MonInput: series; index: monomers
        """

        # load the inputs without NH4 and PO4
        monomer_input = sub_mon_input["Mon"]
        # supplement with NH4 and PO4 from the parameters
        MonInput = pd.concat(
            [
                pd.Series(
                    [self.Input_NH4, self.Input_PO4], index=["Input_NH4", "Input_PO4"]
                ),
                monomer_input,
            ],
            sort=False,
        )
        MonInput = MonInput.astype("float32")

        return MonInput

    def monomer_uptake_reqenzyme(self):
        """
        Derive the monomer-required enzymes.

        Make sure each monomer is taken up by at least one transporter and every transporter takes up at least one monomer
        Same number within a row implies redundancy

        VOC monomers (the last n_VOC rows) can only be taken up by the VOC transporters (the last
        n_VOC columns, matching the VOC1..VOCn genes appended in Microbe.microbe_uptake_gene()),
        and VOC transporters cannot take up any non-VOC monomer.
        ------------------------------------
        Return:
          Uptake_ReqEnz_df: dataframe; Rows-monomers;cols-uptake enzymes
        """

        n_mon_nonVOC = self.n_monomers - self.n_VOC
        n_upt_nonVOC = self.n_uptake - self.n_VOC

        def assign_block(n_mon, n_upt):
            # each monomer gets Uptake_per_monomer transporters (capped at n_upt) from this block
            n_per_monomer = min(self.Uptake_per_monomer, n_upt)
            probability_list = [0] * n_upt
            probability_list[0:n_per_monomer] = [1] * n_per_monomer
            block = np.array(
                [np.random.choice(probability_list, n_upt, replace=False) for i in range(n_mon)],
                dtype="int8",
            ).reshape(n_mon, n_upt)
            # ensure every transporter takes up at least one monomer of this block
            probability_list = [0] * n_mon
            probability_list[0] = 1
            for i in range(n_upt):
                if block[:, i].sum() == 0:
                    block[:, i] = np.random.choice(probability_list, n_mon, replace=False)
            return block

        Uptake_ReqEnz_array = np.zeros((self.n_monomers, self.n_uptake), dtype="int8")
        Uptake_ReqEnz_array[:n_mon_nonVOC, :n_upt_nonVOC] = assign_block(n_mon_nonVOC, n_upt_nonVOC)
        if self.n_VOC > 0:
            Uptake_ReqEnz_array[n_mon_nonVOC:, n_upt_nonVOC:] = assign_block(self.n_VOC, self.n_VOC)

        index = ["Mon" + str(i) for i in range(1, self.n_monomers + 1)]
        columns = ["Upt" + str(i) for i in range(1, self.n_uptake + 1)]
        Uptake_ReqEnz_df = pd.DataFrame(
            data=Uptake_ReqEnz_array,
            index=index,
            columns=columns,
            dtype="int8",
        )

        return Uptake_ReqEnz_df

    def monomer_MW(self, monomer_attributes, monomer_index):
        """
        Assign the molecular weight (MW) of each monomer.

        MW is a conceptual molecule property rather than the real molecular weight: uptake kinetics
        of monomers at MW_ref are not adjusted, smaller monomers are taken up faster but assimilated
        less efficiently (see monomer_uptake_kinetics() and monomer_AE_penalty()).

        Parameters:
            monomer_attributes: dataframe; rows: monomers (NH4,PO4,DeadMic,DeadEnz,Mon3,...,VOC1,...); column: MW
            monomer_index:      index; names of all monomers (incl. VOCs) in pool order
        Return:
            MW: series; index: monomers
        """

        MW = monomer_attributes["MW"].reindex(monomer_index)
        if MW.isna().any():
            raise ValueError(
                "monomer_attributes.csv must provide MW for every monomer; missing: "
                + ", ".join(MW.index[MW.isna()])
            )
        if (MW <= 0).any():
            raise ValueError("monomer_attributes.csv must satisfy MW > 0")
        self.MW = MW.astype("float32")

        return self.MW

    def monomer_uptake_kinetics(self, Uptake_Vmax0, Uptake_Km0):
        """
        Scale the uptake kinetics with molecular weight (MW_mode = 1).

        Vmax*m^beta_Vmax and Km*m^beta_Km with m = MW/MW_ref; the betas set the strength of the
        scaling. beta_Vmax = 0 and beta_Km = 1 convert the mass-based concentration S to molar
        units, Vmax*(S/m)/(Km + S/m): a monomer with lower MW supplies more molecules per unit mass
        and is taken up faster below saturation. beta_Vmax = beta_Km = 1 corresponds to fully molar
        kinetics with a constant molar turnover (less mass of small monomers taken up at saturation).
        Vmax and Km keep their original values at MW = MW_ref.

        Parameters:
            Uptake_Vmax0: dataframe; rows: monomers in pool order; cols: uptake enzymes
            Uptake_Km0:   dataframe; rows: monomers in pool order; cols: uptake enzymes
        Return:
            Uptake_Vmax0, Uptake_Km0: dataframes; scaled when MW_mode = 1
        """

        if self.MW_mode != 1:
            return Uptake_Vmax0, Uptake_Km0

        size_ratio = (self.MW / self.MW_ref).to_numpy(dtype="float32")[:, None]
        Uptake_Vmax0 = Uptake_Vmax0 * size_ratio**self.beta_Vmax
        Uptake_Km0 = Uptake_Km0 * size_ratio**self.beta_Km

        return Uptake_Vmax0.astype("float32"), Uptake_Km0.astype("float32")

    def monomer_AE_penalty(self):
        """
        Derive the monomer-specific reduction of assimilation efficiency (AE).

        MW_mode = 1:         AE_MW/(MW/MW_ref) + AE_C1*[MW < MW_C1]; the first term is the transport
                             cost per molecule, spread over less mass in small monomers (AE_MW at MW_ref).
        C1_penalty_mode = 1: AE_C1*[MW < MW_C1] only.
        Otherwise:           no penalty.

        Return:
            AE_penalty: series; index: monomers
        """

        AE_penalty = pd.Series(data=0, index=self.MW.index, dtype="float32")
        if self.MW_mode == 1:
            AE_penalty += self.AE_MW / (self.MW / self.MW_ref)
        if self.MW_mode == 1 or self.C1_penalty_mode == 1:
            AE_penalty += self.AE_C1 * (self.MW < self.MW_C1)

        return AE_penalty.astype("float32")


def monomer_leaching(Psi):
    """
    Quantify leaching of monomers.

    Parameters:
        Psi:  scalar; water potential
    Return:
        rate: scalar; leaching rate of monomers
    """

    # Constants
    Leaching = np.float32(0.1)  # Abiotic monomer loss rate
    Psi_slope_leach = np.float32(0.1)  # Mositure sensivity of abiotic monomer loss rate

    rate = Leaching * np.exp(Psi_slope_leach * Psi)

    return rate
