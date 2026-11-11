"""Tests for microbial VOC production (VOC_production_mode = 1)."""

import copy

import numpy as np
import pandas as pd
import pytest

from grid import Grid
from initialization import initialize_data
from voc import voc_production


@pytest.fixture
def grid_after_uptake():
    """Grid (grassland setup, VOC production on) after one degradation and uptake step."""
    np.random.seed(1)
    input_dir = "grassland"
    runtime = pd.read_csv(input_dir + "/runtime.txt", header=None, index_col=0, sep="\t")
    runtime.loc["VOC_production_mode", 1] = 1
    grid = Grid(runtime, initialize_data(runtime, input_dir))
    grid.degradation(0)
    grid.uptake(0)
    return grid


def test_voc_production_is_supralinear():
    resp = np.array([0.0, 0.05, 0.1, 0.2])
    voc = voc_production(resp, frac=0.01, resp_ref=0.05, exponent=1.5)
    assert voc[0] == 0
    assert voc[1] == pytest.approx(0.01 * 0.05)  # VOC:CO2 = frac at the reference respiration
    assert voc[2] / voc[1] == pytest.approx(2**1.5, rel=1e-5)
    assert np.all(np.diff(voc[1:] / resp[1:]) > 0)  # VOC:CO2 increases with respiration


def test_voc_production_conserves_carbon(grid_after_uptake):
    grid_off = copy.deepcopy(grid_after_uptake)
    grid_off.VOC_production_mode = 0
    grid_after_uptake.metabolism(0)
    grid_off.metabolism(0)

    produced = grid_after_uptake.VOC_produced.sum()
    by_source = grid_after_uptake.VOC_production
    assert produced > 0
    assert by_source["from_biomass"] > 0 and by_source["from_uptake"] > 0
    assert produced == pytest.approx(by_source["from_biomass"] + by_source["from_uptake"], rel=1e-5)
    assert grid_after_uptake.VOC_production_tax.sum() == pytest.approx(produced, rel=1e-5)
    # microbial C lost to VOC production equals the VOC produced; respiration is unchanged
    lost = grid_off.Microbes["C"].sum() - grid_after_uptake.Microbes["C"].sum()
    assert lost == pytest.approx(produced, rel=1e-4)
    assert grid_after_uptake.Respiration == pytest.approx(grid_off.Respiration)
    assert grid_after_uptake.CUE_system < grid_off.CUE_system


def test_produced_voc_enters_pool_next_day(grid_after_uptake):
    grid = grid_after_uptake
    grid.metabolism(0)
    grid.mortality(0)  # sets VOC monomers to zero (not taken up -> diffused)
    produced = grid.VOC_produced.copy()
    assert produced.sum() > 0
    grid.Microbes[:] = 0  # no uptake, so the pool after uptake() is litter input + microbial VOC
    grid.degradation(1)
    grid.uptake(1)
    voc3_pool = grid.Monomers.loc["VOC3", "C"].values
    litter_input = grid.Input_VOC.loc["VOC3", "C"].values
    np.testing.assert_allclose(voc3_pool, litter_input + produced, rtol=1e-5)
    assert grid.VOC_produced.sum() == 0
    # stoichiometry follows the mixed pool: litter VOC3 (C:N 2:0.04 in grassland) + pure-C microbial VOC
    litter_N = grid.Input_VOC.loc["VOC3", "N"].values
    expected_C_frac = voc3_pool / (voc3_pool + litter_N)
    np.testing.assert_allclose(grid.VOC_ratio["VOC3"]["C"].values, expected_C_frac, rtol=1e-5)
    # more C-rich than the litter VOC3 alone where microbes produced VOC, unchanged elsewhere
    np.testing.assert_allclose(expected_C_frac[produced == 0], 2 / 2.04, rtol=1e-5)
    assert np.all(expected_C_frac[produced > 0] > 2 / 2.04)
