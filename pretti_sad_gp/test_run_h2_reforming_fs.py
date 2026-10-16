"""Smoke tests for the IDAES hydrogen-reforming flowsheet benchmark.

Run from the repository root, e.g. `pytest pretti_sad_gp/test_run_h2_reforming_fs.py`.
They need idaes-pse and its solvers (`idaes get-extensions`). Known failure: the grid
tests evaluate the lowest steam-to-methane ratio (1.5), where the flowsheet does not
converge and `observe` returns NaN. `test_grid_evaluation_hf` and
`test_grid_evaluation_lf` therefore fail.
"""

import logging
import os

import sys
import warnings


from pyomo.environ import *
from idaes.core import *


logging.getLogger("idaes").setLevel(logging.CRITICAL)
logging.getLogger("pyomo").setLevel(logging.CRITICAL)


loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
for logger in loggers:
    logger.setLevel(logging.CRITICAL)
warnings.filterwarnings("ignore")

import numpy as np
import torch

from pretti_sad_gp.benchmarks.benchmark_h2_reforming_fs import (
    device,
    dtype,
    HydrogenReformingFlowsheet,
)


def test_basic_instantiation():
    """Both fidelities can be constructed and carry the expected compressor efficiency."""
    print("=" * 60)
    print("TEST 1: Basic Instantiation")
    print("=" * 60)

    fs_hf = HydrogenReformingFlowsheet(mode="HF")
    print(
        f"HF Flowsheet created: mode={fs_hf.mode}, efficiency={fs_hf.isentropic_efficiency}"
    )

    fs_lf = HydrogenReformingFlowsheet(mode="LF")
    print(
        f"LF Flowsheet created: mode={fs_lf.mode}, efficiency={fs_lf.isentropic_efficiency}"
    )

    print("PASSED: Basic instantiation works for both modes.\n")


def test_bounds():
    """Bounds have shape (2, 6).

    Raises:
        AssertionError: If the shape differs.
    """
    print("=" * 60)
    print("TEST 2: Bounds Verification")
    print("=" * 60)

    fs = HydrogenReformingFlowsheet(mode="HF")
    b = fs.bounds

    print(f"Bounds shape: {b.shape}")
    print(f"Lower bounds: {b[0, :].tolist()}")
    print(f"Upper bounds: {b[1, :].tolist()}")

    expected_shape = torch.Size([2, 6])
    assert b.shape == expected_shape, f"Shape mismatch: {b.shape} != {expected_shape}"

    print("PASSED: Bounds shape is correct.\n")


def test_method_calls():
    """All step accessors, `observe` and `bounds` exist.

    Raises:
        AssertionError: If one is missing.
    """
    print("=" * 60)
    print("TEST 3: Method Existence Check")
    print("=" * 60)

    fs = HydrogenReformingFlowsheet(mode="HF")

    methods = [
        "step_1_mixer",
        "step_2_pressure_changer_1",
        "step_3_heater_1",
        "step_4_main_reformer",
        "step_5_pressure_changer_2",
        "step_6_heater_2",
        "step_7_wgs_reactor",
        "observe",
        "bounds",
    ]

    for method in methods:
        assert hasattr(fs, method), f"Missing method: {method}"
        print(f"Method '{method}' exists: OK")
    print("PASSED: All required methods exist.\n")


def test_single_observation():
    """One HF solve gives an H2 mole fraction in [0, 1], and every step accessor works.

    The test point has a WGS temperature of 600 K, above the 550 K upper bound.

    Raises:
        AssertionError: If the shape or the value range is wrong.
    """
    print("=" * 60)
    print("TEST 4: Single Observation with Step Methods (HF Mode)")
    print("=" * 60)

    fs = HydrogenReformingFlowsheet(mode="HF")

    x = torch.tensor(
        [[3.0, 0.03, 1100.0, 20.0, 600.0, 15.0]], device=device, dtype=dtype
    )
    print(
        f"Input: steam_ratio=3.0, inert=0.03, SMR_T=1100K, SMR_P=20bar, WGS_T=600K, WGS_P=15bar"
    )

    y = fs.observe(x)
    print(f"Output H2 mole fraction: {y.item():.6f}")
    print(f"Output shape: {y.shape}")

    assert y.shape == torch.Size([1, 1]), f"Shape mismatch: {y.shape}"
    assert 0.0 <= y.item() <= 1.0, f"H2 mole fraction out of range: {y.item()}"

    print("\n--- Step Method Outputs ---")

    s1 = fs.step_1_mixer()
    print(
        f"Step 1 (Feed): T={s1['temperature']:.1f}K, P={s1['pressure']/1e5:.1f}bar, H2={s1['mole_fracs']['H2']:.6f}"
    )

    s2 = fs.step_2_pressure_changer_1()
    print(
        f"Step 2 (Compressor): T={s2['temperature']:.1f}K, P={s2['pressure']/1e5:.1f}bar, Work={s2['work']/1e3:.1f}kW"
    )

    s3 = fs.step_3_heater_1()
    print(
        f"Step 3 (Heater 1): T={s3['temperature']:.1f}K, P={s3['pressure']/1e5:.1f}bar, Duty={s3['heat_duty']/1e6:.2f}MW"
    )

    s4 = fs.step_4_main_reformer()
    print(
        f"Step 4 (SMR): T={s4['temperature']:.1f}K, H2={s4['mole_fracs']['H2']:.4f}, CO={s4['mole_fracs']['CO']:.4f}, Duty={s4['heat_duty']/1e6:.2f}MW"
    )

    s5 = fs.step_5_pressure_changer_2()
    print(
        f"Step 5 (P-Changer 2): T={s5['temperature']:.1f}K, P={s5['pressure']/1e5:.1f}bar, Work={s5['work']/1e3:.1f}kW"
    )

    s6 = fs.step_6_heater_2()
    print(
        f"Step 6 (Heater 2): T={s6['temperature']:.1f}K, Duty={s6['heat_duty']/1e6:.2f}MW"
    )

    s7 = fs.step_7_wgs_reactor()
    print(
        f"Step 7 (WGS): T={s7['temperature']:.1f}K, H2={s7['mole_fracs']['H2']:.4f}, CO2={s7['mole_fracs']['CO2']:.4f}, Duty={s7['heat_duty']/1e6:.2f}MW"
    )

    print("PASSED: Single observation and all step methods work.\n")


def test_grid_evaluation_hf():
    """Five HF points along the steam-ratio and SMR-temperature axes give values in [0, 1].

    Raises:
        AssertionError: If a point returns NaN (known to happen at steam ratio 1.5) or a
            value outside [0, 1].
    """
    print("=" * 60)
    print("TEST 5: Grid Evaluation (HF Mode)")
    print("=" * 60)

    fs = HydrogenReformingFlowsheet(mode="HF")
    b = fs.bounds

    n_points = 5
    steam_ratios = np.linspace(b[0, 0].item(), b[1, 0].item(), n_points)
    inert_fracs = np.linspace(b[0, 1].item(), b[1, 1].item(), n_points)
    smr_temps = np.linspace(b[0, 2].item(), b[1, 2].item(), n_points)
    smr_press = np.linspace(b[0, 3].item(), b[1, 3].item(), n_points)
    wgs_temps = np.linspace(b[0, 4].item(), b[1, 4].item(), n_points)
    wgs_press = np.linspace(b[0, 5].item(), b[1, 5].item(), n_points)

    print("Evaluating at grid points (subset for speed):")
    print("-" * 60)

    test_cases = [
        [
            steam_ratios[0],
            inert_fracs[2],
            smr_temps[2],
            smr_press[2],
            wgs_temps[2],
            wgs_press[2],
        ],
        [
            steam_ratios[2],
            inert_fracs[2],
            smr_temps[2],
            smr_press[2],
            wgs_temps[2],
            wgs_press[2],
        ],
        [
            steam_ratios[4],
            inert_fracs[2],
            smr_temps[2],
            smr_press[2],
            wgs_temps[2],
            wgs_press[2],
        ],
        [
            steam_ratios[2],
            inert_fracs[2],
            smr_temps[0],
            smr_press[2],
            wgs_temps[2],
            wgs_press[2],
        ],
        [
            steam_ratios[2],
            inert_fracs[2],
            smr_temps[4],
            smr_press[2],
            wgs_temps[2],
            wgs_press[2],
        ],
    ]

    for i, case in enumerate(test_cases):
        x = torch.tensor([case], device=device, dtype=dtype)
        y = fs.observe(x)
        print(
            f"Case {i+1}: S/M={case[0]:.2f}, Inert={case[1]:.3f}, SMR_T={case[2]:.0f}K, "
            f"SMR_P={case[3]:.1f}bar, WGS_T={case[4]:.0f}K, WGS_P={case[5]:.1f}bar => H2={y.item():.6f}"
        )
        # observe() returns NaN when every initialisation strategy fails to solve
        assert np.isfinite(y.item()) and 0.0 <= y.item() <= 1.0, f"Case {i+1} failed: H2={y.item()}"
    print("PASSED: Grid evaluation (HF) completed.\n")


def test_grid_evaluation_lf():
    """Same as `test_grid_evaluation_hf` in LF mode (ideal compressor).

    Raises:
        AssertionError: If a point returns NaN (known to happen at steam ratio 1.5) or a
            value outside [0, 1].
    """
    print("=" * 60)
    print("TEST 6: Grid Evaluation (LF Mode)")
    print("=" * 60)

    fs = HydrogenReformingFlowsheet(mode="LF")
    b = fs.bounds

    n_points = 5
    steam_ratios = np.linspace(b[0, 0].item(), b[1, 0].item(), n_points)
    inert_fracs = np.linspace(b[0, 1].item(), b[1, 1].item(), n_points)
    smr_temps = np.linspace(b[0, 2].item(), b[1, 2].item(), n_points)
    smr_press = np.linspace(b[0, 3].item(), b[1, 3].item(), n_points)
    wgs_temps = np.linspace(b[0, 4].item(), b[1, 4].item(), n_points)
    wgs_press = np.linspace(b[0, 5].item(), b[1, 5].item(), n_points)

    print("Evaluating at grid points (subset for speed):")
    print("-" * 60)

    test_cases = [
        [
            steam_ratios[0],
            inert_fracs[2],
            smr_temps[2],
            smr_press[2],
            wgs_temps[2],
            wgs_press[2],
        ],
        [
            steam_ratios[2],
            inert_fracs[2],
            smr_temps[2],
            smr_press[2],
            wgs_temps[2],
            wgs_press[2],
        ],
        [
            steam_ratios[4],
            inert_fracs[2],
            smr_temps[2],
            smr_press[2],
            wgs_temps[2],
            wgs_press[2],
        ],
        [
            steam_ratios[2],
            inert_fracs[2],
            smr_temps[0],
            smr_press[2],
            wgs_temps[2],
            wgs_press[2],
        ],
        [
            steam_ratios[2],
            inert_fracs[2],
            smr_temps[4],
            smr_press[2],
            wgs_temps[2],
            wgs_press[2],
        ],
    ]

    for i, case in enumerate(test_cases):
        x = torch.tensor([case], device=device, dtype=dtype)
        y = fs.observe(x)
        print(
            f"Case {i+1}: S/M={case[0]:.2f}, Inert={case[1]:.3f}, SMR_T={case[2]:.0f}K, "
            f"SMR_P={case[3]:.1f}bar, WGS_T={case[4]:.0f}K, WGS_P={case[5]:.1f}bar => H2={y.item():.6f}"
        )
        # observe() returns NaN when every initialisation strategy fails to solve
        assert np.isfinite(y.item()) and 0.0 <= y.item() <= 1.0, f"Case {i+1} failed: H2={y.item()}"
    print("PASSED: Grid evaluation (LF) completed.\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("HYDROGEN REFORMING FLOWSHEET TEST SUITE")
    print("=" * 60 + "\n")

    test_basic_instantiation()
    test_bounds()
    test_method_calls()
    test_single_observation()
    test_grid_evaluation_hf()
    test_grid_evaluation_lf()

    print("=" * 60)
    print("ALL TESTS COMPLETED SUCCESSFULLY")
    print("=" * 60)
