"""IDAES flowsheet benchmark: steam-methane reforming followed by water-gas shift.

Not used for the poster results. The flowsheet is feed -> compressor C1 -> heater H1 ->
equilibrium SMR reactor -> isothermal pressure changer C2 -> heater H2 -> equilibrium
WGS reactor -> product. Thermodynamics use IDAES' Peng-Robinson natural-gas property
package. The objective is the H2 mole fraction in the product. HF and LF differ only
in the compressor's isentropic efficiency (0.85 vs 1.0).

Requires `idaes-pse` and its solver binaries (`idaes get-extensions`, for IPOPT). Every
evaluation builds and solves a fresh Pyomo model, which can take seconds. If none of
the 25 initialisation guesses converges, the value is NaN. This is known to happen at
low steam-to-methane ratios (about 1.5-1.9).

Module attributes:
    device: torch device of returned tensors (CUDA if available).
    dtype: torch dtype of returned tensors (float64).

Importing this module silences all warnings and sets the pyomo and idaes loggers to
CRITICAL.
"""

import logging
import sys
import warnings

logging.getLogger("pyomo").setLevel(logging.CRITICAL)
logging.getLogger("idaes").setLevel(logging.CRITICAL)
warnings.filterwarnings("ignore")
import numpy as np
import pyomo.environ as pyo
import torch
from idaes.core import FlowsheetBlock
from idaes.core.solvers import get_solver
from idaes.core.util.initialization import propagate_state
from idaes.models.properties.modular_properties.base.generic_property import (
    GenericParameterBlock,
)
from idaes.models.properties.modular_properties.base.generic_reaction import (
    ConcentrationForm,
    GenericReactionParameterBlock,
)
from idaes.models.properties.modular_properties.reactions.dh_rxn import constant_dh_rxn
from idaes.models.properties.modular_properties.reactions.equilibrium_constant import (
    van_t_hoff,
)
from idaes.models.properties.modular_properties.reactions.equilibrium_forms import (
    power_law_equil,
)
from idaes.models.unit_models import (
    Compressor,
    EquilibriumReactor,
    Feed,
    Heater,
    Mixer,
    PressureChanger,
    Product,
)
from idaes.models.unit_models.pressure_changer import ThermodynamicAssumption
from idaes.models_extra.power_generation.properties.natural_gas_PR import (
    get_prop as get_prop_pr,
)
from .benchmarks import BenchmarkBase
from pyomo.network import Arc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double


class HydrogenReformingFlowsheet(BenchmarkBase):
    """Six-input hydrogen-reforming flowsheet with per-unit stream access.

    Unlike the synthetic benchmarks, it does not call `BenchmarkBase.__init__` and has no
    `input_dim`, `step_info` or `observe_all_steps`. The fidelity is fixed at
    construction, and `observe` has no `mode` argument.

    Attributes:
        mode: "HF" or "LF".
        components: Component list ["CH4", "H2O", "H2", "CO", "CO2", "N2"].
        isentropic_efficiency: Compressor C1 efficiency (0.85 HF, 1.0 LF).
        bounds: Tensor of shape (2, 6). The inputs are the steam-to-methane molar ratio
            (1.5-6), the N2 mole fraction in the feed (0.01-0.05), the SMR temperature in K
            (900-1300), the SMR pressure in bar (1-30), the WGS temperature in K (450-550)
            and the WGS pressure in bar (1-30).
    """
    def __init__(self, mode="HF"):
        """Set the fidelity and the bounds. No model is built here.

        Args:
            mode: "HF" or anything else, which is treated as LF.
        """
        self.mode = mode
        self.components = ["CH4", "H2O", "H2", "CO", "CO2", "N2"]
        self.isentropic_efficiency = 0.85 if mode == "HF" else 1.0
        self.bounds = torch.tensor(
            [
                [1.5, 0.01, 900.0, 1.0, 450.0, 1.0],
                [6.0, 0.05, 1300.0, 30.0, 550.0, 30.0],
            ],
            device=device,
            dtype=dtype,
        )
        self._last_model = None
        self._last_input = None

    @staticmethod
    def get_ideal_gas_config(components):
        """IDAES modular property configuration for an ideal-gas vapour phase.

        Uses RPP4 heat-capacity correlations and FTPx state variables. It is kept for
        reference but unused: `_solve_single` uses the Peng-Robinson package from
        `_get_thermo_config`.

        Args:
            components: Component names. Names without parameter data are dropped.

        Returns:
            dict: Configuration for `GenericParameterBlock`.
        """
        from idaes.models.properties.modular_properties.eos.ideal import Ideal
        from idaes.models.properties.modular_properties.phase_equil import (
            NoPhaseEquilibrium,
        )
        from idaes.models.properties.modular_properties.pure import RPP4
        from idaes.models.properties.modular_properties.state_definitions import FTPx

        comp_data = {
            "CH4": {
                "type": "Component",
                "cp_mol_ig_comp": RPP4,
                "enth_mol_ig_comp": RPP4,
                "entr_mol_ig_comp": RPP4,
                "parameter_data": {
                    "mw": (16.043e-3, pyo.units.kg / pyo.units.mol),
                    "pressure_crit": (4.6e6, pyo.units.Pa),
                    "temperature_crit": (190.4, pyo.units.K),
                    "cp_mol_ig_comp_coeff": {
                        "A": 19.25,
                        "B": 5.213e-2,
                        "C": 1.197e-5,
                        "D": -1.132e-8,
                    },
                    "enth_mol_form_vap_comp_ref": (-74600, pyo.units.J / pyo.units.mol),
                    "entr_mol_form_vap_comp_ref": (
                        186.3,
                        pyo.units.J / pyo.units.mol / pyo.units.K,
                    ),
                },
            },
            "H2O": {
                "type": "Component",
                "cp_mol_ig_comp": RPP4,
                "enth_mol_ig_comp": RPP4,
                "entr_mol_ig_comp": RPP4,
                "parameter_data": {
                    "mw": (18.015e-3, pyo.units.kg / pyo.units.mol),
                    "pressure_crit": (22.1e6, pyo.units.Pa),
                    "temperature_crit": (647.3, pyo.units.K),
                    "cp_mol_ig_comp_coeff": {
                        "A": 32.24,
                        "B": 1.924e-3,
                        "C": 1.055e-5,
                        "D": -3.596e-9,
                    },
                    "enth_mol_form_vap_comp_ref": (
                        -241830,
                        pyo.units.J / pyo.units.mol,
                    ),
                    "entr_mol_form_vap_comp_ref": (
                        188.8,
                        pyo.units.J / pyo.units.mol / pyo.units.K,
                    ),
                },
            },
            "H2": {
                "type": "Component",
                "cp_mol_ig_comp": RPP4,
                "enth_mol_ig_comp": RPP4,
                "entr_mol_ig_comp": RPP4,
                "parameter_data": {
                    "mw": (2.016e-3, pyo.units.kg / pyo.units.mol),
                    "pressure_crit": (1.3e6, pyo.units.Pa),
                    "temperature_crit": (33.2, pyo.units.K),
                    "cp_mol_ig_comp_coeff": {
                        "A": 27.14,
                        "B": 9.274e-3,
                        "C": -1.381e-5,
                        "D": 7.645e-9,
                    },
                    "enth_mol_form_vap_comp_ref": (0, pyo.units.J / pyo.units.mol),
                    "entr_mol_form_vap_comp_ref": (
                        130.6,
                        pyo.units.J / pyo.units.mol / pyo.units.K,
                    ),
                },
            },
            "CO": {
                "type": "Component",
                "cp_mol_ig_comp": RPP4,
                "enth_mol_ig_comp": RPP4,
                "entr_mol_ig_comp": RPP4,
                "parameter_data": {
                    "mw": (28.010e-3, pyo.units.kg / pyo.units.mol),
                    "pressure_crit": (3.5e6, pyo.units.Pa),
                    "temperature_crit": (132.9, pyo.units.K),
                    "cp_mol_ig_comp_coeff": {
                        "A": 30.87,
                        "B": -1.285e-2,
                        "C": 2.789e-5,
                        "D": -1.272e-8,
                    },
                    "enth_mol_form_vap_comp_ref": (
                        -110540,
                        pyo.units.J / pyo.units.mol,
                    ),
                    "entr_mol_form_vap_comp_ref": (
                        197.7,
                        pyo.units.J / pyo.units.mol / pyo.units.K,
                    ),
                },
            },
            "CO2": {
                "type": "Component",
                "cp_mol_ig_comp": RPP4,
                "enth_mol_ig_comp": RPP4,
                "entr_mol_ig_comp": RPP4,
                "parameter_data": {
                    "mw": (44.010e-3, pyo.units.kg / pyo.units.mol),
                    "pressure_crit": (7.38e6, pyo.units.Pa),
                    "temperature_crit": (304.2, pyo.units.K),
                    "cp_mol_ig_comp_coeff": {
                        "A": 19.80,
                        "B": 7.344e-2,
                        "C": -5.602e-5,
                        "D": 1.715e-8,
                    },
                    "enth_mol_form_vap_comp_ref": (
                        -393510,
                        pyo.units.J / pyo.units.mol,
                    ),
                    "entr_mol_form_vap_comp_ref": (
                        213.8,
                        pyo.units.J / pyo.units.mol / pyo.units.K,
                    ),
                },
            },
            "N2": {
                "type": "Component",
                "cp_mol_ig_comp": RPP4,
                "enth_mol_ig_comp": RPP4,
                "entr_mol_ig_comp": RPP4,
                "parameter_data": {
                    "mw": (28.014e-3, pyo.units.kg / pyo.units.mol),
                    "pressure_crit": (3.4e6, pyo.units.Pa),
                    "temperature_crit": (126.2, pyo.units.K),
                    "cp_mol_ig_comp_coeff": {
                        "A": 31.15,
                        "B": -1.357e-2,
                        "C": 2.680e-5,
                        "D": -1.168e-8,
                    },
                    "enth_mol_form_vap_comp_ref": (0, pyo.units.J / pyo.units.mol),
                    "entr_mol_form_vap_comp_ref": (
                        191.6,
                        pyo.units.J / pyo.units.mol / pyo.units.K,
                    ),
                },
            },
        }

        selected_comps = {c: comp_data[c] for c in components if c in comp_data}

        config = {
            "components": selected_comps,
            "phases": {
                "Vap": {
                    "type": "VaporPhase",
                    "equation_of_state": Ideal,
                },
            },
            "state_definition": FTPx,
            "state_bounds": {
                "flow_mol": (0, 1000, 10000, pyo.units.mol / pyo.units.s),
                "temperature": (200, 500, 2000, pyo.units.K),
                "pressure": (1e4, 1e5, 1e8, pyo.units.Pa),
            },
            "pressure_ref": (1e5, pyo.units.Pa),
            "temperature_ref": (298.15, pyo.units.K),
            "base_units": {
                "time": pyo.units.s,
                "length": pyo.units.m,
                "mass": pyo.units.kg,
                "amount": pyo.units.mol,
                "temperature": pyo.units.K,
            },
        }
        return config

    @staticmethod
    def get_smr_reaction_config(components):
        """Reaction configuration for steam-methane reforming, CH4 + H2O <-> CO + 3 H2.

        Equilibrium in mole-fraction power-law form, with dH = +206 kJ/mol and van 't Hoff
        K_eq = 12.727 at 973.15 K.

        Args:
            components: Component names. Unlisted ones get stoichiometry 0.

        Returns:
            dict: Configuration for `GenericReactionParameterBlock`.
        """
        stoich = {("Vap", c): 0 for c in components}
        stoich[("Vap", "CH4")] = -1
        stoich[("Vap", "H2O")] = -1
        stoich[("Vap", "CO")] = 1
        stoich[("Vap", "H2")] = 3

        return {
            "base_units": {
                "time": pyo.units.s,
                "length": pyo.units.m,
                "mass": pyo.units.kg,
                "amount": pyo.units.mol,
                "temperature": pyo.units.K,
            },
            "equilibrium_reactions": {
                "SMR": {
                    "stoichiometry": stoich,
                    "heat_of_reaction": constant_dh_rxn,
                    "equilibrium_constant": van_t_hoff,
                    "equilibrium_form": power_law_equil,
                    "concentration_form": ConcentrationForm.moleFraction,
                    "parameter_data": {
                        "dh_rxn_ref": (206000, pyo.units.J / pyo.units.mol),
                        "k_eq_ref": (12.727, None),
                        "T_eq_ref": (973.15, pyo.units.K),
                    },
                },
            },
        }

    @staticmethod
    def get_wgs_reaction_config(components):
        """Reaction configuration for water-gas shift, CO + H2O <-> CO2 + H2.

        Equilibrium in mole-fraction power-law form, with dH = -41 kJ/mol and van 't Hoff
        K_eq = 1.6248 at 973.15 K.

        Args:
            components: Component names. Unlisted ones get stoichiometry 0.

        Returns:
            dict: Configuration for `GenericReactionParameterBlock`.
        """
        stoich = {("Vap", c): 0 for c in components}
        stoich[("Vap", "H2O")] = -1
        stoich[("Vap", "CO")] = -1
        stoich[("Vap", "H2")] = 1
        stoich[("Vap", "CO2")] = 1

        return {
            "base_units": {
                "time": pyo.units.s,
                "length": pyo.units.m,
                "mass": pyo.units.kg,
                "amount": pyo.units.mol,
                "temperature": pyo.units.K,
            },
            "equilibrium_reactions": {
                "WGS": {
                    "stoichiometry": stoich,
                    "heat_of_reaction": constant_dh_rxn,
                    "equilibrium_constant": van_t_hoff,
                    "equilibrium_form": power_law_equil,
                    "concentration_form": ConcentrationForm.moleFraction,
                    "parameter_data": {
                        "dh_rxn_ref": (-41000, pyo.units.J / pyo.units.mol),
                        "k_eq_ref": (1.6248, None),
                        "T_eq_ref": (973.15, pyo.units.K),
                    },
                },
            },
        }

    def _get_thermo_config(self):
        """IDAES Peng-Robinson natural-gas property configuration, without state bounds.

        Returns:
            dict: Configuration for `GenericParameterBlock`.
        """
        cfg = get_prop_pr(components=self.components)
        if "state_bounds" in cfg:
            del cfg["state_bounds"]
        return cfg

    def _solve_single(self, x_row):
        """Build, initialise and solve the flowsheet for one input point.

        The feed is 100 mol/s at 300 K and 1 bar, with traces (1e-6) of H2, CO and CO2.
        Initial guesses for the SMR and WGS outlet compositions are tried in turn until IPOPT
        reports an optimal solution. On success, the solved model is kept for the `step_*`
        accessors. Every call first clears the previously kept model.

        Args:
            x_row: One input point, NumPy array of shape (6,) in the units of `bounds`.

        Returns:
            float: H2 mole fraction in the product, or NaN if no guess converged.
        """
        # Forget the previous solve, so step_* methods never report another point's streams
        self._last_model = None
        self._last_input = None
        steam_ratio = x_row[0]
        inert_frac = x_row[1]
        smr_temp = x_row[2]
        smr_pres = x_row[3] * 1e5
        wgs_temp = x_row[4]
        wgs_pres = x_row[5] * 1e5

        remaining = 1.0 - inert_frac
        ch4_frac = remaining / (1.0 + steam_ratio)
        h2o_frac = steam_ratio * ch4_frac
        n2_frac = max(inert_frac - 3e-6, 1e-8)

        from idaes.core.util.exceptions import InitializationError

        def build_model():
            """Build the flowsheet with all degrees of freedom fixed from `x_row`.

            Returns:
                pyomo.environ.ConcreteModel: The unsolved model.
            """
            m = pyo.ConcreteModel()
            m.fs = FlowsheetBlock(dynamic=False)

            thermo_cfg = self._get_thermo_config()
            m.fs.thermo_params = GenericParameterBlock(**thermo_cfg)

            smr_rxn_cfg = self.get_smr_reaction_config(self.components)
            m.fs.smr_rxn_params = GenericReactionParameterBlock(
                property_package=m.fs.thermo_params, **smr_rxn_cfg
            )

            wgs_rxn_cfg = self.get_wgs_reaction_config(self.components)
            m.fs.wgs_rxn_params = GenericReactionParameterBlock(
                property_package=m.fs.thermo_params, **wgs_rxn_cfg
            )

            m.fs.feed = Feed(property_package=m.fs.thermo_params)
            m.fs.C1 = Compressor(property_package=m.fs.thermo_params)
            m.fs.C1.efficiency_isentropic.fix(self.isentropic_efficiency)
            m.fs.H1 = Heater(
                property_package=m.fs.thermo_params,
                has_pressure_change=False,
                has_phase_equilibrium=False,
            )
            m.fs.SMR = EquilibriumReactor(
                property_package=m.fs.thermo_params,
                reaction_package=m.fs.smr_rxn_params,
                has_equilibrium_reactions=True,
                has_rate_reactions=False,
                has_heat_of_reaction=True,
                has_heat_transfer=True,
                has_pressure_change=False,
            )
            m.fs.C2 = PressureChanger(
                property_package=m.fs.thermo_params,
                thermodynamic_assumption=ThermodynamicAssumption.isothermal,
            )
            m.fs.H2 = Heater(
                property_package=m.fs.thermo_params,
                has_pressure_change=False,
                has_phase_equilibrium=False,
            )
            m.fs.WGS = EquilibriumReactor(
                property_package=m.fs.thermo_params,
                reaction_package=m.fs.wgs_rxn_params,
                has_equilibrium_reactions=True,
                has_rate_reactions=False,
                has_heat_of_reaction=True,
                has_heat_transfer=True,
                has_pressure_change=False,
            )
            m.fs.product = Product(property_package=m.fs.thermo_params)

            m.fs.s01 = Arc(source=m.fs.feed.outlet, destination=m.fs.C1.inlet)
            m.fs.s02 = Arc(source=m.fs.C1.outlet, destination=m.fs.H1.inlet)
            m.fs.s03 = Arc(source=m.fs.H1.outlet, destination=m.fs.SMR.inlet)
            m.fs.s04 = Arc(source=m.fs.SMR.outlet, destination=m.fs.C2.inlet)
            m.fs.s05 = Arc(source=m.fs.C2.outlet, destination=m.fs.H2.inlet)
            m.fs.s06 = Arc(source=m.fs.H2.outlet, destination=m.fs.WGS.inlet)
            m.fs.s07 = Arc(source=m.fs.WGS.outlet, destination=m.fs.product.inlet)

            pyo.TransformationFactory("network.expand_arcs").apply_to(m)

            m.fs.feed.outlet.flow_mol.fix(100)
            m.fs.feed.outlet.temperature.fix(300)
            m.fs.feed.outlet.pressure.fix(1e5)
            m.fs.feed.outlet.mole_frac_comp[0, "CH4"].fix(ch4_frac)
            m.fs.feed.outlet.mole_frac_comp[0, "H2O"].fix(h2o_frac)
            m.fs.feed.outlet.mole_frac_comp[0, "H2"].fix(1e-6)
            m.fs.feed.outlet.mole_frac_comp[0, "CO"].fix(1e-6)
            m.fs.feed.outlet.mole_frac_comp[0, "CO2"].fix(1e-6)
            m.fs.feed.outlet.mole_frac_comp[0, "N2"].fix(n2_frac)

            m.fs.C1.outlet.pressure.fix(smr_pres)
            m.fs.H1.outlet.temperature.fix(smr_temp)
            m.fs.SMR.outlet.temperature.fix(smr_temp)
            m.fs.C2.outlet.pressure.fix(wgs_pres)
            m.fs.H2.outlet.temperature.fix(wgs_temp)
            m.fs.WGS.outlet.temperature.fix(wgs_temp)
            return m

        def get_init_strategies():
            """Initial guesses for the reactor outlet compositions.

            Returns:
                list[dict]: 25 guesses (5 H2 x 5 CO mole fractions), each with "smr_out" and
                "wgs_out" dicts that map component to mole fraction.
            """
            base_n2 = max(n2_frac, 0.05)
            strategies = []
            h2_guesses = [0.50, 0.55, 0.45, 0.60, 0.40]
            co_guesses = [0.12, 0.15, 0.10, 0.08, 0.18]
            for h2_g in h2_guesses:
                for co_g in co_guesses:
                    ch4_out = 0.05
                    h2o_out = 0.15
                    co2_out = max(0.02, 1.0 - h2_g - co_g - ch4_out - h2o_out - base_n2)
                    if co2_out < 0:
                        co2_out = 0.05
                        h2o_out = max(
                            0.05, 1.0 - h2_g - co_g - ch4_out - co2_out - base_n2
                        )
                    strategies.append(
                        {
                            "smr_out": {
                                "CH4": ch4_out,
                                "H2O": h2o_out,
                                "H2": h2_g,
                                "CO": co_g,
                                "CO2": co2_out,
                                "N2": base_n2,
                            },
                            "wgs_out": {
                                "CH4": ch4_out,
                                "H2O": 0.08,
                                "H2": h2_g + 0.05,
                                "CO": max(0.02, co_g - 0.08),
                                "CO2": co2_out + 0.08,
                                "N2": base_n2,
                            },
                        }
                    )
            return strategies

        def try_initialize(m, smr_fracs, wgs_fracs):
            """Seed all state blocks, initialise units sequentially and solve.

            Args:
                m: Model from `build_model`.
                smr_fracs: Guessed SMR outlet mole fractions.
                wgs_fracs: Guessed WGS outlet mole fractions.

            Returns:
                The Pyomo solver results (IPOPT, max_iter 500).

            Raises:
                InitializationError: If a unit fails to initialise. Other solver errors can also
                    propagate; the caller catches all of them.
            """
            solver = get_solver()
            solver.options["max_iter"] = 500

            def seed_props(blk, temp, pres, fracs):
                """Set initial values of a state block.

                Args:
                    blk: IDAES state block.
                    temp: Temperature in K.
                    pres: Pressure in Pa.
                    fracs: Mole fractions per component.
                """
                blk.flow_mol.set_value(100)
                blk.temperature.set_value(temp)
                blk.pressure.set_value(pres)
                for c, f in fracs.items():
                    blk.mole_frac_comp[c].set_value(f)

            feed_fracs = {
                "CH4": ch4_frac,
                "H2O": h2o_frac,
                "H2": 1e-6,
                "CO": 1e-6,
                "CO2": 1e-6,
                "N2": n2_frac,
            }

            seed_props(m.fs.C1.control_volume.properties_in[0], 300, 1e5, feed_fracs)
            seed_props(
                m.fs.C1.control_volume.properties_out[0], 350, smr_pres, feed_fracs
            )
            seed_props(
                m.fs.H1.control_volume.properties_in[0], 350, smr_pres, feed_fracs
            )
            seed_props(
                m.fs.H1.control_volume.properties_out[0], smr_temp, smr_pres, feed_fracs
            )
            seed_props(
                m.fs.SMR.control_volume.properties_in[0], smr_temp, smr_pres, feed_fracs
            )
            seed_props(
                m.fs.SMR.control_volume.properties_out[0], smr_temp, smr_pres, smr_fracs
            )
            seed_props(
                m.fs.C2.control_volume.properties_in[0], smr_temp, smr_pres, smr_fracs
            )
            seed_props(
                m.fs.C2.control_volume.properties_out[0], smr_temp, wgs_pres, smr_fracs
            )
            seed_props(
                m.fs.H2.control_volume.properties_in[0], smr_temp, wgs_pres, smr_fracs
            )
            seed_props(
                m.fs.H2.control_volume.properties_out[0], wgs_temp, wgs_pres, smr_fracs
            )
            seed_props(
                m.fs.WGS.control_volume.properties_in[0], wgs_temp, wgs_pres, smr_fracs
            )
            seed_props(
                m.fs.WGS.control_volume.properties_out[0], wgs_temp, wgs_pres, wgs_fracs
            )

            m.fs.feed.initialize(outlvl=logging.CRITICAL)
            propagate_state(arc=m.fs.s01)

            m.fs.C1.initialize(outlvl=logging.CRITICAL)
            propagate_state(arc=m.fs.s02)

            m.fs.H1.initialize(outlvl=logging.CRITICAL)
            propagate_state(arc=m.fs.s03)

            m.fs.SMR.initialize(outlvl=logging.CRITICAL)
            propagate_state(arc=m.fs.s04)

            m.fs.C2.initialize(outlvl=logging.CRITICAL)
            propagate_state(arc=m.fs.s05)

            m.fs.H2.initialize(outlvl=logging.CRITICAL)
            propagate_state(arc=m.fs.s06)

            m.fs.WGS.initialize(outlvl=logging.CRITICAL)
            propagate_state(arc=m.fs.s07)

            m.fs.product.initialize(outlvl=logging.CRITICAL)
            return solver.solve(m, tee=False)

        strategies = get_init_strategies()

        for strategy in strategies:
            m = build_model()
            try:
                results = try_initialize(m, strategy["smr_out"], strategy["wgs_out"])
                if (
                    results.solver.termination_condition
                    == pyo.TerminationCondition.optimal
                ):
                    self._last_model = m
                    self._last_input = x_row
                    return pyo.value(m.fs.product.inlet.mole_frac_comp[0, "H2"])
            except (InitializationError, Exception):
                continue
        return float("nan")

    def _get_stream_values(self, port):
        """Read the state of a port of the solved model.

        Args:
            port: Pyomo port, e.g. `m.fs.SMR.outlet`.

        Returns:
            dict: "flow_mol" (mol/s), "temperature" (K), "pressure" (Pa) and "mole_fracs"
            (component -> mole fraction).
        """
        flow = pyo.value(port.flow_mol[0])
        temp = pyo.value(port.temperature[0])
        pres = pyo.value(port.pressure[0])
        mole_fracs = {c: pyo.value(port.mole_frac_comp[0, c]) for c in self.components}
        return {
            "flow_mol": flow,
            "temperature": temp,
            "pressure": pres,
            "mole_fracs": mole_fracs,
        }

    def step_1_mixer(self, idx=0):
        """Feed stream of the last successful solve.

        The flowsheet has no mixer unit; the name is kept for interface compatibility.

        Args:
            idx: Unused.

        Returns:
            dict or None: Stream values (see `_get_stream_values`), or None if the last solve
            failed or nothing was solved yet.
        """
        if self._last_model is None:
            return None
        m = self._last_model
        return self._get_stream_values(m.fs.feed.outlet)

    def step_2_pressure_changer_1(self, idx=0):
        """Compressor C1 outlet of the last successful solve.

        Args:
            idx: Unused.

        Returns:
            dict or None: Stream values plus "work" (mechanical work in W), or None.
        """
        if self._last_model is None:
            return None
        m = self._last_model
        outlet = self._get_stream_values(m.fs.C1.outlet)
        outlet["work"] = pyo.value(m.fs.C1.work_mechanical[0])
        return outlet

    def step_3_heater_1(self, idx=0):
        """Heater H1 outlet of the last successful solve.

        Args:
            idx: Unused.

        Returns:
            dict or None: Stream values plus "heat_duty" (W), or None.
        """
        if self._last_model is None:
            return None
        m = self._last_model
        outlet = self._get_stream_values(m.fs.H1.outlet)
        outlet["heat_duty"] = pyo.value(m.fs.H1.heat_duty[0])
        return outlet

    def step_4_main_reformer(self, idx=0):
        """SMR reactor outlet of the last successful solve.

        Args:
            idx: Unused.

        Returns:
            dict or None: Stream values plus "heat_duty" (W), or None.
        """
        if self._last_model is None:
            return None
        m = self._last_model
        outlet = self._get_stream_values(m.fs.SMR.outlet)
        outlet["heat_duty"] = pyo.value(m.fs.SMR.heat_duty[0])
        return outlet

    def step_5_pressure_changer_2(self, idx=0):
        """Isothermal pressure changer C2 outlet of the last successful solve.

        Args:
            idx: Unused.

        Returns:
            dict or None: Stream values plus "work" (W), or None.
        """
        if self._last_model is None:
            return None
        m = self._last_model
        outlet = self._get_stream_values(m.fs.C2.outlet)
        outlet["work"] = pyo.value(m.fs.C2.work_mechanical[0])
        return outlet

    def step_6_heater_2(self, idx=0):
        """Heater H2 outlet of the last successful solve.

        Args:
            idx: Unused.

        Returns:
            dict or None: Stream values plus "heat_duty" (W), or None.
        """
        if self._last_model is None:
            return None
        m = self._last_model
        outlet = self._get_stream_values(m.fs.H2.outlet)
        outlet["heat_duty"] = pyo.value(m.fs.H2.heat_duty[0])
        return outlet

    def step_7_wgs_reactor(self, idx=0):
        """WGS reactor outlet of the last successful solve.

        Args:
            idx: Unused.

        Returns:
            dict or None: Stream values plus "heat_duty" (W), or None.
        """
        if self._last_model is None:
            return None
        m = self._last_model
        outlet = self._get_stream_values(m.fs.WGS.outlet)
        outlet["heat_duty"] = pyo.value(m.fs.WGS.heat_duty[0])
        return outlet

    def observe(self, x):
        """Solve the flowsheet for each input row.

        After the call, the `step_*` accessors describe the last row's solve.

        Args:
            x: Inputs, tensor or array of shape (N, 6) or (6,), in the units of `bounds`.

        Returns:
            torch.Tensor: H2 mole fractions, shape (N, 1), on the module `device`. Rows that did
            not converge are NaN.
        """
        if torch.is_tensor(x):
            x = x.detach().cpu().numpy()
        if x.ndim == 1:
            x = x.reshape(1, -1)
        n_samples = x.shape[0]
        results = np.zeros(n_samples)

        for i in range(n_samples):
            results[i] = self._solve_single(x[i, :])
        return torch.tensor(results, device=device, dtype=dtype).unsqueeze(-1)
