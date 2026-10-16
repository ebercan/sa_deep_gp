"""BioSTEAM flowsheet benchmark: sugarcane-to-ethanol biorefinery, with the IRR as objective.

Not used for the poster results. The flowsheet follows BioSTEAM's sugarcane ethanol
example: juicing, juice clarification, evaporation, fermentation, distillation,
molecular sieves, and a boiler-turbogenerator with utilities. The objective is the
internal rate of return from `biorefineries.sugarcane.create_tea`. HF and LF differ in
the cell-growth conversion, centrifuge split, heat-transfer coefficients, and boiler
and turbogenerator efficiencies.

Requires `biosteam` and `biorefineries`. Every evaluation rebuilds the flowsheet from
scratch and resets BioSTEAM's global settings (`bst.nbtutorial()`), so evaluations are
slow and not thread-safe. Failed simulations, and IRRs that are not finite or lie
outside [0, 1], are reported as 0.0001.

Module attributes:
    device: torch device of returned tensors (CUDA if available).
    dtype: torch dtype of returned tensors (float64).
"""

import math

import biorefineries.sugarcane as sc
import biosteam as bst
import numpy as np
import torch
from biosteam import units
from .benchmarks import BenchmarkBase

# DETECT GPU: Critical for your test


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double


class BioethanolBenchmark(BenchmarkBase):
    """25-input sugarcane-to-ethanol flowsheet; the objective is the IRR.

    Unlike the synthetic benchmarks, it does not call `BenchmarkBase.__init__` and has no
    `input_dim`, `step_info` or `observe_all_steps`. The fidelity is fixed at construction.

    Note:
        Some inputs have no effect, because flowsheet specifications overwrite them during
        the simulation. Imbibition water flow (input 3), the acid/water ratio (5) and the
        CaO/water ratio (6) are reset in proportion to the cane feed. Wash water flow (8)
        is reset in proportion to the solids.

    Attributes:
        mode: "HF" or "LF".
        cell_growth_reaction_X: Conversion of the fermentation cell-growth reaction
            (0.15 HF, 0.0 LF).
        centrifuge_split: Yeast centrifuge split (0.90 HF, 0.99 LF).
        hx_process_U: Heat-transfer coefficient of the beer/stillage exchanger
            (0.95 HF, 1.3 LF).
        beer_reboiler_U: Heat-transfer coefficient of the beer-column reboiler
            (1.4 HF, 1.85 LF).
        boiler_efficiency: Boiler efficiency (0.80 HF, 0.99 LF).
        turbogenerator_efficiency: Turbogenerator efficiency (0.85 HF, 0.99 LF).
        bounds: Tensor of shape (2, 25); the inputs and units are listed in `__init__`.
    """
    def __init__(self, mode="HF"):
        """Set the fidelity parameters and the bounds.

        The inputs, in order, are: (1) cane water content (mass parts relative to the other
        cane components), (2) cane flow in kg/h, (3) imbibition water flow in kg/h,
        (4) imbibition water temperature in K, (5) acid/water ratio, (6) CaO/water ratio,
        (7) polymer (flocculant) flow in kg/h, (8) wash water flow in kg/h, (9) wash water
        temperature in K, (10) enzyme treatment temperature in K, (11) juice storage
        residence time in h, (12) temperature before acid addition in K, (13) lime mixing
        residence time in h, (14) acid-lime mixing residence time in h, (15) temperature
        before flocculant in K, (16) number of fermentors (rounded to an integer), (17) beer
        tank residence time in h, (18) beer column pressure in bar, (19, 20) beer column
        light- and heavy-key recoveries, (21) beer column reflux factor k, (22) rectifier
        pressure in bar, (23) rectifier top ethanol mole fraction, (24) rectifier bottom
        ethanol mole fraction and (25) rectifier reflux factor k.

        Args:
            mode: "HF" or anything else, which is treated as LF.
        """
        self.mode = mode
        if mode == "HF":
            self.cell_growth_reaction_X = 0.15
            self.centrifuge_split = 0.90
            self.hx_process_U = 0.95
            self.beer_reboiler_U = 1.4
            self.boiler_efficiency = 0.80
            self.turbogenerator_efficiency = 0.85
        else:
            self.cell_growth_reaction_X = 0.0
            self.centrifuge_split = 0.99
            self.hx_process_U = 1.3
            self.beer_reboiler_U = 1.85
            self.boiler_efficiency = 0.99
            self.turbogenerator_efficiency = 0.99
        lower = torch.tensor(
            [
                0.7,  # 1: sugarcane water ratio
                100000,  # 2: sugarcane flowrate kg/h
                50000,  # 3: imbibition water flowrate
                280,  # 4: imbibition water temp K
                2,  # 5: acid/water ratio
                0.05,  # 6: CaO/water ratio
                0.2,  # 7: polymer flowrate kg/h
                10000,  # 8: wash water flowrate
                280,  # 9: wash water temp K
                313.15,  # 10: enzyme temp K
                1,  # 11: juice tau h
                323,  # 12: heatex temp before acid mix K
                0.01,  # 13: lime mix tau h
                0.01,  # 14: acid-lime mix tau h
                300,  # 15: heatex temp before flocculant K
                2,  # 16: N_fermentor
                1,  # 17: beer tank tau h
                1.2,  # 18: column_1 P bar
                0.5,  # 19: column_1 Lr
                0.5,  # 20: column_1 Hr
                1.01,  # 21: column_1 k
                8,  # 22: column_2 P bar
                0.75,  # 23: column_2 y_top
                1.0e-06,  # 24: column_2 x_bot
                1.01,  # 25: column_2 k
            ],
            device=device,
            dtype=dtype,
        )

        upper = torch.tensor(
            [
                2,  # 1: sugarcane water ratio
                600000,  # 2: sugarcane flowrate kg/h
                150000,  # 3: imbibition water flowrate
                370,  # 4: imbibition water temp K
                8,  # 5: acid/water ratio
                0.25,  # 6: CaO/water ratio
                1.3,  # 7: polymer flowrate kg/h
                25000,  # 8: wash water flowrate
                370,  # 9: wash water temp K
                333.15,  # 10: enzyme temp K
                15,  # 11: juice tau h
                370,  # 12: heatex temp before acid mix K
                0.7,  # 13: lime mix tau h
                0.7,  # 14: acid-lime mix tau h
                363,  # 15: heatex temp before flocculant K
                8,  # 16: N_fermentor
                15,  # 17: beer tank tau h
                5,  # 18: column_1 P bar
                0.995,  # 19: column_1 Lr
                0.93,  # 20: column_1 Hr
                5,  # 21: column_1 k
                25,  # 22: column_2 P bar
                0.90,  # 23: column_2 y_top
                3.5e-06,  # 24: column_2 x_bot
                5,  # 25: column_2 k
            ],
            device=device,
            dtype=dtype,
        )
        self.bounds = torch.stack([lower, upper], dim=0)

    def observe(self, x):
        """Simulate the flowsheet for each input row.

        Args:
            x: Inputs, torch tensor of shape (N, 25) or (25,). NumPy arrays are not accepted.

        Returns:
            torch.Tensor: IRR values, shape (N, 1), on the module `device`. Failed points are
            0.0001.
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        n_samples = x.shape[0]
        results = []

        for i in range(n_samples):
            irr = self._simulate_single(x[i])
            results.append(irr)
        return torch.tensor(results, device=device, dtype=dtype).unsqueeze(-1)

    def _simulate_single(self, xi):
        """Build, simulate and evaluate the techno-economics of the flowsheet for one input point.

        Prints the IRR of every evaluation.

        Args:
            xi: One input point, tensor of shape (25,).

        Returns:
            float: The IRR in [0, 1], or 0.0001 if the simulation fails or the IRR is not
            finite or lies outside [0, 1].
        """
        v = xi.cpu().numpy()

        sugarcane_water_ratio = float(v[0])
        sugarcane_flowrate = float(v[1])
        imbibition_water_flow = float(v[2])
        imbibition_water_temp = float(v[3])
        acid_water_ratio = float(v[4])
        cao_water_ratio = float(v[5])
        polymer_flowrate = float(v[6])
        wash_water_flow = float(v[7])
        wash_water_temp = float(v[8])
        enzyme_temp = float(v[9])
        juice_tau = float(v[10])
        heatex_acid_temp = float(v[11])
        lime_mix_tau = float(v[12])
        acid_lime_mix_tau = float(v[13])
        heatex_flocculant_temp = float(v[14])
        n_fermentor = int(round(v[15]))
        beer_tank_tau = float(v[16])
        col1_P_bar = float(v[17])
        col1_Lr = float(v[18])
        col1_Hr = float(v[19])
        col1_k = float(v[20])
        col2_P_bar = float(v[21])
        col2_y_top = float(v[22])
        col2_x_bot = float(v[23])
        col2_k = float(v[24])

        bst.nbtutorial()  # Ignore warnings and reset local BioSTEAM preferences
        bst.settings.set_thermo(
            [
                "Water",  # Define chemicals by name
                "Ethanol",
                "Octane",
                "Glucose",
                "Sucrose",
                "H3PO4",
                "P4O10",
                "CO2",
                "O2",
                "N2",
                "CH4",
                "Cellulose",
                "Hemicellulose",
                "Lignin",
                "Flocculant",
                "Solids",
                "DryYeast",
                "CaO",
                "Ash",
            ],
            # Use the BioSTEAM database (not the default ChEDL) for common
            # biorefinery chemicals like lignocellulosic components.
            db="BioSTEAM",
        )

        bst.settings.CEPCI = 300
        bst.settings.electricity_price = 0.065
        # Steam is produced on-site by a boiler,
        # so make it the only available heating agent.

        steam_utility = bst.settings.get_agent("low_pressure_steam")
        bst.settings.heating_agents = [steam_utility]
        steam_utility.heat_transfer_efficiency = 0.9
        steam_utility.T = 529.2
        steam_utility.P = 44e5
        # Steam, cooling water, and chilled water are regenerated by
        # on-site facilities. The regeneration and heat transfer
        # prices are given accounted for by the capital cost and
        # electricity consumption of these facilities

        steam_utility.regeneration_price = 0.0
        bst.settings.get_agent("cooling_water").regeneration_price = 0
        bst.settings.get_agent("chilled_water").heat_transfer_price = 0

        # Raw material price (USD/kg)

        price = {
            "Sugar cane": 0.03455,  # 70% m.c
            "Water": 0.000353,
            "HCl": 0.205,
            "Lime": 0.077,
            "H3PO4": 0,  # Not significant
            "NaOH": 0.41,
            "Protease": 0.5,
            "Polymer": 0,  # Not significant
            "Steam": 0.017,
            "Ethanol": 0.789,
            "Waste": -0.33,
            "Gasoline": 0.756,
        }  # 2 USD/gal

        bst.main_flowsheet.set_flowsheet("sugarcane_ethanol")

        # We can create streams and set component splits faster by defining chemical groups

        chemicals = bst.settings.chemicals
        chemicals.define_group(
            name="Fiber",
            IDs=["Cellulose", "Hemicellulose", "Lignin"],
            composition=[0.4704, 0.2775, 0.2520],
            wt=True,  # Composition is given as weight
        )
        chemicals.define_group(
            name="Sugar",
            IDs=["Sucrose", "Glucose"],
            # Default composition as equimolar
        )

        sugarcane = bst.Stream(
            "sugarcane",
            Water=sugarcane_water_ratio,  # BOVAR
            Glucose=0.01208,
            Sucrose=0.1369,
            Ash=0.006,
            Fiber=0.13,
            Solids=0.015,
            total_flow=sugarcane_flowrate,  # BOVAR
            units="kg/hr",
            price=price["Sugar cane"],
        )

        enzyme = bst.Stream(
            "enzyme", Cellulose=100, Water=900, units="kg/hr", price=price["Protease"]
        )

        imbibition_water = bst.Stream(
            "imbibition_water",
            Water=imbibition_water_flow,
            units="kg/hr",
            T=imbibition_water_temp,
        )  # BOVAR variables 3 and 4

        acid_total = 90.0
        acid_h3po4 = acid_total * acid_water_ratio / (1.0 + acid_water_ratio)
        acid_water = acid_total - acid_h3po4
        H3PO4 = bst.Stream(
            "H3PO4",
            H3PO4=acid_h3po4,
            Water=acid_water,
            units="kg/hr",
            price=price["H3PO4"],  # BOVAR, keep total of 90. apply ratio (variable 5)
        )  # to T203

        lime_total = 2533.0
        lime_cao = lime_total * cao_water_ratio / (1.0 + cao_water_ratio)
        lime_water = lime_total - lime_cao
        lime = bst.Stream(
            "lime",
            CaO=lime_cao,
            Water=lime_water,
            units="kg/hr",
            price=price["Lime"],  # BOVAR, keep total the same. apply ratio (variable 6)
        )  # to P5

        polymer = bst.Stream(
            "polymer",
            Flocculant=polymer_flowrate,  # BOVAR
            units="kg/hr",
            price=price["Polymer"],
        )  # to T205

        rvf_wash_water = bst.Stream(
            "rvf_wash_water",
            Water=wash_water_flow,
            units="kg/hr",
            T=wash_water_temp,  # BOVAR, variables 8 and 9
        )  # to C202

        ### Unit operations ###

        # Feed from storage

        U101 = units.ConveyingBelt("U101", sugarcane)

        # Separate metals

        U102 = units.MagneticSeparator("U102", U101 - 0)

        # Shredded cane

        U103 = units.Shredder("U103", U102 - 0)

        # Hydrolyze starch

        T201 = units.EnzymeTreatment(
            "T201", [U103 - 0, enzyme], T=enzyme_temp
        )  # BOVAR, enzyme temp.

        # Finely crush lipid cane

        imbibition_water_recycle = bst.Stream()  # To connect later
        U201 = units.CrushingMill(
            "U201",
            [T201 - 0, imbibition_water_recycle],
            split=dict(Ash=0.92, Fiber=0.92, Sugar=0.04, Solids=1),
            moisture_content=0.5,
        )

        # Convey out bagasse

        U202 = units.ConveyingBelt("U202", U201 - 0, outs="Bagasse")

        # Screen out fibers

        S201 = units.VibratingScreen(
            "S201",
            U201 - 1,
            split=dict(Ash=0.35, Fiber=0.35, Sugar=0.88, Water=0.88, Solids=0),
        )

        # Mix in water

        M201 = units.Mixer(
            "M201", [S201 - 1, imbibition_water], imbibition_water_recycle
        )

        # Store juice before treatment

        T202 = units.StorageTank(
            "T202", S201 - 0, tau=juice_tau, vessel_material="Carbon steel"
        )  # BOVAR 1 to 1000

        # Heat up before adding acid

        H201 = units.HXutility("H201", T202 - 0, T=heatex_acid_temp)  # BOVAR

        # Mix in acid

        T203 = units.MixTank("T203", [H201 - 0, H3PO4])

        # Pump acid solution

        P201 = units.Pump("P201", T203 - 0)

        # Mix lime solution

        T204 = units.MixTank(
            "T204", [P201 - 0, lime], tau=lime_mix_tau
        )  # bovar: 0.1 to 10

        # Blend acid lipid solution with lime

        T205 = units.MixTank(
            "T205", T204 - 0, tau=acid_lime_mix_tau
        )  # bovar: 0.1 to 10
        P202 = units.Pump("P202", T205 - 0)

        # Mix recycle

        RVF_recycle = bst.Stream()  # From rotary vacuum filter; connect later
        M202 = units.Mixer("M202", [P202 - 0, RVF_recycle])

        # Heat before adding flocculant

        H202 = units.HXutility(
            "H202", M202 - 0, T=heatex_flocculant_temp
        )  # bovar variable 15.

        # Mix in flocculant

        T206 = units.MixTank("T206", [H202 - 0, polymer])
        T206.tau = 0.10

        # Separate residual solids

        C201 = units.Clarifier(
            "C201",
            T206 - 0,
            split=dict(
                Ash=0,
                CaO=0,
                Fiber=0,
                Flocculant=0.522,
                Sugar=0.522,
                H3PO4=0.522,
                Water=0.522,
            ),
        )

        # Remove solids as filter cake

        C202 = units.RVF(
            "C202",
            [C201 - 1, rvf_wash_water],
            outs=("filter_cake", ""),
            moisture_content=0.80,
            split=dict(Ash=0.85, CaO=0.85, Fiber=0.85, Sugar=0.01),
        )
        P203 = units.Pump("P203", C202 - 1, RVF_recycle)

        # Screen out small fibers from sugar stream

        S202 = units.VibratingScreen(
            "S202",
            C201 - 0,
            outs=("", "fiber_fines"),
            split=dict(
                Ash=1.0,
                CaO=1.0,
                Fiber=1.0,
                Flocculant=0.0,
                Sugar=0.998,
                H3PO4=1.0,
                Water=0.998,
            ),
        )
        S202.mesh_opening = 2

        ### Process specifications ###

        # Specifications dependent on lipid cane flow rate

        @U103.add_specification(run=True)  # Run unit operation afterwords
        def correct_flows():
            """Shredder specification: scale enzyme, lime, acid and imbibition water with the feed."""
            feedstock = U101.ins[0]
            F_mass = feedstock.F_mass
            # correct enzyme, lime, phosphoric acid, and imbibition water

            enzyme.imass["Cellulose", "Water"] = 0.003 * F_mass * np.array([0.1, 0.9])
            lime.imass["CaO", "Water"] = 0.001 * F_mass * np.array([0.046, 0.954])
            H3PO4.imass["H3PO4", "Water"] = 0.00025 * F_mass
            imbibition_water.imass["Water"] = 0.25 * F_mass

        # Specifications within a system

        @P202.add_specification(run=True)
        def correct_wash_water():
            """Specification on pump P202: set the filter wash water from the solids flow."""
            solids = P202.ins[0].imol["Ash", "CaO", "Fiber"].sum()
            rvf_wash_water.imol["Water"] = 0.0574 * solids

        ### Streams ###

        # Fresh water

        stripping_water = bst.Stream("stripping_water", Water=3000, units="kg/hr")

        # Gasoline

        denaturant = bst.Stream(
            "denaturant", Octane=230.69, units="kg/hr", price=price["Gasoline"]
        )

        # Yeast

        yeast = bst.Stream("yeast", Water=24700, DryYeast=10300, units="kg/hr")

        # Ethanol product

        ethanol = bst.Stream("ethanol", price=price["Ethanol"])

        ### Units ###

        # Split sugar solution

        S301 = units.Splitter("S301", S202 - 0, split=0.10)

        # Concentrate sugars

        F301 = units.MultiEffectEvaporator(
            "F301",
            S301 - 1,
            P=(101325, 73581, 50892, 32777),
            V_definition="First-effect",
            V=0.1,
        )  # fraction evaporated
        P306 = units.Pump("P306", F301 - 0)

        # Mix sugar solutions

        M301 = units.Mixer("M301", [P306 - 0, S301 - 0])

        F301.target_sugar_concentration = 0.23  # wt. % sugar

        @F301.add_bounded_numerical_specification(x0=0, x1=1, xtol=1e-5, ytol=1e-2)
        def sugar_concentration_at_fraction_evaporated(V):
            """Residual of the evaporator specification, solved for the vapour fraction V.

            Args:
                V: Fraction evaporated in the first effect.

            Returns:
                float: Target sugar mass fraction (0.23) minus the actual one after mixer M301.
            """
            F301.V = V
            F301.run_until(
                M301, inclusive=True
            )  # Run all units starting from F301 to M301
            sugar_concentration = M301.outs[0].get_mass_fraction("Sugar")
            return F301.target_sugar_concentration - sugar_concentration

        # Cool for fermentation

        H301 = units.HXutility("H301", M301 - 0, T=295.15)

        # Yeast preparation

        T305 = units.MixTank("T305", yeast)
        T305.tau = 0.1

        # Ethanol Production

        R301 = units.Fermentation(
            "R301",
            [H301 - 0, T305 - 0],
            outs=("CO2", ""),
            tau=9,
            efficiency=0.90,
            N=n_fermentor,  # bovar,var16
        )
        R301.cell_growth_reaction.X = (
            self.cell_growth_reaction_X
        )  # bofidel: 0 for LF, 0.15 for HF

        T301 = units.StorageTank(
            "T301", R301 - 1, tau=beer_tank_tau, vessel_material="Carbon steel"
        )  # bovar var17
        T301.line = "Beer tank"  # Changes name on the diagram

        D301 = units.VentScrubber(
            "D301", ins=(stripping_water, R301 - 0), outs=("vent", ""), gas=("CO2",)
        )

        # Separate 99% of yeast

        centrifuge_split_val = (
            self.centrifuge_split
        )  # bofidel: 0.99 for LF, 0.90 for HF
        C301 = units.SolidsCentrifuge(
            "C301",
            T301 - 0,
            outs=("recycle_yeast", ""),
            moisture_content=0.5,
            split=(centrifuge_split_val, centrifuge_split_val, centrifuge_split_val),
            order=("Ethanol", "Glucose", "DryYeast"),
            solids=("DryYeast",),
        )
        C301.split[:] = 1.0 - C301.split

        # Add bottoms from scrubber

        M302 = units.Mixer("M302", [C301 - 1, D301 - 1])
        P301 = units.Pump("P301", M302 - 0)

        # Heat up before beer column
        # Exchange heat with stillage

        bottoms_product = (
            bst.Stream()
        )  # Bottoms product from beer column, connect later
        H302 = units.HXprocess(
            "H302", [P301 - 0, bottoms_product], U=self.hx_process_U
        )  # bofidel: 0.95 for HF, 1.3 for LF

        # Beer column

        D302 = units.BinaryDistillation(
            "D302",
            H302 - 0,
            P=col1_P_bar * 101325,  # BOVARS: vars 18-21
            Lr=col1_Lr,
            Hr=col1_Hr,  # Light and heavy key recoveries
            LHK=("Ethanol", "Water"),  # Light and heavy key
            k=col1_k,  # Ratio of actual reflux over minimum reflux
            Rmin=0.001,  # Minimum allowable reflux ratio
        )
        D302.tray_material = "Stainless steel 304"
        D302.vessel_material = "Stainless steel 304"
        D302.reboiler.U = self.beer_reboiler_U  # bofidel: 1.4 for HF, 1.85 for LF
        P302 = units.Pump("P302", D302 - 1, bottoms_product)

        # Mix ethanol Recycle (Set-up)

        molecular_sieve_recycle = bst.Stream()
        M303 = units.Mixer("M303", [D302 - 0, molecular_sieve_recycle])

        D303 = units.BinaryDistillation(
            "D303",
            M303 - 0,
            P=col2_P_bar  # BOVAR, vars 22-25
            * 101325,  # Higher pressure to enable heat exchanger between condenser and multi-effect evaporator
            y_top=col2_y_top,
            x_bot=col2_x_bot,
            k=col2_k,
            LHK=("Ethanol", "Water"),
            tray_material="Stainless steel 304",
            vessel_material="Stainless steel 304",
            is_divided=True,
        )

        D303.reboiler.U = 1.85
        P303 = units.Pump("P303", D303 - 1)

        # Superheat vapor for mol sieve

        H303 = units.HXutility("H303", D303 - 0, T=115 + 273.15, V=1, heat_only=True)

        # Molecular sieve

        U301 = units.MolecularSieve(
            "U301",
            H303 - 0,
            [molecular_sieve_recycle, ""],
            split=dict(Ethanol=0.1621, Water=0.925),
        )

        # Condense ethanol product

        H304 = units.HXutility("H304", U301 - 1, V=0, T=340.0)
        T302 = units.StorageTank(
            "T302",
            H304 - 0,
            tau=7 * 24,  # 1 week storage capacity
            vessel_type="Floating roof",
            vessel_material="Carbon steel",
        )
        P304 = units.Pump("P304", T302 - 0)

        # Storage for denaturant

        T303 = units.StorageTank(
            "T303",
            denaturant,
            tau=7 * 24,
            vessel_type="Floating roof",
            vessel_material="Carbon steel",
        )
        P305 = units.Pump("P305", T303 - 0)

        # Denatured ethanol product

        T304 = units.MixTank("T304", [P305 - 0, P304 - 0], outs=ethanol)
        T304.tau = 0.10  # 6 min residence time

        # Waste water

        M305 = units.Mixer("M305", [P303 - 0, F301 - 1], outs="wastewater")

        @P304.add_specification(run=True)
        def adjust_denaturant():
            """Specification on pump P304: set the octane denaturant flow from the ethanol product."""
            pure_ethanol = P304.ins[0]
            denaturant.imol["Octane"] = 0.021 * pure_ethanol.F_mass / 114.232

        s = bst.main_flowsheet.stream
        BT = bst.BoilerTurbogenerator(
            "BT",
            (U202 - 0, "", "boiler_makeup_water", "natural_gas", "", ""),
            boiler_efficiency=self.boiler_efficiency,  # bofidel: 0.80 for HF, 0.99 for LF
            turbogenerator_efficiency=self.turbogenerator_efficiency,  # bofidel: 0.85 for HF, 0.99 for LF
        )
        CT = bst.CoolingTower("CT")
        makeup_water = bst.Stream("makeup_water", price=0.000254)
        CWP = bst.ChilledWaterPackage("CWP")
        PWC = bst.ProcessWaterCenter(
            "PWC",
            ins=(bst.Stream(), makeup_water, P303 - 0),
            outs=(),
            makeup_water_streams=(s.cooling_tower_makeup_water, s.boiler_makeup_water),
            process_water_streams=(
                imbibition_water,
                rvf_wash_water,
                stripping_water,
                s.cooling_tower_makeup_water,
                s.boiler_makeup_water,
            ),
        )
        HXN = bst.HeatExchangerNetwork("HXN", units=[F301, D303.condenser])

        sugarcane_sys = bst.main_flowsheet.create_system("sugarcane_sys")
        try:
            sugarcane_sys.simulate()
            sugarcane_tea = sc.create_tea(sugarcane_sys)
            irr = sugarcane_tea.solve_IRR()
            print(f"irr is {irr}")
            try:
                if math.isfinite(irr) and 0 <= irr <= 1:
                    return irr
                else:
                    return 0.0001
            except Exception:
                return 0.0001
        except Exception:
            return 0.0001
