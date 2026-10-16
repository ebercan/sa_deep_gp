"""Synthetic multi-step process benchmarks used in the ESCAPE-36 poster.

All benchmarks are algebraic, so evaluations are fast, but they mimic the hierarchical
step-to-step dependencies of real processes. Each one has a high-fidelity (HF) mode,
the expensive "plant" that BO queries, and a low-fidelity (LF) mode, cheap per-step
data used for the DGP prior means. All inputs and outputs are dimensionless.

- `DiscontinuousPolymerization` ("Polymerization") and `DiscontinuousPolymerizationV2`
  ("PolymerizationV2"): 3 inputs, mixing -> thermal -> reaction. Deterministic.
- `ComplexPharmaFlowsheet` ("PharmaV1") and `ComplexPharmaFlowsheetV2` ("PharmaV2"):
  5 inputs, activation -> reactor -> nanofiltration -> crystallisation. HF mode adds
  heteroscedastic noise drawn from NumPy's global RNG.

Use `get_benchmark(name)` to build one by its short name.

Note:
    Importing this module calls `warnings.filterwarnings("ignore")`, which silences all
    warnings in the process.
"""

import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore")


class BenchmarkBase:
    """Common interface of the multi-step benchmarks.

    Attributes:
        device: torch device of returned tensors and of `bounds`.
        dtype: torch dtype of returned tensors and of `bounds`.
        bounds: Box constraints, tensor of shape (2, input_dim) (row 0 lower, row 1 upper).
            Set by subclasses.
    """
    def __init__(self, device=None, dtype=torch.double):
        """Set device and dtype.

        Args:
            device: torch device. None selects CUDA if available, else the CPU.
            dtype: torch dtype of returned tensors.
        """
        # Auto-detect device if none provided, robust for HPC vs Local M3
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.dtype = dtype
        self.bounds = None

    @property
    def input_dim(self):
        """Number of process inputs D.

        Returns:
            int: The input dimension.

        Raises:
            NotImplementedError: Always, in the base class.
        """
        raise NotImplementedError

    @property
    def output_dim(self):
        """Number of objectives.

        Returns:
            int: Always 1 (the final yield).
        """
        return 1

    @property
    def step_info(self):
        """Connectivity of the process steps.

        Returns:
            list[dict]: One dict per step with "global_inputs" (indices of x used in this
            step), "prev_outputs" (list of (step_idx, output_idx) tuples that feed this step)
            and "output_dim" (number of outputs of this step).

        Raises:
            NotImplementedError: Always, in the base class.
        """
        raise NotImplementedError

    def get_meta(self):
        """Summarise the benchmark structure.

        Returns:
            dict: "input_dim", "output_dim", "n_steps" and "structure" (the `step_info` list).
        """
        return {
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "n_steps": len(self.step_info),
            "structure": self.step_info
        }

    def _prep_input(self, x):
        """Convert an input to a 2-D NumPy array for the physics functions.

        Args:
            x: Tensor or array of shape (N, D) or (D,).

        Returns:
            np.ndarray: Shape (N, D), detached and on the CPU.
        """
        if torch.is_tensor(x):
            x = x.detach().cpu().numpy()
        if x.ndim == 1:
            x = x.reshape(1, -1)
        return x

    def _to_tensor(self, x):
        """Convert a NumPy array to a tensor on the benchmark's device and dtype.

        Args:
            x: Array-like.

        Returns:
            torch.Tensor: Same shape as `x`.
        """
        return torch.tensor(x, device=self.device, dtype=self.dtype)

    # --- Abstract Methods ---
    def evaluate_step(self, x_in, step_index, mode='HF'):
        """Evaluate one process step on given step inputs.

        Args:
            x_in: Step inputs, shape (N, k). The column layout depends on the step.
            step_index: 0-based step index.
            mode: "HF" or "LF".

        Raises:
            NotImplementedError: Always, in the base class.
        """
        raise NotImplementedError

    def observe(self, x, mode='HF'):
        """Evaluate the full process (final yield).

        Args:
            x: Process inputs, shape (N, D) or (D,).
            mode: "HF" or "LF".

        Raises:
            NotImplementedError: Always, in the base class.
        """
        raise NotImplementedError

    def observe_all_steps(self, x, mode='HF'):
        """Evaluate the full process and return every intermediate.

        Args:
            x: Process inputs, shape (N, D) or (D,).
            mode: "HF" or "LF".

        Raises:
            NotImplementedError: Always, in the base class.
        """
        raise NotImplementedError


class DiscontinuousPolymerization(BenchmarkBase):
    """3-D discontinuous polymerisation benchmark ("Polymerization").

    Inputs, all in [0, 1]: x0 catalyst amount, x1 monomer amount, x2 temperature setting.
    Steps:

    1. Mixing: effective catalyst concentration C_eff = x0 * (1 - exp(-5 x1)).
    2. Thermal: system temperature T_sys = x2 (pass-through in this version).
    3. Reaction: Arrhenius-like base yield 4 C_eff exp(-2 / (T_sys + 0.1)), plus a
       gel-effect oscillation that switches on above T_sys = 0.5. Above the boundary
       T_sys > 0.9 - 0.8 C_eff^2 + shift the yield drops to 10 % (a discontinuity).

    HF and LF differ in the boundary shift (0 vs 0.1) and the oscillation frequency
    (20 vs 15). There is no noise.
    """
    def __init__(self, **kwargs):
        """Create the benchmark with bounds [0, 1]^3.

        Args:
            **kwargs: Passed to `BenchmarkBase` (device, dtype).
        """
        super().__init__(**kwargs)
        self.bounds = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
                                   device=self.device, dtype=self.dtype)

    @property
    def input_dim(self):
        """Number of process inputs.

        Returns:
            int: 3.
        """
        return 3

    @property
    def step_info(self):
        """Connectivity: mixing (x0, x1) -> thermal (x2, C_eff) -> reaction (C_eff, T_sys).

        Returns:
            list[dict]: See `BenchmarkBase.step_info`.
        """
        return [
            # Step 0: Mixing -> Returns C_eff (scalar)
            {"global_inputs": [0, 1], "prev_outputs": [], "output_dim": 1},
            # Step 1: Thermal -> Returns T_sys (scalar)
            {"global_inputs": [2], "prev_outputs": [(0, 0)], "output_dim": 1},
            # Step 2: Reaction -> Returns Yield (scalar)
            {"global_inputs": [], "prev_outputs": [(0, 0), (1, 0)], "output_dim": 1}
        ]

    def _get_params(self, mode):
        """Fidelity-dependent parameters.

        Args:
            mode: "HF" or anything else, which is treated as LF.

        Returns:
            dict: "boundary_shift" (0.0 HF, 0.1 LF) and "wiggle_freq" (20.0 HF, 15.0 LF).
        """
        return {'boundary_shift': 0.0 if mode == 'HF' else 0.1,
                'wiggle_freq': 20.0 if mode == 'HF' else 15.0}

    # --- Physics (UNCHANGED) ---
    def step_1_mixing(self, x_cat, x_mono, mode='HF'):
        """Mixing step: effective catalyst concentration.

        Args:
            x_cat: Catalyst amount, shape (N,).
            x_mono: Monomer amount, shape (N,).
            mode: Unused; the step is the same in both fidelities.

        Returns:
            np.ndarray: C_eff, shape (N,).
        """
        return x_cat * (1 - np.exp(-5 * x_mono))

    def step_2_thermal(self, C_in, x_temp, mode='HF'):
        """Thermal step: system temperature (pass-through in this version).

        Args:
            C_in: Effective catalyst concentration, shape (N,). Unused here.
            x_temp: Temperature setting, shape (N,).
            mode: Unused.

        Returns:
            np.ndarray: T_sys = x_temp, shape (N,).
        """
        return x_temp

    def step_3_reaction(self, C_eff, T_sys, mode='HF'):
        """Reaction step: polymer yield with gel-effect oscillation and a discontinuity.

        Args:
            C_eff: Effective catalyst concentration, shape (N,).
            T_sys: System temperature, shape (N,).
            mode: "HF" or "LF" (see `_get_params`).

        Returns:
            np.ndarray: Yield, shape (N,). It can be negative where the oscillation term
            outweighs the base yield.
        """
        p = self._get_params(mode)
        k = np.exp(-2.0 / (T_sys + 0.1))
        base_yield = 4.0 * C_eff * k
        gel = 1 / (1 + np.exp(-15 * (T_sys - 0.5)))
        freq = p['wiggle_freq'] * (1 + 2 * C_eff)
        wiggle = gel * 0.5 * np.sin(freq * T_sys) * np.cos(freq * C_eff) * T_sys
        raw = base_yield + wiggle
        limit = 0.9 - 0.8 * (C_eff ** 2) + p['boundary_shift']
        return np.where(T_sys > limit, 0.1 * raw, raw)

    def evaluate_step(self, x_in, step_index, mode='HF'):
        """Evaluate a single step on explicit step inputs.

        Args:
            x_in: Step inputs, shape (N, 2) or (2,). Step 0 takes [x_cat, x_mono], step 1
                [C_in, x_temp] and step 2 [C_eff, T_sys].
            step_index: 0, 1 or 2.
            mode: "HF" or "LF".

        Returns:
            np.ndarray: Step output, shape (N,).

        Raises:
            ValueError: If `step_index` is not 0, 1 or 2.
        """
        x = self._prep_input(x_in)
        if step_index == 0:
            return self.step_1_mixing(x[:, 0], x[:, 1], mode)
        elif step_index == 1:
            return self.step_2_thermal(x[:, 0], x[:, 1], mode)
        elif step_index == 2:
            return self.step_3_reaction(x[:, 0], x[:, 1], mode)
        else:
            raise ValueError(f"Invalid step {step_index}")

    def observe(self, x, mode='HF'):
        """Evaluate the full process.

        Args:
            x: Inputs, tensor or array of shape (N, 3) or (3,).
            mode: "HF" or "LF".

        Returns:
            torch.Tensor: Yield, shape (N, 1).
        """
        x = self._prep_input(x)
        y1 = self.step_1_mixing(x[:, 0], x[:, 1], mode)
        y2 = self.step_2_thermal(y1, x[:, 2], mode)
        y3 = self.step_3_reaction(y1, y2, mode)
        return self._to_tensor(y3).unsqueeze(-1)

    def observe_all_steps(self, x, mode='HF'):
        """Evaluate the full process and return every intermediate.

        Args:
            x: Inputs, tensor or array of shape (N, 3) or (3,).
            mode: "HF" or "LF".

        Returns:
            list[torch.Tensor]: [C_eff, T_sys, yield], each of shape (N, 1).
        """
        x = self._prep_input(x)
        y1 = self.step_1_mixing(x[:, 0], x[:, 1], mode)
        y2 = self.step_2_thermal(y1, x[:, 2], mode)
        y3 = self.step_3_reaction(y1, y2, mode)

        # FIX: Ensure intermediate scalars are returned as (N, 1) to match output_dim=1
        return [
            self._to_tensor(y1).view(-1, 1),
            self._to_tensor(y2).view(-1, 1),
            self._to_tensor(y3).view(-1, 1)
        ]


class DiscontinuousPolymerizationV2(DiscontinuousPolymerization):
    # Inherits input_dim and step_info (same connectivity)

    """3-D polymerisation benchmark, version 2 ("PolymerizationV2").

    Same inputs and connectivity as `DiscontinuousPolymerization`, with different physics.
    The thermal step saturates (heat-exchanger efficiency, thermal mass grows with C_eff).
    The reaction step has main and side reactions and a viscosity safety trip, which sets
    the yield to 0.05. The yield is otherwise floored at 0. There is no noise.

    Note:
        In the stored runs, every run of both methods reached the same best value (about
        0.95), so this benchmark does not separate the methods.
    """
    def _get_params(self, mode):
        """Fidelity-dependent parameters.

        Args:
            mode: "HF" or anything else, which is treated as LF.

        Returns:
            dict: "k_main_pre" (8.0 / 8.5; returned but unused, because `step_3_reaction`
            hard-codes 7.0 / 7.5), "k_side_pre" (2.0 / 1.5), "hx_eff" (0.8 / 1.0) and
            "visc_lim" (2.5 / 3.0), given as HF / LF.
        """
        return {
            'k_main_pre': 8.0 if mode == 'HF' else 8.5,
            'k_side_pre': 2.0 if mode == 'HF' else 1.5,
            'hx_eff': 0.8 if mode == 'HF' else 1.0,
            'visc_lim': 2.5 if mode == 'HF' else 3.0
        }

    def step_2_thermal(self, C_in, x_temp, mode='HF'):
        """Thermal step with saturation: T_sys = tanh(2.5 x_temp hx_eff / (1 + 0.5 C_in)).

        Args:
            C_in: Effective catalyst concentration, shape (N,).
            x_temp: Temperature setting, shape (N,).
            mode: "HF" or "LF".

        Returns:
            np.ndarray: T_sys, shape (N,).
        """
        p = self._get_params(mode)
        mass = 1.0 + 0.5 * C_in
        return np.tanh(2.5 * x_temp * p['hx_eff'] / mass)

    def step_3_reaction(self, C_eff, T_sys, mode='HF'):
        """Reaction step with a side reaction and a viscosity safety trip.

        The main rate is k_main = pre * C_eff * exp(-1.5 / (T_sys + 0.05)), with pre 7.0 (HF)
        or 7.5 (LF). The yield is max(0, k_main - 0.5 * side rate). If the viscosity
        k_main^2 / (T_sys + 0.05) exceeds "visc_lim", the yield is 0.05.

        Args:
            C_eff: Effective catalyst concentration, shape (N,).
            T_sys: System temperature, shape (N,).
            mode: "HF" or "LF".

        Returns:
            np.ndarray: Yield, shape (N,), >= 0.
        """
        p = self._get_params(mode)
        T_p = T_sys + 0.05
        k_main = (7.0 if mode == 'HF' else 7.5) * C_eff * np.exp(-1.5 / T_p)
        rate_side = p['k_side_pre'] * (C_eff ** 2) * np.exp(-0.8 / T_p)
        raw = k_main - 0.5 * rate_side
        visc = (k_main ** 2) / T_p

        # Safety Trip logic handles scalars or arrays
        is_safe = visc <= p['visc_lim']
        res = np.where(is_safe, np.maximum(0.0, raw), 0.05)
        return res


class ComplexPharmaFlowsheet(BenchmarkBase):
    """5-D pharmaceutical flowsheet benchmark ("PharmaV1").

    Inputs: x0 metal concentration in [0, 0.02], x1 ligand concentration in [0, 0.02],
    x2 reactor temperature in [0, 1], x3 nanofiltration pressure in [0, 1] and x4
    residence time in [0, 1] (tau = 0.5 + 4.5 x4). Temperature, pressure and time are
    normalised. Steps:

    1. Activation: metal-ligand speciation into ML3 (active), ML2 and free ligand.
    2. Reactor: product, degradant and side product. Free ligand inhibits, a mixing term
       oscillates with residence time, and temperatures above 0.95 trip a safety shutdown.
    3. Nanofiltration: permeate of each species. The flux is a soft minimum of the
       pressure-driven flux and a limiting flux at a critical pressure.
    4. Crystallisation: final yield, with a polymorph "chirp" and oiling out at high
       impurity fractions.

    HF and LF differ in the degradation rate (1.5 vs 1.0). HF also adds heteroscedastic
    noise in the crystallisation step, drawn from NumPy's global RNG.

    Note:
        The yield can be negative. The critical pressure 0.8 - 2 * side / total is not
        floored, so with much side product the flux, and with it the yield, drops below
        zero. In a uniform random sample of the input box, about 70 % of LF yields are
        negative, though small (down to about -0.01). Version 2 floors the critical
        pressure at 0.1.
    """
    def __init__(self, **kwargs):
        """Create the benchmark with bounds [0, 0.02]^2 x [0, 1]^3.

        Args:
            **kwargs: Passed to `BenchmarkBase` (device, dtype).
        """
        super().__init__(**kwargs)
        self.bounds = torch.tensor([[0., 0., 0., 0., 0.], [0.02, 0.02, 1., 1., 1.]],
                                   device=self.device, dtype=self.dtype)

    @property
    def input_dim(self):
        """Number of process inputs.

        Returns:
            int: 5.
        """
        return 5

    @property
    def step_info(self):
        """Connectivity of the four pharma steps.

        Activation (x0, x1) -> reactor (x2, x4) -> nanofiltration (x3) -> crystallisation. The
        crystallisation step takes only the permeate streams.

        Returns:
            list[dict]: See `BenchmarkBase.step_info`.
        """
        return [
            # Step 0: Activation -> Returns [ML3, ML2, FreeL] (3D)
            {"global_inputs": [0, 1], "prev_outputs": [], "output_dim": 3},
            # Step 1: Reactor -> Returns [Prod, Deg, Side] (3D)
            {"global_inputs": [2, 4], "prev_outputs": [(0, 0), (0, 1), (0, 2)], "output_dim": 3},
            # Step 2: Nano -> Returns [P_perm, D_perm, S_perm] (3D)
            {"global_inputs": [3], "prev_outputs": [(1, 0), (1, 1), (1, 2)], "output_dim": 3},
            # Step 3: Cryst -> Returns Yield (1D)
            {"global_inputs": [], "prev_outputs": [(2, 0), (2, 1), (2, 2)], "output_dim": 1}
        ]

    def _get_params(self, mode):
        """Fidelity-dependent parameters.

        Args:
            mode: "HF" or anything else, which is treated as LF.

        Returns:
            dict: "deg_rate" (1.5 HF, 1.0 LF) and "noise" (True for HF only).
        """
        return {'deg_rate': 1.5 if mode == 'HF' else 1.0, 'noise': True if mode == 'HF' else False}

    # --- Physics (UNCHANGED) ---
    def step_1_activation(self, x_m, x_l, mode='HF'):
        """Activation step: closed-form metal-ligand speciation.

        Args:
            x_m: Metal concentration, shape (N,) or scalar.
            x_l: Ligand concentration, shape (N,) or scalar.
            mode: Unused.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: (ML3, ML2, free ligand), each shaped like
            the broadcast inputs.
        """
        ml2_pot = np.minimum(x_m, x_l / 2.0)
        l_rem = np.maximum(0, x_l - 2.0 * ml2_pot)
        ml3 = np.minimum(ml2_pot, l_rem)
        ml2 = ml2_pot - ml3
        fl = np.maximum(0, l_rem - ml3)
        return ml3, ml2, fl

    def step_2_reactor(self, ml3, ml2, fl, temp, time, mode='HF'):
        """Reactor step: product, degradant and side-product concentrations.

        Args:
            ml3: Active complex concentration, shape (N,).
            ml2: ML2 complex concentration, shape (N,).
            fl: Free ligand concentration, shape (N,).
            temp: Normalised temperature, shape (N,). Above 0.95 the safety trip sets the
                product to 0 and the degradant to 20 ML3 + 40 ML2.
            time: Normalised residence time, shape (N,).
            mode: "HF" or "LF" (degradation rate).

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: (product, degradant, side product).
        """
        p = self._get_params(mode)
        T_r = temp + 0.1;
        tau = 0.5 + 4.5 * time
        mix = np.clip(0.6 + 0.35 * np.sin(14 * time - 0.5) + 0.1 * time, 0.1, 1.0)

        inhib = 1.0 + 200.0 * fl * (1.0 + 5.0 * np.exp(2.0 * (temp - 0.5)))
        eff_ml3 = ml3 / inhib

        k_main = (600.0 * eff_ml3 * np.exp(-1.5 / T_r) + 150.0 * ml2 * np.exp(-1.2 / T_r)) * mix
        k_side = (300.0 * ml2 * np.exp(-1.2 / T_r) + 100.0 * eff_ml3 * np.exp(-1.5 / T_r)) * (1.2 - mix)

        k_deg = (0.2 * p['deg_rate']) * (np.exp(-3.0 / T_r) + 0.001)
        decay = (1.0 - np.exp(-2.0 * np.exp(-5. / T_r) * tau)) / (2.0 * np.exp(-5. / T_r) * tau + 1e-9)

        real_k = k_main * decay
        C_A = 1.0 / (1.0 + (real_k + k_side) * tau)
        C_P_raw = real_k * C_A * tau
        C_D = k_deg * C_P_raw * (tau ** 1.7)

        # Safety Check
        is_unsafe = (temp > 0.95) if isinstance(temp, np.ndarray) else (temp > 0.95)
        crash_val = 20.0 * ml3 + 40.0 * ml2

        return np.where(is_unsafe, 0.0, np.maximum(0, C_P_raw - C_D)), \
            np.where(is_unsafe, crash_val, C_D), \
            k_side * C_A * tau

    def step_3_nanofiltration(self, prod, deg, side, press, mode='HF'):
        """Nanofiltration step: permeate concentrations.

        Args:
            prod: Product concentration, shape (N,).
            deg: Degradant concentration, shape (N,).
            side: Side-product concentration, shape (N,).
            press: Normalised pressure, shape (N,) or scalar.
            mode: Unused.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: Permeate (product, degradant, side
            product). These can be negative when the critical pressure is negative (see the
            class note).
        """
        tot = prod + 0.1 * deg + 0.5 * side + 1e-9
        visc = 1.0 + 8.0 * (tot ** 2)
        P_c = 0.8 - 2.0 * (side / tot)

        J = (1.5 * press) / visc;
        J_l = (1.5 * P_c) / visc
        flux = -np.log(np.exp(-10 * J) + np.exp(-10 * J_l)) / 10.0

        over_P = np.maximum(0, press - P_c)
        pen = 1.0 + 5.0 * (over_P ** 2)

        return (4 * prod * flux * (0.98 / (pen * 1.5)), 4 * deg * flux * 0.99, 4 * side * flux * (0.6 / pen))

    def step_4_crystallization(self, p_p, d_p, s_p, mode='HF'):
        """Crystallisation step: final yield.

        With impurity fraction imp = (d_p + s_p) / total, the yield is
        0.95 * p_p * efficiency(imp) * oiling_out(imp). The efficiency has a high-frequency
        chirp, and oiling out is a sigmoid centred at imp = 0.25. In HF mode, Gaussian noise
        with std 0.05 * ramp * oil * 0.95 * p_p is added and the result is clipped to
        [0, 0.95 p_p]. For negative p_p, np.clip returns the upper bound, so the yield stays
        negative.

        Args:
            p_p: Product in the permeate, shape (N,).
            d_p: Degradant in the permeate, shape (N,).
            s_p: Side product in the permeate, shape (N,).
            mode: "HF" (noisy) or "LF" (noise-free).

        Returns:
            np.ndarray: Yield, shape (N,).
        """
        p = self._get_params(mode)
        tot = p_p + d_p + s_p + 1e-9;
        imp = (d_p + s_p) / tot

        chirp = 0.6 + 0.15 * np.cos(400.0 * (imp ** 2) * 1.12)
        ramp = np.clip(imp * 3.5, 0., 1.)
        eff = (1. - ramp) + ramp * chirp
        oil = 1.0 / (1.0 + np.exp(50.0 * (imp - 0.25)))

        y_phys = p_p * 0.95 * eff * oil

        if p['noise']:
            scale = 0.05 * ramp * oil
            noise = np.random.normal(0, 1.0, size=y_phys.shape) if isinstance(y_phys, np.ndarray) else np.random.normal(
                0, 1.)
            return np.clip(y_phys + noise * scale * p_p * 0.95, 0.0, p_p * 0.95)
        return y_phys

    # --- Execution ---
    def evaluate_step(self, x_in, step_index, mode='HF'):
        """Evaluate a single step on explicit step inputs.

        Args:
            x_in: Step inputs, shape (N, k). Step 0 takes [x_m, x_l], step 1 [ml3, ml2, fl,
                temp, time], step 2 [prod, deg, side, press] and step 3 [p_p, d_p, s_p].
            step_index: 0, 1, 2 or 3.
            mode: "HF" or "LF".

        Returns:
            np.ndarray: Shape (N, 3) for steps 0-2 and (N,) for step 3.

        Raises:
            ValueError: If `step_index` is not in 0..3.
        """
        x = self._prep_input(x_in)
        if step_index == 0:
            m3, m2, fl = self.step_1_activation(x[:, 0], x[:, 1], mode)
            return np.stack([m3, m2, fl], axis=1)
        elif step_index == 1:
            p, d, s = self.step_2_reactor(x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4], mode)
            return np.stack([p, d, s], axis=1)
        elif step_index == 2:
            p, d, s = self.step_3_nanofiltration(x[:, 0], x[:, 1], x[:, 2], x[:, 3], mode)
            return np.stack([p, d, s], axis=1)
        elif step_index == 3:
            return self.step_4_crystallization(x[:, 0], x[:, 1], x[:, 2], mode)
        else:
            raise ValueError(f"Invalid step {step_index}")

    def observe(self, x, mode='HF'):
        """Evaluate the full process.

        Args:
            x: Inputs, tensor or array of shape (N, 5) or (5,).
            mode: "HF" (noisy) or "LF".

        Returns:
            torch.Tensor: Yield, shape (N, 1).
        """
        x = self._prep_input(x)
        y1 = self.step_1_activation(x[:, 0], x[:, 1], mode)
        y2 = self.step_2_reactor(y1[0], y1[1], y1[2], x[:, 2], x[:, 4], mode)
        y3 = self.step_3_nanofiltration(y2[0], y2[1], y2[2], x[:, 3], mode)
        y4 = self.step_4_crystallization(y3[0], y3[1], y3[2], mode)
        return self._to_tensor(y4).unsqueeze(-1)

    def observe_all_steps(self, x, mode='HF'):
        """Evaluate the full process and return every intermediate.

        Args:
            x: Inputs, tensor or array of shape (N, 5) or (5,).
            mode: "HF" or "LF".

        Returns:
            list[torch.Tensor]: Activation (N, 3), reactor (N, 3), nanofiltration (N, 3) and
            yield (N, 1).
        """
        x = self._prep_input(x)
        y1 = self.step_1_activation(x[:, 0], x[:, 1], mode)
        y2 = self.step_2_reactor(y1[0], y1[1], y1[2], x[:, 2], x[:, 4], mode)
        y3 = self.step_3_nanofiltration(y2[0], y2[1], y2[2], x[:, 3], mode)
        y4 = self.step_4_crystallization(y3[0], y3[1], y3[2], mode)
        # Stacks are correct here because Pharma outputs are tuples of arrays
        return [
            self._to_tensor(np.stack(y1, axis=1)),
            self._to_tensor(np.stack(y2, axis=1)),
            self._to_tensor(np.stack(y3, axis=1)),
            self._to_tensor(y4).view(-1, 1)
        ]


class ComplexPharmaFlowsheetV2(ComplexPharmaFlowsheet):
    # Inherits Input Dim / Step Info

    """5-D pharmaceutical flowsheet, version 2 ("PharmaV2").

    Same inputs and connectivity as `ComplexPharmaFlowsheet`, with these changes: ML2
    aggregation in the activation step, a critical pressure floored at 0.1 and a lower
    side-product passage in nanofiltration, and a "seeding hill" crystallisation efficiency
    with the optimum near 12 % impurity. HF noise is doubled.

    Note:
        Yields can still be slightly negative at very low pressure. The flux is the soft
        minimum -log(exp(-10 J) + exp(-10 J_lim)) / 10, which is below zero when both
        fluxes are near zero. This affects about 2 % of a uniform random sample.
    """
    def _get_params(self, mode):
        """Fidelity-dependent parameters (same values as version 1).

        Args:
            mode: "HF" or anything else, which is treated as LF.

        Returns:
            dict: "deg_rate" (1.5 HF, 1.0 LF) and "noise" (True for HF only).
        """
        return {'deg_rate': 1.5 if mode == 'HF' else 1.0, 'noise': True if mode == 'HF' else False}

    def step_1_activation(self, x_m, x_l, mode='HF'):
        """Activation step with aggregation: ML2 above 0.003 loses 80 % of the excess.

        Args:
            x_m: Metal concentration, shape (N,) or scalar.
            x_l: Ligand concentration, shape (N,) or scalar.
            mode: Unused.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: (ML3, ML2 after aggregation, free
            ligand).
        """
        # V2: Aggregation
        ml2_pot = np.minimum(x_m, x_l / 2.0)
        l_rem = np.maximum(0, x_l - 2.0 * ml2_pot)
        ml3 = np.minimum(ml2_pot, l_rem)
        ml2 = ml2_pot - ml3
        fl = np.maximum(0, l_rem - ml3)
        agg = np.maximum(0, ml2 - 0.003) * 0.8
        return ml3, np.maximum(0, ml2 - agg), fl

    def step_3_nanofiltration(self, prod, deg, side, press, mode='HF'):
        """Nanofiltration with the critical pressure max(0.1, 0.8 - 2.5 side / total).

        Args:
            prod: Product concentration, shape (N,).
            deg: Degradant concentration, shape (N,).
            side: Side-product concentration, shape (N,).
            press: Normalised pressure, shape (N,) or scalar.
            mode: Unused.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: Permeate (product, degradant, side
            product).
        """
        # V2: Different P_crit logic
        tot = prod + 0.1 * deg + 0.5 * side + 1e-9
        P_c = np.maximum(0.1, 0.8 - 2.5 * (side / tot))
        visc = 1.0 + 8.0 * (tot ** 2)

        J = (1.5 * press) / visc;
        J_l = (1.5 * P_c) / visc
        flux = -np.log(np.exp(-10 * J) + np.exp(-10 * J_l)) / 10.0

        over_P = np.maximum(0, press - P_c)
        pen = 1.0 + 5.0 * (over_P ** 2)

        return (4 * prod * flux * (0.98 / (pen * 1.5)), 4 * deg * flux * 0.99, 4 * side * flux * (0.2 / pen))

    def step_4_crystallization(self, p_p, d_p, s_p, mode='HF'):
        """Crystallisation with a seeding optimum near 12 % impurity.

        The efficiency is a Gaussian "seeding hill" times a chirp that ramps in above 15 %
        impurity, and oiling out is centred at 30 %. In HF mode, noise with std
        0.10 * ramp * oil * p_p is added and the result is clipped to [0, p_p].

        Args:
            p_p: Product in the permeate, shape (N,).
            d_p: Degradant in the permeate, shape (N,).
            s_p: Side product in the permeate, shape (N,).
            mode: "HF" (noisy) or "LF".

        Returns:
            np.ndarray: Yield, shape (N,).
        """
        # V2: Seeding hill
        p = self._get_params(mode)
        tot = p_p + d_p + s_p + 1e-9;
        imp = (d_p + s_p) / tot

        seed = 0.1 + 0.85 * np.exp(-150.0 * (imp - 0.12) ** 2)
        chirp = 0.6 + 0.15 * np.cos(400.0 * (imp ** 2) * 1.12)
        ramp = np.clip((imp - 0.15) * 10.0, 0., 1.)

        eff = (1. - ramp) * seed + ramp * (seed * chirp)
        oil = 1.0 / (1.0 + np.exp(50.0 * (imp - 0.30)))

        y_phys = p_p * eff * oil

        if p['noise']:
            scale = 0.10 * ramp * oil
            noise = np.random.normal(0, 1.0, size=y_phys.shape) if isinstance(y_phys, np.ndarray) else np.random.normal(
                0, 1.)
            return np.clip(y_phys + noise * scale * p_p, 0.0, p_p)
        return y_phys


def get_benchmark(name, **kwargs):
    """Build a benchmark by its short name.

    Args:
        name: "Polymerization", "PolymerizationV2", "PharmaV1" or "PharmaV2".
        **kwargs: Passed to the constructor (device, dtype).

    Returns:
        BenchmarkBase: The benchmark instance.

    Raises:
        ValueError: If `name` is unknown.
    """
    if name == "Polymerization":
        return DiscontinuousPolymerization(**kwargs)
    elif name == "PolymerizationV2":
        return DiscontinuousPolymerizationV2(**kwargs)
    elif name == "PharmaV1":
        return ComplexPharmaFlowsheet(**kwargs)
    elif name == "PharmaV2":
        return ComplexPharmaFlowsheetV2(**kwargs)
    else:
        raise ValueError(f"Unknown benchmark: {name}")
