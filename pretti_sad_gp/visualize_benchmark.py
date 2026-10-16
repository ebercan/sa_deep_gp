"""Interactive Plotly views of the PharmaV1 benchmark's individual steps (helper study).

Run from `pretti_sad_gp/`. The figures open in the browser. Which plots are shown is
chosen in the `__main__` block. All functions take a `ComplexPharmaFlowsheet` instance
`env` and call its step functions in their default HF mode.
"""

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

# Force browser opening
pio.renderers.default = "browser"

# Import the UPDATED Benchmark
from benchmarks import ComplexPharmaFlowsheet


def generate_grid(max_val=1.0, resolution=50):
    """Square grid on [0, max_val]^2.

    Args:
        max_val: Upper end of both axes.
        resolution: Points per axis.

    Returns:
        tuple: (x, y, X, Y), with the 1-D axes of shape (resolution,) and the meshgrids of
        shape (resolution, resolution).
    """
    x = np.linspace(0, max_val, resolution)
    y = np.linspace(0, max_val, resolution)
    X, Y = np.meshgrid(x, y)
    return x, y, X, Y


def plot_hydrodynamic_kinetics(env):
    """Reactor outlet concentrations along residence time.

    Metal, ligand and temperature are held fixed.

    Args:
        env: `ComplexPharmaFlowsheet` instance.
    """
    print("Generating Hydrodynamic Kinetics Plot...")

    # 1. Setup Input Line (Time 0 -> 1)
    resolution = 200
    x_time = np.linspace(0, 1.0, resolution)

    # Constants for other inputs
    fixed_metal = 0.005
    fixed_ligand = 0.015  # 3:1 Ratio
    fixed_temp = 0.65  # Moderate Temp

    # Prepare Inputs
    ml3, ml2, free_l = env.step_1_activation(
        np.full(resolution, fixed_metal),
        np.full(resolution, fixed_ligand)
    )

    x_temp_arr = np.full(resolution, fixed_temp)

    # Run Step 2
    cp, cd, cs = env.step_2_reactor(ml3, ml2, free_l, x_temp_arr, x_time)

    # 2. Plotting
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Trace 1: Main Product (Green) - Shows Baffles
    fig.add_trace(go.Scatter(
        x=x_time, y=cp,
        mode='lines', name='Main Product (Yield)',
        line=dict(color='green', width=4)
    ), secondary_y=False)

    # Trace 2: Side Product (Orange) - Shows Dead Zones
    fig.add_trace(go.Scatter(
        x=x_time, y=cs,
        mode='lines', name='Side Product (Impurity)',
        line=dict(color='orange', width=3, dash='dash')
    ), secondary_y=True)

    # Trace 3: Degradation (Red) - Shows 'Cooking'
    fig.add_trace(go.Scatter(
        x=x_time, y=cd,
        mode='lines', name='Degradation',
        line=dict(color='red', width=2, dash='dot')
    ), secondary_y=True)

    # Annotations to explain Physics
    # Find peaks to label Baffles
    peak_idx = np.argmax(cp)

    fig.add_annotation(
        x=x_time[peak_idx], y=cp[peak_idx],
        text="Baffle Zone<br>(High Mixing)",
        showarrow=True, arrowhead=1, ax=0, ay=-40
    )

    # Find a valley to label Dead Zone
    valley_idx = np.argmin(cp[50:150]) + 50  # Look in middle
    fig.add_annotation(
        x=x_time[valley_idx], y=cp[valley_idx],
        text="Dead Zone<br>(Stagnation)",
        showarrow=True, arrowhead=1, ax=0, ay=40
    )

    fig.update_layout(
        title='Reactor Hydrodynamics: Baffles vs Dead Zones',
        xaxis_title='Residence Time (Normalized)',
        yaxis_title='Main Product Conc (M)',
        template='plotly_white',
        legend=dict(x=0.01, y=0.99),
        height=600
    )

    fig.update_yaxes(title_text="Impurity / Degradation Conc (M)", secondary_y=True)

    fig.show()


def plot_nanofiltration_cliff(env):
    """Permeate product as a surface over pressure and side-product load.

    Args:
        env: `ComplexPharmaFlowsheet` instance.
    """
    print("Generating Nanofiltration Cliff Plot...")

    # Grid: Pressure (0.1 - 1.0) vs Impurity Load (Small to Large)
    res = 60
    x_press = np.linspace(0.1, 1.0, res)
    x_side = np.linspace(0.0, 0.005, res)  # 0 to 0.5% Side Product

    P_grid, S_grid = np.meshgrid(x_press, x_side)

    # Fixed Inputs
    fixed_prod = 0.004
    fixed_deg = 0.0001

    # Run Step 3
    # Note: Flatten for processing, reshape for plotting
    p_flat = np.full(res * res, fixed_prod)
    d_flat = np.full(res * res, fixed_deg)
    s_flat = S_grid.flatten()
    press_flat = P_grid.flatten()

    p_perm, d_perm, s_perm = env.step_3_nanofiltration(p_flat, d_flat, s_flat, press_flat)

    # We plot Permeate Purity (P / Total) to see the crash
    total_perm = p_perm + d_perm + s_perm + 1e-9
    purity = p_perm / total_perm
    purity_reshaped = purity.reshape(res, res)

    # Also plot Flux (Yield)
    yield_reshaped = p_perm.reshape(res, res)

    fig = go.Figure()

    # Surface: Yield (Flux)
    fig.add_trace(go.Surface(
        z=yield_reshaped, x=x_press, y=x_side,
        colorscale='Viridis', name='Yield (Flux)',
        colorbar=dict(x=0, title='Yield'),
        opacity=0.9
    ))

    # Add a 'Cliff' marker line where selectivity collapses
    # We can approximate this visually or just let the surface show it

    fig.update_layout(
        title='Nanofiltration: The Pressure/Fouling Cliff',
        scene=dict(
            xaxis_title='Pressure (Bar)',
            yaxis_title='Side Product Load',
            zaxis_title='Yield (Flux)',
            camera=dict(eye=dict(x=1.5, y=1.5, z=0.5))
        ),
        width=900, height=800,
        margin=dict(l=0, r=0, b=0, t=50)
    )
    fig.show()


def plot_full_process_landscape(env):
    """Final yield over temperature x residence time at fixed metal, ligand and pressure 0.6.

    Args:
        env: `ComplexPharmaFlowsheet` instance.
    """
    print("Generating Full Process Landscape...")

    # Grid
    res = 50
    x_temp, x_time, T_grid, Time_grid = generate_grid(max_val=1.0, resolution=res)

    # Fixed Context
    fixed_metal = 0.005
    fixed_ligand = 0.015  # 3:1
    fixed_pressure = 0.6  # Moderate Pressure

    # Step 1
    ml3, ml2, free_l = env.step_1_activation(
        np.full_like(T_grid, fixed_metal),
        np.full_like(T_grid, fixed_ligand)
    )

    # Step 2
    cp, cd, cs = env.step_2_reactor(ml3, ml2, free_l, T_grid, Time_grid)

    # Step 3
    p_perm, d_perm, s_perm = env.step_3_nanofiltration(
        cp, cd, cs, np.full_like(T_grid, fixed_pressure)
    )

    # Step 4
    final_yield = env.step_4_crystallization(p_perm, d_perm, s_perm)

    fig = go.Figure()

    fig.add_trace(go.Surface(
        z=final_yield, x=x_temp, y=x_time,
        colorscale='Turbo',
        contours_z=dict(show=True, usecolormap=True, highlightcolor="limegreen", project_z=True)
    ))

    fig.update_layout(
        title='Global Optima Landscape (Temp vs Time)',
        scene=dict(
            xaxis_title='Temperature',
            yaxis_title='Residence Time',
            zaxis_title='Final Yield',
            camera=dict(eye=dict(x=-1.5, y=-1.5, z=0.8))
        ),
        width=1000, height=800,
    )
    fig.show()


def plot_step_1_interactive_speciation(env):
    """ML3, ML2 and free-ligand surfaces over metal x ligand concentration.

    Args:
        env: `ComplexPharmaFlowsheet` instance.
    """
    # Grid [0 to 0.02 M]
    x_metal, x_ligand, X_M, X_L = generate_grid(max_val=0.02, resolution=60)

    # Get all species
    Z_ML3, Z_ML2, Z_FreeL = env.step_1_activation(X_M, X_L)

    fig = go.Figure()

    # Surface 1: ML3 (Active) - Green
    fig.add_trace(go.Surface(
        z=Z_ML3, x=x_metal, y=x_ligand,
        colorscale='Greens', name='ML3 (Active)',
        showscale=False, opacity=0.9,
        showlegend=True  # <--- FIX
    ))

    # Surface 2: ML2 (Unstable) - Orange
    fig.add_trace(go.Surface(
        z=Z_ML2, x=x_metal, y=x_ligand,
        colorscale='Oranges', name='ML2 (Unstable)',
        showscale=False, opacity=0.8,
        showlegend=True  # <--- FIX
    ))

    # Surface 3: Free Ligand - Blue
    fig.add_trace(go.Surface(
        z=Z_FreeL, x=x_metal, y=x_ligand,
        colorscale='Blues', name='Free Ligand',
        showscale=False, opacity=0.5,
        showlegend=True  # <--- FIX
    ))

    # Add 3:1 Line for reference
    line_x = np.linspace(0, 0.0066, 50)
    line_y = 3.0 * line_x
    line_z = np.full_like(line_x, 0.006)

    fig.add_trace(go.Scatter3d(
        x=line_x, y=line_y, z=line_z,
        mode='lines', name='Ideal 3:1 Ratio',
        line=dict(color='black', width=5, dash='dash')
    ))

    fig.update_layout(
        title='Step 1 Speciation: Click Legend to Toggle Surfaces',
        scene=dict(
            xaxis_title='Metal (M)',
            yaxis_title='Ligand (M)',
            zaxis_title='Concentration (M)',
            zaxis=dict(range=[0, 0.006]),
            camera=dict(eye=dict(x=-1.5, y=-1.5, z=0.5))
        ),
        # Fix sizing if needed, though usually fine for static plots
        width=900, height=800,
        legend=dict(x=0, y=1)
    )
    fig.show()


def plot_step_2_ratio_explorer(env):
    """Reactor outlet surfaces over temperature x time, with a ligand:metal ratio slider.

    The product surface is shown; the side-product and degradant surfaces can be toggled in
    the legend.

    Args:
        env: `ComplexPharmaFlowsheet` instance.
    """
    # Grid for Temp vs Time
    x_temp, x_time, T_grid, Time_grid = generate_grid(max_val=1.0, resolution=50)

    # Ratios to animate
    ratios = np.linspace(1.0, 5.0, 17)  # 0.25 steps
    fixed_metal = 0.005

    frames = []

    print("Generating Ratio Animation (Yield, Side, Deg)...")
    for r in ratios:
        curr_ligand = fixed_metal * r

        # 1. Step 1 (Get Species)
        ml3, ml2, free_l = env.step_1_activation(fixed_metal, curr_ligand)

        # 2. Step 2 (Reactor Surfaces)
        y2_p, y2_d, y2_s = env.step_2_reactor(ml3, ml2, free_l, T_grid, Time_grid)

        frames.append(go.Frame(
            data=[
                # Trace 0: Yield (Visible)
                go.Surface(z=y2_p, x=x_temp, y=x_time),
                # Trace 1: Side Products (Hidden by default)
                go.Surface(z=y2_s, x=x_temp, y=x_time),
                # Trace 2: Degradation (Hidden by default)
                go.Surface(z=y2_d, x=x_temp, y=x_time)
            ],
            name=f"{r:.2f}"
        ))

    # Initial Data (Ratio = 3.0)
    init_r = 3.0
    ml3, ml2, free_l = env.step_1_activation(fixed_metal, fixed_metal * init_r)
    y2_p, y2_d, y2_s = env.step_2_reactor(ml3, ml2, free_l, T_grid, Time_grid)

    fig = go.Figure(
        data=[
            # Trace 0: Yield
            go.Surface(z=y2_p, x=x_temp, y=x_time,
                       colorscale='Magma', name='Yield (P)',
                       showscale=False, showlegend=True),

            # Trace 1: Side Products (Impurity)
            go.Surface(z=y2_s, x=x_temp, y=x_time,
                       colorscale='Ice', name='Side (S)',
                       visible='legendonly',  # <--- Click legend to see
                       showscale=False, showlegend=True, opacity=0.8),

            # Trace 2: Degradation
            go.Surface(z=y2_d, x=x_temp, y=x_time,
                       colorscale='Hot', name='Degradation (D)',
                       visible='legendonly',  # <--- Click legend to see
                       showscale=False, showlegend=True, opacity=0.8)
        ],
        frames=frames
    )

    sliders = [{
        "pad": {"b": 10, "t": 50},
        "len": 0.9, "x": 0.1,
        "currentvalue": {"prefix": "Ligand:Metal Ratio: "},
        "steps": [
            {
                "args": [[f"{r:.2f}"], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
                "label": f"{r:.2f}",
                "method": "animate"
            } for r in ratios
        ]
    }]

    fig.update_layout(
        title='Reactor Performance Explorer (Temp vs Time)',
        scene=dict(
            xaxis_title='Temperature (0-1)',
            yaxis_title='Time (0-1)',
            zaxis_title='Concentration',
            # FIX: Lock Axis Range and Aspect Ratio
            zaxis=dict(range=[0, 1.0]),
            aspectmode='cube'
        ),
        sliders=sliders,
        # FIX: Hardcode dimensions to prevent jitter
        width=1000,
        height=900,
        margin=dict(l=0, r=0, b=0, t=40),
        legend=dict(x=0.7, y=0.9, bgcolor='rgba(255,255,255,0.5)')
    )
    fig.show()


def plot_reactor_dual_sliders(env):
    """Reactor outlets over the ligand:metal ratio, with time and temperature sliders.

    Args:
        env: `ComplexPharmaFlowsheet` instance.
    """

    # --- 1. SETUP GRID ---
    fixed_metal = 0.003
    ratios = np.linspace(1.0, 6.0, 100)
    ligands = fixed_metal * ratios

    # Grid Resolutions
    n_temps = 21  # 0.0, 0.05, ... 1.0
    temps = np.linspace(0.0, 1.0, n_temps)

    n_times = 11  # 0.0, 0.1, ... 1.0
    times = np.linspace(0.0, 1.0, n_times)

    # --- 2. PRE-CALCULATE ALL DATA ---
    # Structure: data_matrix[time_idx][temp_idx] = (P, D, S arrays)
    print("Calculating Kinetic Landscape (This might take a moment)...")

    # We pre-calculate everything to make the UI snappy
    data_matrix = [[None for _ in range(n_temps)] for _ in range(n_times)]

    # Optimize: Pre-calculate Step 1 (Speciation) once, as it doesn't depend on T or Time
    ml3, ml2, free_l = env.step_1_activation(np.full_like(ratios, fixed_metal), ligands)

    for t_i, time_val in enumerate(times):
        vec_time = np.full_like(ratios, time_val)

        for temp_i, temp_val in enumerate(temps):
            vec_temp = np.full_like(ratios, temp_val)

            # Step 2: Reactor
            cp, cd, cs = env.step_2_reactor(ml3, ml2, free_l, vec_temp, vec_time)
            data_matrix[t_i][temp_i] = (cp, cd, cs)

    # --- 3. INITIALIZE FIGURE WITH TRACES ---
    fig = go.Figure()

    # We create 3 traces (P, S, D) for EVERY Time step.
    # Total traces = 3 * n_times.
    # Initially, only the middle time step is visible.

    initial_temp_idx = 0  # Start Cold
    mid_time_idx = 5  # Start at Time=0.5

    trace_indices = {}  # Map time_idx -> [trace_idx_P, trace_idx_S, trace_idx_D]

    curr_trace_id = 0

    for t_i, time_val in enumerate(times):
        # Get data for this time at Temp=0
        p, d, s = data_matrix[t_i][initial_temp_idx]

        is_visible = (t_i == mid_time_idx)

        # Trace P (Green)
        fig.add_trace(go.Scatter(
            x=ratios, y=p, mode='lines', name=f'Product',
            line=dict(color='green', width=4),
            visible=is_visible
        ))

        # Trace S (Orange)
        fig.add_trace(go.Scatter(
            x=ratios, y=s, mode='lines', name=f'Side Prod',
            line=dict(color='orange', width=3, dash='dash'),
            visible=is_visible
        ))

        # Trace D (Red)
        fig.add_trace(go.Scatter(
            x=ratios, y=d, mode='lines', name=f'Degradation',
            line=dict(color='red', width=3, dash='dot'),
            visible=is_visible
        ))

        trace_indices[t_i] = [curr_trace_id, curr_trace_id + 1, curr_trace_id + 2]
        curr_trace_id += 3

    # --- 4. CREATE FRAMES (For Temperature Animation) ---
    # Each frame updates ALL traces (for all times) to the new Temperature.
    frames = []
    for temp_i, temp_val in enumerate(temps):
        frame_data = []

        # Loop through all time steps and append their new Y-values
        for t_i in range(n_times):
            p, d, s = data_matrix[t_i][temp_i]
            frame_data.append(go.Scatter(y=p))
            frame_data.append(go.Scatter(y=s))
            frame_data.append(go.Scatter(y=d))

        frames.append(go.Frame(data=frame_data, name=f"{temp_val:.2f}"))

    fig.frames = frames

    # --- 5. SLIDER 1: RESIDENCE TIME (Restyle Visibility) ---
    # This slider toggles which set of 3 traces is visible.
    time_steps = []
    for t_i, time_val in enumerate(times):
        # Construct visibility array: All False except the 3 for this time step
        vis_array = [False] * (3 * n_times)
        indices = trace_indices[t_i]
        vis_array[indices[0]] = True
        vis_array[indices[1]] = True
        vis_array[indices[2]] = True

        step = {
            'method': 'restyle',
            'args': ['visible', vis_array],
            'label': f"{time_val:.1f}"
        }
        time_steps.append(step)

    sliders_time = {
        'active': mid_time_idx,
        'currentvalue': {"prefix": "Residence Time: "},
        'pad': {"t": 50},
        'steps': time_steps,
        'y': 1.15  # Position at top
    }

    # --- 6. SLIDER 2: TEMPERATURE (Animate Data) ---
    # This slider triggers the frames.
    temp_steps = []
    for temp_val in temps:
        step = {
            'method': 'animate',
            'args': [[f"{temp_val:.2f}"],
                     {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
            'label': f"{temp_val:.2f}"
        }
        temp_steps.append(step)

    sliders_temp = {
        'active': 0,
        'currentvalue': {"prefix": "Temperature: "},
        'pad': {"t": 50},
        'steps': temp_steps,
        'y': 0.05  # Position at bottom
    }

    # --- 7. FINAL LAYOUT ---
    fig.update_layout(
        title='Reactor Kinetics: Interaction of Time & Temp',
        xaxis_title='Ligand : Metal Ratio',
        yaxis_title='Concentration (M)',
        yaxis=dict(range=[0, 1.0]),
        sliders=[sliders_time, sliders_temp],
        template='plotly_white',
        margin=dict(t=100, b=100),  # Space for sliders
        legend=dict(x=0.8, y=0.95)
    )

    # Reference Line
    fig.add_vline(x=3.0, line_dash="dot", line_color="gray", annotation_text="Ideal 3:1")

    fig.show()


def plot_interactive_yield_surface(env):
    """Final yield over temperature x time, with a pressure slider and preset dropdowns.

    The slider animates pressure only for the default chemistry (metal 0.005, ratio 3).
    The dropdowns show static surfaces at pressure 0.5.

    Args:
        env: `ComplexPharmaFlowsheet` instance.
    """

    # --- 1. SETUP GRID ---
    resolution = 40
    x_temp = np.linspace(0, 1, resolution)
    x_time = np.linspace(0, 1, resolution)
    T_Grid, Time_Grid = np.meshgrid(x_temp, x_time)

    # --- 2. DEFINE SCENARIOS (Dropdown Options) ---
    # We pre-calculate the Reactor Surface for these combinations to avoid lag
    metals = {'Low (0.002M)': 0.002, 'Med (0.005M)': 0.005, 'High (0.010M)': 0.010}
    ratios = {'Lean (2.0)': 2.0, 'Ideal (3.0)': 3.0, 'Rich (5.0)': 5.0}

    # Pressure steps for the Slider
    pressures = np.linspace(0.1, 1.0, 10)

    # Storage for pre-computed reactor surfaces
    # Structure: reactor_data[metal_key][ratio_key] = (cp, cd, cs)
    reactor_data = {}

    print("Pre-computing Chemistry Scenarios...")

    for m_key, m_val in metals.items():
        reactor_data[m_key] = {}
        for r_key, r_val in ratios.items():
            # Step 1: Activation
            curr_ligand = m_val * r_val
            ml3, ml2, free_l = env.step_1_activation(m_val, curr_ligand)

            # Step 2: Reactor (The expensive part)
            cp, cd, cs = env.step_2_reactor(ml3, ml2, free_l, T_Grid, Time_Grid)

            reactor_data[m_key][r_key] = (cp, cd, cs)

    # --- 3. BUILD PLOT ---
    fig = go.Figure()

    # We need a 'default' state for the initial plot
    default_m = 'Med (0.005M)'
    default_r = 'Ideal (3.0)'

    # Calculate initial surface at Pressure=0.1
    cp, cd, cs = reactor_data[default_m][default_r]
    p_perm, d_perm, s_perm = env.step_3_nanofiltration(cp, cd, cs, pressures[0])
    z_init = env.step_4_crystallization(p_perm, d_perm, s_perm)

    # Add the single Surface Trace
    fig.add_trace(go.Surface(
        z=z_init, x=x_temp, y=x_time,
        colorscale='Viridis',
        cmin=0, cmax=1.0,
        colorbar=dict(title='Yield')
    ))

    # --- 4. CREATE FRAMES (For Pressure Slider) ---
    # Note: Frames in Plotly usually update data based on the *current* traces.
    # Limitation: We can only generate frames for the DEFAULT chemistry.
    # Changing the Dropdown will technically require re-generating frames,
    # which static Plotly cannot do.
    # WORKAROUND: The Slider will animate Pressure for the *current view*,
    # but switching dropdowns might reset the animation context.

    frames = []
    for p in pressures:
        # We use the default chemistry for the frames
        # (Users will see proper animation for Med/Ideal case)
        cp, cd, cs = reactor_data[default_m][default_r]
        p_perm, d_perm, s_perm = env.step_3_nanofiltration(cp, cd, cs, p)
        z_final = env.step_4_crystallization(p_perm, d_perm, s_perm)

        frames.append(go.Frame(data=[go.Surface(z=z_final)], name=f"{p:.1f}"))

    fig.frames = frames

    # --- 5. DEFINE DROPDOWNS (UpdateMenus) ---
    # These buttons will update the Z-data of the surface trace directly

    # List to hold button configs
    metal_buttons = []

    for m_key in metals.keys():
        # When we switch Metal, we want to keep the current Ratio if possible.
        # But Plotly args are static. We must define the Target Z for a specific combination.
        # Simplified: Clicking Metal resets Ratio to 'Ideal'.

        cp, cd, cs = reactor_data[m_key]['Ideal (3.0)']
        # Assume Pressure = 0.5 (Midpoint) for the static update
        p_perm, d_perm, s_perm = env.step_3_nanofiltration(cp, cd, cs, 0.5)
        z_new = env.step_4_crystallization(p_perm, d_perm, s_perm)

        metal_buttons.append({
            'label': m_key,
            'method': 'restyle',
            'args': [{'z': [z_new]}]  # Update the Z data
        })

    ratio_buttons = []
    for r_key in ratios.keys():
        # Simplified: Clicking Ratio resets Metal to 'Med'.
        cp, cd, cs = reactor_data['Med (0.005M)'][r_key]
        p_perm, d_perm, s_perm = env.step_3_nanofiltration(cp, cd, cs, 0.5)
        z_new = env.step_4_crystallization(p_perm, d_perm, s_perm)

        ratio_buttons.append({
            'label': r_key,
            'method': 'restyle',
            'args': [{'z': [z_new]}]
        })

    # --- 6. LAYOUT & CONTROLS ---
    fig.update_layout(
        title='Process Yield Explorer: Final Output',
        scene=dict(
            xaxis_title='Temperature',
            yaxis_title='Residence Time',
            zaxis_title='Final Yield',
            zaxis=dict(range=[0, 1.0]),
            camera=dict(eye=dict(x=-1.5, y=-1.5, z=0.5))
        ),
        # Add Dropdowns
        updatemenus=[
            dict(
                buttons=metal_buttons,
                direction="down",
                x=0.1, y=0.9,
                showactive=True,
                name="Metal"
            ),
            dict(
                buttons=ratio_buttons,
                direction="down",
                x=0.3, y=0.9,
                showactive=True,
                name="Ratio"
            )
        ],
        # Add Slider
        sliders=[{
            "pad": {"b": 10, "t": 50},
            "len": 0.9, "x": 0.1,
            "currentvalue": {"prefix": "Pressure (Animation): "},
            "steps": [
                {
                    "args": [[f"{p:.1f}"], {"frame": {"duration": 50, "redraw": True}, "mode": "immediate"}],
                    "label": f"{p:.1f}",
                    "method": "animate"
                } for p in pressures
            ]
        }],
        # Annotations for Dropdowns
        annotations=[
            dict(text="Metal Conc:", x=0.1, y=0.96, xref="paper", yref="paper", showarrow=False),
            dict(text="Ligand Ratio:", x=0.3, y=0.96, xref="paper", yref="paper", showarrow=False)
        ]
    )

    fig.show()


def plot_mountain_range_demo(env):
    """Crystallisation yield over the impurity fraction (chirp and oiling-out cliff).

    Args:
        env: `ComplexPharmaFlowsheet` instance.
    """
    # Sweep Impurity from 0% to 45%
    impurity = np.linspace(0, 0.45, 1000)

    # Assume a constant input mass of 1.0 for visualization
    p_perm = np.ones_like(impurity)

    # Back-calculate d_perm to achieve the specific impurity fraction
    # formula: I = d / (p + d)  ->  d = p * I / (1 - I)
    d_perm = p_perm * impurity / (1 - impurity)
    s_perm = np.zeros_like(d_perm)

    # Run Step 4
    y_final = env.step_4_crystallization(p_perm, d_perm, s_perm)

    fig = go.Figure()

    # The Yield Curve
    fig.add_trace(go.Scatter(
        x=impurity, y=y_final,
        mode='lines', name='Yield',
        line=dict(color='purple', width=3)
    ))

    # Annotation: Stable Zone
    fig.add_annotation(
        x=0.05, y=0.9, text="<b>Stable Zone</b><br>(Stationary)",
        showarrow=False, font=dict(color="green")
    )

    # Annotation: Chirp Zone
    fig.add_annotation(
        x=0.25, y=0.5, text="<b>Polymorph Chirp</b><br>(High Freq Instationarity)",
        showarrow=True, arrowhead=1, ax=0, ay=-40
    )

    # Annotation: Cliff
    fig.add_vline(x=0.35, line_dash="dash", line_color="red")
    fig.add_annotation(
        x=0.36, y=0.8, text="<b>Oiling Out</b><br>(Crash)",
        showarrow=False, xanchor="left", font=dict(color="red")
    )

    fig.update_layout(
        title='Step 4 Physics: The Polymorph Trap',
        xaxis_title='Impurity Fraction',
        yaxis_title='Crystallization Yield',
        template='plotly_white',
        xaxis=dict(range=[0, 0.45])
    )
    fig.show()

def plot_bimodal_kinetics(env):
    """Reactor outlets along residence time at metal 0.005, ratio 3 and temperature 0.65.

    Args:
        env: `ComplexPharmaFlowsheet` instance.
    """
    import plotly.graph_objects as go
    import numpy as np

    # Setup Inputs
    times_norm = np.linspace(0, 1.0, 100)
    fixed_metal = 0.005
    fixed_ratio = 3.0
    fixed_temp = 0.65

    # Calculate
    metal_in = np.full_like(times_norm, fixed_metal)
    ligand_in = np.full_like(times_norm, fixed_metal * fixed_ratio)
    temp_in = np.full_like(times_norm, fixed_temp)

    ml3, ml2, free_l = env.step_1_activation(metal_in, ligand_in)
    cp, cd, cs = env.step_2_reactor(ml3, ml2, free_l, temp_in, times_norm)

    fig = go.Figure()

    # Product (Green) - Should be W-shaped or N-shaped
    fig.add_trace(go.Scatter(x=times_norm, y=cp, mode='lines', name='Total Product', line=dict(color='green', width=4)))

    # Impurity (Orange) - Should peak in the valley
    fig.add_trace(go.Scatter(x=times_norm, y=cs, mode='lines', name='Impurity (Intermediate)',
                             line=dict(color='orange', width=3, dash='dash')))

    # Degradation (Red) - Should rise steadily
    fig.add_trace(
        go.Scatter(x=times_norm, y=cd, mode='lines', name='Degradation', line=dict(color='red', width=3, dash='dot')))

    # Annotations
    fig.add_annotation(x=0.1, y=np.max(cp) * 0.8, text="Kinetic Peak (Unstable)", showarrow=False,
                       font=dict(color="green"))
    fig.add_annotation(x=0.8, y=np.max(cp) * 0.9, text="Thermodynamic Peak (Stable)", showarrow=False,
                       font=dict(color="green"))

    fig.update_layout(title='Kinetic vs Thermodynamic Control', xaxis_title='Residence Time', yaxis_title='Conc')
    fig.show()


def plot_global_landscape_steady_state(env):
    """Final yield over temperature x residence time at fixed metal, ligand and pressure 0.7.

    Args:
        env: `ComplexPharmaFlowsheet` instance.
    """
    print("Generating Global Landscape...")

    res = 60
    x_temp, x_time, T_grid, Time_grid = generate_grid(max_val=1.0, resolution=res)

    fixed_metal = 0.005
    fixed_ligand = 0.015
    fixed_pressure = 0.7

    # Step 1
    ml3, ml2, free_l = env.step_1_activation(
        np.full_like(T_grid, fixed_metal),
        np.full_like(T_grid, fixed_ligand)
    )

    # Step 2
    cp, cd, cs = env.step_2_reactor(ml3, ml2, free_l, T_grid, Time_grid)

    # Step 3 (Pure Steady State)
    p_perm, d_perm, s_perm = env.step_3_nanofiltration(
        cp, cd, cs,
        np.full_like(T_grid, fixed_pressure)
    )

    # Step 4
    final_yield = env.step_4_crystallization(p_perm, d_perm, s_perm)

    fig = go.Figure()

    fig.add_trace(go.Surface(
        z=final_yield, x=x_temp, y=x_time,
        colorscale='Turbo',
        contours_z=dict(show=True, usecolormap=True, highlightcolor="limegreen", project_z=True)
    ))

    fig.update_layout(
        title='Global Landscape (Steady State)',
        scene=dict(
            xaxis_title='Temperature',
            yaxis_title='Residence Time',
            zaxis_title='Final Yield',
            camera=dict(eye=dict(x=-1.5, y=-1.5, z=0.8))
        ),
        width=1000, height=800,
    )
    fig.show()

if __name__ == "__main__":
    env = ComplexPharmaFlowsheet()

    plot_global_landscape_steady_state(env)

    print("Plotting Step 1 Speciation...")
    # plot_step_1_interactive_speciation(env)
    #
    print("Plotting Step 2 Regime Explorer...")
    plot_step_2_ratio_explorer(env)
    #
    # plot_bimodal_kinetics(env)

    print("Plotting Step 2 Reactor Output Lines...")
    # plot_reactor_dual_sliders(env)

    print("Plotting Full Process Dashbaord...")
    plot_interactive_yield_surface(env)

    print("Plotting Full Process Dashbaord...")
    plot_mountain_range_demo(env)

# 1. 2D Hydrodynamics (Baffles)
#     plot_hydrodynamic_kinetics(env)

    # 2. 3D Nanofiltration (Regime Switching)
    # plot_nanofiltration_cliff(env)

    # 3. Full Global Surface
    # plot_full_process_landscape(env)