import torch
import torch.nn as nn
import numpy as np
import json
from spikingjelly.activation_based import neuron, functional, surrogate
from typing import Optional, Callable


class GLIF3Neuron(neuron.BaseNode):
    """
    Generalized Leaky Integrate-and-Fire Model 3 (GLIF3) Neuron Implementation
    
    This implementation provides a GLIF3 neuron model with after-spike currents (ASC) 
    for adaptation mechanisms and support for multiple synaptic time constants.
    
    Mathematical Model:
    The membrane potential V(t) follows:
    C_m * dV/dt = -g * (V - E_L) - I_ASC(t) - I_syn(t) + I_ext(t)
    
    Where:
    - I_ASC(t): After-spike current for adaptation
    - I_syn(t): Synaptic current from alpha-function shaped inputs
    - I_ext(t): External injected current
    
    References:
    Teeter, C., Iyer, R., Menon, V., et al. (2018). Generalized leaky integrate-and-fire 
    models classify multiple neuron types. Nature Communications, 9(1), 709.
    """
    
    def __init__(self, 
                 V_m: float = -70.0,           # Initial membrane potential (mV)
                 V_th: float = -50.0,          # Spike threshold potential (mV) 
                 g: float = 5.0,               # Membrane leak conductance (nS)
                 E_L: float = -70.0,           # Leak reversal potential (mV)
                 C_m: float = 100.0,           # Membrane capacitance (pF)
                 t_ref: float = 2.0,           # Absolute refractory period (ms)
                 V_reset: float = -70.0,       # Post-spike reset potential (mV)
                 asc_init: list = [0.0, 0.0], # Initial after-spike current values (pA)
                 asc_decay: list = [0.003, 0.1],  # ASC decay time constants (1/ms)
                 asc_amps: list = [-10.0, -100.0], # ASC amplitude increments per spike (pA)
                 tau_syn: list = [5.5, 8.5, 2.8, 5.8], # Synaptic time constants (ms)
                 dt: float = 0.1,              # Integration time step (ms)
                 
                 # SpikingJelly BaseNode parameters
                 v_reset: Optional[float] = None,  # SpikingJelly reset voltage (will be computed from V_reset)
                 surrogate_function: Callable = None,
                 detach_reset: bool = False,
                 step_mode: str = 's',
                 backend: str = 'torch',
                 store_v_seq: bool = False,
                 
                 # Training options
                 refractory_smoothness: float = 100.0,  # Controls smoothness of refractory transition (higher = sharper)
                 **kwargs):
        """
        Initialize the GLIF3 neuron with specified biophysical parameters.
        
        Parameters:
        -----------
        V_m : float
            Initial membrane potential in millivolts. Sets the starting voltage state.
        V_th : float  
            Spike threshold potential in millivolts. When membrane potential exceeds
            this value, a spike is generated.
        g : float
            Membrane leak conductance in nanosiemens. Determines passive membrane properties.
        E_L : float
            Leak reversal potential in millivolts. The resting potential towards which
            the membrane relaxes in absence of input.
        C_m : float
            Membrane capacitance in picofarads. Affects integration time constant.
        t_ref : float
            Absolute refractory period in milliseconds. Duration after spike during
            which no new spikes can be generated.
        V_reset : float
            Post-spike reset potential in millivolts. Membrane potential immediately
            after spike generation.
        asc_init : list
            Initial values for after-spike currents in picoamperes. Typically zero
            at simulation start.
        asc_decay : list  
            After-spike current decay rates in 1/milliseconds. Higher values indicate
            faster decay. Must match length of asc_amps.
        asc_amps : list
            After-spike current amplitude increments in picoamperes. Added to ASC
            state variables upon each spike. Negative values provide adaptation.
        tau_syn : list
            Synaptic time constants in milliseconds for alpha-function shaped
            postsynaptic currents. Each element creates a distinct synaptic receptor type.
        dt : float
            Integration time step in milliseconds. Should be small enough for
            numerical stability (typically 0.1 ms or smaller).
        
        # SpikingJelly parameters
        v_reset : float, optional
            Reset voltage. If None, will be computed from V_reset.
        surrogate_function : Callable, optional
            Surrogate function for gradient computation. If None, uses Sigmoid().
        detach_reset : bool
            Whether to detach reset computation graph.
        step_mode : str
            Step mode ('s' for single-step, 'm' for multi-step).
        backend : str
            Backend to use ('torch', 'cupy').
        store_v_seq : bool
            Whether to store voltage sequence in multi-step mode.
            
        # Training options
        refractory_smoothness : float
            Controls the smoothness of refractory period transitions for training.
            Higher values create sharper transitions closer to hard threshold behavior.
            Default is 100.0 for good balance between differentiability and accuracy.
            
        **kwargs
            Additional arguments passed to parent neuron.BaseNode class.
        """
        # Calculate normalized parameters
        v_threshold_sj = 1.0  # Normalized threshold (always 1.0)
        
        # Compute v_reset (normalized between 0 and 1)
        if v_reset is None:
            # Normalize reset voltage relative to voltage range
            voltage_range = V_th - E_L
            v_reset_sj = (V_reset - E_L) / voltage_range
        else:
            v_reset_sj = v_reset
        
        # Initialize parent class with normalized parameters
        super().__init__(
            v_threshold=v_threshold_sj,
            v_reset=v_reset_sj,
            surrogate_function=surrogate_function,
            detach_reset=detach_reset,
            step_mode=step_mode,
            backend=backend,
            store_v_seq=store_v_seq,
            **kwargs
        )
        
        # Validate input parameters for biological plausibility and numerical stability
        self._validate_params(V_m, V_th, g, E_L, C_m, t_ref, dt, asc_decay, asc_amps, tau_syn)
        
        # Store fundamental biophysical parameters as instance attributes
        self.V_m, self.V_th, self.g = V_m, V_th, g
        self.E_L, self.C_m, self.t_ref = E_L, C_m, t_ref
        self.V_reset, self.dt = V_reset, dt
        
        # Store differentiability options
        self.refractory_smoothness = refractory_smoothness
        
        # Compute derived membrane properties
        self.tau_m = C_m / g  # Membrane time constant (ms)
        self.num_asc = len(asc_decay)  # Number of after-spike current components
        self.num_syn = len(tau_syn)    # Number of synaptic receptor types
        
        # Pre-compute all integration parameters using NEST-compatible methods
        self._compute_params(asc_decay, asc_amps, tau_syn, asc_init)
        
        # Initialize all state variables to their resting values
        self._init_states()
    
    def _validate_params(self, V_m, V_th, g, E_L, C_m, t_ref, dt, asc_decay, asc_amps, tau_syn):
        """
        Validate all input parameters for biological plausibility and numerical stability.
        
        Parameters:
        -----------
        All parameters as defined in __init__ method.
        
        Raises:
        -------
        AssertionError
            If any parameter violates biological or numerical constraints.
        """
        assert V_th > V_m, f"Threshold potential ({V_th}mV) must exceed resting potential ({V_m}mV)"
        assert g > 0, f"Membrane conductance ({g}nS) must be positive"
        assert C_m > 0, f"Membrane capacitance ({C_m}pF) must be positive"  
        assert t_ref >= 0, f"Refractory period ({t_ref}ms) cannot be negative"
        assert 0 < dt <= 1.0, f"Integration time step ({dt}ms) must be in range (0, 1]"
        assert len(asc_decay) == len(asc_amps), "ASC decay rates and amplitudes must have equal length"
        assert all(k > 0 for k in asc_decay), "All ASC decay rates must be positive"
        assert all(tau > 0 for tau in tau_syn), "All synaptic time constants must be positive"
    
    def _compute_params(self, asc_decay, asc_amps, tau_syn, asc_init):
        """
        Pre-compute all parameters required for numerically stable integration.
        
        Parameters:
        -----------
        asc_decay : list
            After-spike current decay rates (1/ms)
        asc_amps : list  
            After-spike current amplitudes (pA)
        tau_syn : list
            Synaptic time constants (ms)
        asc_init : list
            Initial ASC values (pA)
        """
        h = self.dt  # Integration time step
        
        # === After-Spike Current (ASC) Parameters ===
        # Convert to numpy arrays for vectorized computation
        asc_decay_array = np.array(asc_decay, dtype=np.float32)
        asc_amps_array = np.array(asc_amps, dtype=np.float32)
        
        # Compute exact exponential decay factors for discrete time steps
        asc_decay_dt = np.exp(-asc_decay_array * h)
        
        # Compute stable integration coefficients for ASC contribution to membrane voltage
        # These coefficients account for the exact integral of ASC over each time step
        asc_stable_coeff = ((1.0 / asc_decay_array) / h) * (1.0 - asc_decay_dt)
        
        # Compute refractory decay rates for ASC
        # These determine how much ASC remains after the refractory period
        asc_refractory_decay_rates = np.exp(-asc_decay_array * self.t_ref)
        
        # Store as instance attributes
        self.asc_decay_dt = torch.from_numpy(asc_decay_dt)
        self.asc_stable_coeff = torch.from_numpy(asc_stable_coeff)
        self.asc_amps = torch.from_numpy(asc_amps_array)
        self.asc_refractory_decay_rates = torch.from_numpy(asc_refractory_decay_rates)
        
        # === Membrane Voltage Integration Parameters ===
        # Exact solution for passive membrane equation: V' = -(V-E_L)/tau_m + I/C_m
        P33 = np.exp(-h / self.tau_m)  # Voltage decay factor
        P30 = (1.0 / self.C_m) * (1.0 - P33) * self.tau_m  # Current integration factor
        
        self.P33 = torch.tensor(P33, dtype=torch.float32)
        self.P30 = torch.tensor(P30, dtype=torch.float32)
        
        # === Synaptic Current Parameters ===
        # Alpha-function synaptic currents: I_syn(t) = (t/tau) * exp(-t/tau)
        tau_syn_array = np.array(tau_syn, dtype=np.float32)
        
        # Discrete-time evolution parameters for alpha-function state variables
        P11 = np.exp(-h / tau_syn_array)  # Decay factor (P22 is identical)
        P21 = h * P11  # Coupling between y1 and y2 state variables
        
        # Initial amplitude scaling for unit synaptic input
        PSC_initial = np.e / tau_syn_array  # Ensures peak current of 1 pA for weight 1
        
        self.P11 = torch.from_numpy(P11)
        self.P21 = torch.from_numpy(P21)
        self.PSC_initial = torch.from_numpy(PSC_initial)
        
        # === Synaptic-Membrane Coupling Parameters ===
        # Compute exact integration of synaptic currents' effect on membrane voltage
        P31, P32 = self._compute_syn_coupling(tau_syn_array, h)
        self.P31 = torch.from_numpy(P31)
        self.P32 = torch.from_numpy(P32)
        
        # === Additional Integration Parameters ===
        self.v_range = torch.tensor(self.V_th - self.E_L, dtype=torch.float32)
        self.dt_tensor = torch.tensor(h, dtype=torch.float32)
        self.t_ref_tensor = torch.tensor(self.t_ref, dtype=torch.float32)
        self.asc_init_values = torch.tensor(asc_init, dtype=torch.float32)
    
    def _compute_syn_coupling(self, tau_syn, h):
        """
        Calculate exact integration coefficients for synaptic current effects on membrane voltage.
        
        Parameters:
        -----------
        tau_syn : np.array
            Synaptic time constants in milliseconds
        h : float
            Integration time step in milliseconds
            
        Returns:
        --------
        tuple
            (P31, P32) arrays containing coupling coefficients for each synapse type
        """
        P31 = np.zeros(len(tau_syn), dtype=np.float32)
        P32 = np.zeros(len(tau_syn), dtype=np.float32)
        
        for i, tau_s in enumerate(tau_syn):
            # IAFPropagatorAlpha logic
            beta = tau_s * self.tau_m / (self.tau_m - tau_s)
            gamma = beta / self.C_m
            
            # Regular case: tau_membrane ≠ tau_synaptic
            exp_h_tau_syn = np.exp(-h / tau_s)
            expm1_h_tau = np.expm1(h * (1/tau_s - 1/self.tau_m))
            
            P32[i] = gamma * exp_h_tau_syn * expm1_h_tau
            P31[i] = gamma * exp_h_tau_syn * (beta * expm1_h_tau - h)
        
        return P31, P32
    
    def _init_states(self):
        """
        Initialize all dynamic state variables to their appropriate starting values.
        """
        # Initialize membrane potential relative to leak reversal potential
        # Store U_ as relative to E_L_, so V_m - E_L
        initial_voltage = self.V_m - self.E_L
        self.U = torch.tensor(initial_voltage, dtype=torch.float32)
        
        # Initialize normalized voltage
        # Convert initial absolute voltage to normalized voltage (0 to 1, where 1 is threshold)
        voltage_range = self.V_th - self.E_L
        initial_v_normalized = initial_voltage / voltage_range
        
        # Set v directly (BaseNode already registered it as memory)
        self.v = torch.tensor(initial_v_normalized, dtype=torch.float32)
        
        # Initialize after-spike current state vector
        self.asc = self.asc_init_values.clone()
        
        # Initialize refractory period counter
        self.ref_count = torch.tensor(0.0, dtype=torch.float32)
        
        # Initialize synaptic state variables for alpha-function dynamics
        # y1 and y2 represent the two state variables of the alpha function
        self.y1 = torch.zeros(self.num_syn, dtype=torch.float32)
        self.y2 = torch.zeros(self.num_syn, dtype=torch.float32)
    
    def neuronal_charge(self, x: torch.Tensor, syn_inputs: Optional[torch.Tensor] = None):
        """
        Update all neuronal state variables for one integration time step.
        
        Parameters:
        -----------
        x : torch.Tensor
            External current injection in picoamperes. Can be scalar or tensor.
        syn_inputs : torch.Tensor, optional
            Synaptic input currents for each receptor type in picoamperes.
            Shape should be (num_syn,) or broadcastable to this shape.
            If None, no synaptic input is applied.
        """
        # === Refractory Period Handling ===
        not_refractory = torch.relu(1.0 - self.refractory_smoothness * self.ref_count)
        not_refractory = torch.clamp(not_refractory, 0.0, 1.0)
        
        # === Membrane Voltage Update (vectorized) ===
        asc_sum = torch.sum(self.asc_stable_coeff * self.asc)
        
        # Update membrane voltage using exact solution of passive membrane equation
        total_current = x + asc_sum
        voltage_update = self.U * self.P33 + total_current * self.P30
        
        # Add synaptic contributions using pre-computed coupling coefficients
        synaptic_contribution = torch.sum(self.P31 * self.y1 + self.P32 * self.y2)
        voltage_update += synaptic_contribution
        
        # Apply voltage update with refractory modulation
        self.U = not_refractory * voltage_update + (1.0 - not_refractory) * self.U
        
        # ASC update: update ASC values after calculating contribution
        asc_update = self.asc * self.asc_decay_dt
        self.asc = not_refractory * asc_update + (1.0 - not_refractory) * self.asc
        
        # === Synaptic State Update (alpha-function evolution) ===
        # Update both state variables of alpha function for all synapses simultaneously
        new_y2 = self.P21 * self.y1 + self.P11 * self.y2
        new_y1 = self.P11 * self.y1
        
        # Apply new synaptic inputs (vectorized)
        if syn_inputs is not None:
            # Ensure input dimensionality matches number of synaptic receptors
            syn_inputs_padded = torch.zeros(self.num_syn, dtype=torch.float32)
            input_length = min(len(syn_inputs), self.num_syn)
            syn_inputs_padded[:input_length] = syn_inputs[:input_length]
            
            # Add weighted synaptic inputs to appropriate state variables
            new_y1 += self.PSC_initial * syn_inputs_padded
        
        # Update synaptic state variables
        self.y1 = new_y1
        self.y2 = new_y2
        
        # === Update normalized voltage ===
        # Convert absolute voltage to normalized voltage (0 to 1, where 1 is threshold)
        voltage_range = self.V_th - self.E_L
        self.v = self.U / voltage_range
        
        # === Refractory Period Update ===
        # Decrement refractory counter, ensuring it doesn't go below zero
        self.ref_count = torch.clamp(self.ref_count - self.dt_tensor, min=0.0)

    def neuronal_reset(self, spike: torch.Tensor):
        """
        Reset neuronal state variables following spike generation.
        
        Parameters:
        -----------
        spike : torch.Tensor
            Continuous spike tensor from surrogate function, values between 0 and 1.
            Values close to 1 indicate strong spiking, values close to 0 indicate no spike.
        """
        # Handle detach_reset for gradient computation
        if self.detach_reset:
            spike_d = spike.detach()
        else:
            spike_d = spike
        
        # === Voltage Reset ===
        if self.v_reset is None:
            # Soft reset: subtract threshold
            self.v = self.jit_soft_reset(self.v, spike_d, self.v_threshold)
        else:
            # Hard reset: set to reset value
            self.v = self.jit_hard_reset(self.v, spike_d, self.v_reset)
        
        # === Update absolute voltage ===
        # Convert normalized v back to absolute voltage
        voltage_range = self.V_th - self.E_L
        self.U = self.v * voltage_range
        
        # === After-Spike Current Update ===
        spike_continuous = spike  # Use continuous values directly
        asc_increment = self.asc_amps * spike_continuous
        asc_decay_factor = self.asc_refractory_decay_rates * spike_continuous
        
        # Update ASC with smooth transition
        self.asc = self.asc + asc_increment + (asc_decay_factor - 1.0) * self.asc * spike_continuous
        
        # === Refractory Period Initiation ===
        spike_continuous = spike
        refractory_increment = self.t_ref_tensor * spike_continuous
        self.ref_count = self.ref_count + refractory_increment
    
    def forward(self, x: torch.Tensor, syn_inputs: Optional[torch.Tensor] = None):
        """
        Execute one complete forward pass of the GLIF3 neuron model.
        
        Parameters:
        -----------
        x : torch.Tensor
            External current input in picoamperes. Shape can be scalar or batch.
        syn_inputs : torch.Tensor, optional
            Synaptic input currents for each receptor type. If provided, shape
            should be (num_syn,) or broadcastable.
            
        Returns:
        --------
        torch.Tensor
            Spike output as floating-point tensor (0.0 = no spike, 1.0 = spike).
        """
        # Ensure v is a tensor
        self.v_float_to_tensor(x)
        
        # Update all state variables based on current inputs
        self.neuronal_charge(x, syn_inputs)
        
        # Detect spike generation based on updated states
        spike = self.neuronal_fire()
        
        # Apply post-spike reset mechanisms if spike occurred
        self.neuronal_reset(spike)
        
        # Return spike as float
        return spike.float()
    
    def get_state(self):
        """
        Retrieve current state of all neuronal variables for monitoring and analysis.
        
        Returns:
        --------
        dict
            Dictionary containing current values of all state variables:
            - 'v_mV': Absolute membrane potential in millivolts
            - 'U_rel_mV': Membrane potential relative to leak reversal potential
            - 'asc_total': Sum of all after-spike current components
            - 'syn_total': Sum of all synaptic current contributions
            - 'ref_count': Remaining refractory period duration
            - 'detailed': Nested dictionary with individual component values
        """
        return {
            'v_mV': (self.U + self.E_L).item(),
            'U_rel_mV': self.U.item(),
            'asc_total': self.asc.sum().item(),
            'syn_total': self.y2.sum().item(),
            'ref_count': self.ref_count.item(),
            'detailed': {
                'asc_components': self.asc.detach().cpu().numpy(),
                'synaptic_y1': self.y1.detach().cpu().numpy(),
                'synaptic_y2': self.y2.detach().cpu().numpy(),
            }
        }

# ============================================================================
# Utility Functions and Analysis Tools
# ============================================================================

def load_params(json_file):
    """
    Load neuron parameters from Allen Institute JSON parameter file.
    
    Parameters:
    -----------
    json_file : str
        Path to JSON file containing neuron parameters in Allen Institute format.
        
    Returns:
    --------
    dict
        Dictionary containing all neuron parameters with standard names.
    """
    with open(json_file, 'r') as f:
        return json.load(f)


def create_glif3_from_params(params_file, dt=0.1):
    """
    Factory function to create GLIF3 neuron from Allen Institute parameter file.
    
    Parameters:
    -----------
    params_file : str
        Path to JSON parameter file from Allen Institute Cell Types Database.
    dt : float, optional
        Integration time step in milliseconds. Default is 0.1 ms for numerical stability.
        
    Returns:
    --------
    GLIF3Neuron
        Fully configured neuron instance ready for simulation.
    """
    params = load_params(params_file)
    return GLIF3Neuron(
        V_m=params['V_m'],
        V_th=params['V_th'], 
        g=params['g'],
        E_L=params['E_L'],
        C_m=params['C_m'],
        t_ref=params['t_ref'],
        V_reset=params['V_reset'],
        asc_init=params['asc_init'],
        asc_decay=params['asc_decay'],
        asc_amps=params['asc_amps'],
        tau_syn=params['tau_syn'],
        dt=dt
    )


def test_fi_curve(neuron, current_range, duration=1000, dt=0.1, stim_start=200, stim_end=800):
    """
    Measure neuron frequency-current (F-I) relationship with timed current injection.
    
    Parameters:
    -----------
    neuron : GLIF3Neuron
        Neuron instance to be tested. Will be reset before each current level.
    current_range : iterable
        Sequence of current injection levels in picoamperes to test.
    duration : float, optional
        Total simulation duration in milliseconds. Default is 1000 ms.
    dt : float, optional
        Integration time step in milliseconds. Should match neuron's dt parameter.
        Default is 0.1 ms.
    stim_start : float, optional
        Time to start current injection in milliseconds. Default is 200 ms.
    stim_end : float, optional
        Time to end current injection in milliseconds. Default is 800 ms.
        
    Returns:
    --------
    list
        List of dictionaries, one per current level, containing:
        - 'current': Injected current level (pA)
        - 'frequency': Firing frequency during stimulation period (Hz)  
        - 'spike_count_stim': Number of spikes during stimulation period
        - 'spike_count_total': Total number of spikes during entire simulation
        - 'spike_times': List of all spike occurrence times (ms) for first 10 spikes
        - 'stim_duration': Duration of current injection (ms)
    """
    results = []
    n_steps = int(duration / dt)
    stim_duration = stim_end - stim_start  # Duration of current injection
    
    print(f"Testing F-I curve with {len(current_range)} current levels...")
    print(f"Total duration: {duration} ms, Current injection: {stim_start}-{stim_end} ms ({stim_duration} ms)")
    print(f"Time step: {dt} ms")
    print(f"{'Current (pA)':>12} {'Frequency (Hz)':>14} {'Stim Spikes':>12} {'Total Spikes':>13}")
    print("-" * 55)
    
    for i_inj in current_range:
        # Reset neuron to initial state for independent measurement
        functional.reset_net(neuron)
        
        # Track spike times for detailed analysis
        spike_times = []
        spike_times_stim = []  # Spikes during stimulation period only
        
        # Simulate neuron response over specified duration
        for t_step in range(n_steps):
            current_time = t_step * dt
            
            # Determine current input based on time window
            if stim_start <= current_time < stim_end:
                current_input = torch.tensor(i_inj, dtype=torch.float32)
            else:
                current_input = torch.tensor(0.0, dtype=torch.float32)
            
            # Execute one simulation step
            spike_output = neuron(current_input)
            
            # Record spike occurrence with precise timing
            if spike_output.item() > 0.5:  # Threshold for spike detection
                spike_times.append(current_time)
                # Also record if spike occurred during stimulation period
                if stim_start <= current_time < stim_end:
                    spike_times_stim.append(current_time)
        
        # Calculate firing statistics
        spike_count_total = len(spike_times)
        spike_count_stim = len(spike_times_stim)
        
        # Calculate frequency based on stimulation period only
        frequency = spike_count_stim / (stim_duration / 1000.0)  # Convert to Hz
        
        # Store comprehensive results
        result_data = {
            'current': i_inj,
            'frequency': frequency,
            'spike_count_stim': spike_count_stim,
            'spike_count_total': spike_count_total,
            'spike_times': spike_times[:10],  # First 10 spikes for analysis
            'stim_duration': stim_duration
        }
        results.append(result_data)
        
        # Display progress
        print(f"{i_inj:>12.1f} {frequency:>14.2f} {spike_count_stim:>12d} {spike_count_total:>13d}")
    
    print("-" * 55)
    print(f"F-I curve measurement completed.")
    print(f"Frequency calculated based on {stim_duration} ms stimulation period.")
    
    return results


def main():
    """
    Demonstrate the GLIF3 neuron implementation.
    """
    # Define path to Allen Institute parameter file
    params_file = ('/home/user/Documents/Training-data-driven-V1-model-test/'
                   'Allen_V1_param/components/cell_models/nest_models/475622680_glif_psc.json')
    
    try:
        # Create neuron instance from experimental parameters
        print("Loading neuron parameters from Allen Institute database...")
        neuron = create_glif3_from_params(params_file, dt=0.1)
        print(f"Created neuron: {neuron}")
        print()
        
        # Perform comprehensive F-I curve analysis
        print("Performing F-I curve analysis...")
        current_range = range(0, 701, 50)  # 0 to 700 pA in 50 pA increments
        fi_results = test_fi_curve(neuron, current_range, duration=1000, dt=0.1)
        
    except FileNotFoundError:
        print(f"Error: Parameter file not found at {params_file}")
        print("Please ensure the Allen Institute parameter files are available.")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 