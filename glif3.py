import torch
import numpy as np
import json
from spikingjelly.activation_based import neuron, functional, surrogate
from typing import Optional, Callable


class MathUtils:
    """Mathematical utilities for neural dynamics computations."""
    
    @staticmethod
    def integration_coefficient(rates, dt: float, scaling=1.0):
        """Integration coefficient: scaling * (1 - exp(-rates*dt)) / rates"""
        rates_arr = np.asarray(rates)
        decay_factors = np.exp(-rates_arr * dt)
        return np.asarray(scaling) * (1.0 - decay_factors) / rates_arr
    
    @staticmethod
    def alpha_coefficients(tau: float, dt: float) -> tuple:
        """Alpha-function coefficients: (exp(-dt/tau), dt*exp(-dt/tau))"""
        decay_factor = np.exp(-dt / tau)
        return decay_factor, dt * decay_factor
    
    @staticmethod
    def exponential_coupling(tau_source: float, tau_target: float, 
                           dt: float, scaling: float = 1.0) -> tuple:
        """Coupling coefficients between two exponential systems"""
        if abs(tau_source - tau_target) < 1e-10:
            # Handle special case where time constants are nearly equal
            exp_factor = np.exp(-dt / tau_source)
            coeff1 = scaling * dt * exp_factor
            coeff2 = scaling * dt * exp_factor
        else:
            # General case: different time constants
            beta = tau_source * tau_target / (tau_target - tau_source)
            gamma = beta * scaling
            
            exp_h_tau_source = np.exp(-dt / tau_source)
            expm1_h_tau = np.expm1(dt * (1/tau_source - 1/tau_target))
            
            coeff2 = gamma * exp_h_tau_source * expm1_h_tau
            coeff1 = gamma * exp_h_tau_source * (beta * expm1_h_tau - dt)
        
        return coeff1, coeff2


class ExponentialIntegrator:
    """Universal exponential dynamics integrator for neural components"""
    
    def __init__(self, rates, dt: float, scaling=None, amplitudes=None, t_ref: float = 0.0):
        """
        Initialize exponential integrator
        
        rates: decay rate(s) - scalar for membrane, array for ASC
        dt: time step
        scaling: scaling factor(s) for integration  
        amplitudes: spike increment amplitudes (for ASC only)
        t_ref: refractory period (for ASC only)
        """
        self.dt = dt
        self.t_ref = t_ref
        
        # Handle both scalar and array inputs
        self.rates = np.asarray(rates, dtype=np.float32)
        self.is_scalar = np.isscalar(rates)
        
        # Compute decay factors - ensure array format for torch.from_numpy
        decay_vals = np.exp(-self.rates * dt)
        self.decay_factors = torch.from_numpy(np.atleast_1d(decay_vals))
        
        # Compute integration coefficients
        if scaling is not None:
            integration_vals = MathUtils.integration_coefficient(self.rates, dt, scaling)
            self.integration_coeffs = torch.from_numpy(np.atleast_1d(integration_vals))
        else:
            self.integration_coeffs = None
            
        # ASC-specific parameters
        if amplitudes is not None:
            self.amplitudes = torch.from_numpy(np.asarray(amplitudes, dtype=np.float32))
            refractory_vals = np.exp(-self.rates * t_ref) if t_ref > 0 else np.ones_like(self.rates)
            self.refractory_decay = torch.from_numpy(np.atleast_1d(refractory_vals))
        else:
            self.amplitudes = None
            self.refractory_decay = None
    
    def integrate_step(self, current_state: torch.Tensor, 
                      input_current: torch.Tensor = None,
                      modulation_factor: torch.Tensor = None) -> torch.Tensor:
        """Universal integration step"""
        if modulation_factor is None:
            modulation_factor = torch.tensor(1.0)
            
        # Get decay factors in appropriate shape
        decay_factors = self.decay_factors[0] if self.is_scalar else self.decay_factors
        
        # Basic exponential decay
        updated_state = current_state * decay_factors
        
        # Add input contribution if provided
        if input_current is not None and self.integration_coeffs is not None:
            integration_coeffs = self.integration_coeffs[0] if self.is_scalar else self.integration_coeffs
            updated_state += input_current * integration_coeffs
            
        # Apply modulation factor
        return modulation_factor * updated_state + (1.0 - modulation_factor) * current_state
    
    def get_membrane_contribution(self, state: torch.Tensor) -> torch.Tensor:
        """Calculate contribution to membrane potential (for ASC)"""
        if self.integration_coeffs is not None and not self.is_scalar:
            return torch.sum(self.integration_coeffs * state)
        return state
    
    def apply_spike_increment(self, current_state: torch.Tensor, 
                             spike_signal: torch.Tensor) -> torch.Tensor:
        """Apply increment during spike events (for ASC)"""
        if self.amplitudes is None:
            return current_state
            
        increment = self.amplitudes * spike_signal
        refractory_decay = self.refractory_decay[0] if self.is_scalar else self.refractory_decay
        decay_factor = refractory_decay * spike_signal
        
        return (current_state + increment + 
                (decay_factor - 1.0) * current_state * spike_signal)


class AlphaSynapseProcessor:
    """Alpha-function synapse processor for multiple receptor types"""
    
    def __init__(self, tau_syn: list, dt: float, membrane_params: dict = None):
        self.num_syn = len(tau_syn)
        self.dt = dt
        self.tau_syn = tau_syn
        
        # Convert to numpy array
        self.tau_syn_array = np.array(tau_syn, dtype=np.float32)
        
        # Compute synaptic dynamics parameters
        self._compute_synapse_params(membrane_params)
        
        # Initialize synaptic states
        self.reset_states()
    
    def _compute_synapse_params(self, membrane_params):
        
        # Compute alpha function coefficients
        alpha_coeffs = [
            MathUtils.alpha_coefficients(tau, self.dt) 
            for tau in self.tau_syn
        ]
        
        # Extract alpha function coefficients
        alpha_decay_list, alpha_coupling_list = zip(*alpha_coeffs)
        self.alpha_decay_factors = torch.from_numpy(np.array(alpha_decay_list, dtype=np.float32))
        self.alpha_coupling_factors = torch.from_numpy(np.array(alpha_coupling_list, dtype=np.float32))
        
        # Compute input scaling factors
        input_scaling_list = [
            np.e / tau
            for tau in self.tau_syn
        ]
        self.input_scaling_factors = torch.from_numpy(np.array(input_scaling_list, dtype=np.float32))
        
        # Compute membrane coupling coefficients
        if membrane_params is not None:
            self._compute_membrane_coupling(membrane_params['tau_m'], membrane_params['C_m'])
        else:
            # Set to zero when no membrane parameters
            self.y1_membrane_coeffs = torch.zeros(self.num_syn)
            self.y2_membrane_coeffs = torch.zeros(self.num_syn)
    
    def _compute_membrane_coupling(self, tau_m, C_m):
        coupling_coeffs = [
            MathUtils.exponential_coupling(
                tau_s, tau_m, self.dt, scaling=1.0/C_m
            )
            for tau_s in self.tau_syn
        ]
        
        # Extract membrane coupling coefficients
        y1_coupling_list, y2_coupling_list = zip(*coupling_coeffs)
        self.y1_membrane_coeffs = torch.from_numpy(np.array(y1_coupling_list, dtype=np.float32))
        self.y2_membrane_coeffs = torch.from_numpy(np.array(y2_coupling_list, dtype=np.float32))
    
    def reset_states(self):
        self.y1 = torch.zeros(self.num_syn, dtype=torch.float32)
        self.y2 = torch.zeros(self.num_syn, dtype=torch.float32)
    
    def get_current_contribution(self) -> torch.Tensor:
        """Total synaptic current contribution"""
        return torch.sum(self.y2)
    
    def get_membrane_contribution(self) -> torch.Tensor:
        """Direct synaptic contribution to membrane potential"""
        return torch.sum(self.y1_membrane_coeffs * self.y1 + self.y2_membrane_coeffs * self.y2)
    
    def integrate_step(self, syn_inputs: Optional[torch.Tensor] = None):
        """One step synaptic integration"""
        # Update synaptic state variables
        new_y2 = self.alpha_coupling_factors * self.y1 + self.alpha_decay_factors * self.y2
        new_y1 = self.alpha_decay_factors * self.y1
        
        # Apply new synaptic inputs
        if syn_inputs is not None:
            # Ensure input dimensionality matches number of synaptic receptors
            syn_inputs_padded = torch.zeros(self.num_syn, dtype=torch.float32)
            input_length = min(len(syn_inputs), self.num_syn)
            syn_inputs_padded[:input_length] = syn_inputs[:input_length]
            
            # Add weighted synaptic inputs to appropriate state variables
            new_y1 += self.input_scaling_factors * syn_inputs_padded
        
        self.y1 = new_y1
        self.y2 = new_y2
    
    def get_state(self):
        """Current synaptic state"""
        return {
            'y1': self.y1.detach().cpu().numpy(),
            'y2': self.y2.detach().cpu().numpy(),
            'current_total': self.get_current_contribution().item()
        }


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
        # Initialize surrogate function
        if surrogate_function is None:
            surrogate_function = surrogate.Sigmoid()

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
        
        self.V_m, self.V_th, self.g = V_m, V_th, g
        self.E_L, self.C_m, self.t_ref = E_L, C_m, t_ref
        self.V_reset, self.dt = V_reset, dt
        self.refractory_smoothness = refractory_smoothness
        
        # Compute derived membrane properties
        self.tau_m = C_m / g  # Membrane time constant (ms)
        self.num_asc = len(asc_decay)  # Number of after-spike current components
        self.num_syn = len(tau_syn)    # Number of synaptic receptor types
        
        # Pre-compute all integration parameters
        self._compute_params(asc_decay, asc_amps, tau_syn, asc_init)
        
        # Initialize all state variables to their resting values
        self._init_states()
    
    def _validate_params(self, V_m, V_th, g, E_L, C_m, t_ref, dt, asc_decay, asc_amps, tau_syn):
        """Validate all input parameters for biological plausibility and numerical stability."""
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
        self.asc_init = asc_init
        
        # Initialize membrane voltage integrator
        membrane_rate = 1.0 / self.tau_m
        self.membrane_integrator = ExponentialIntegrator(
            rates=membrane_rate, dt=self.dt, scaling=1.0/self.C_m
        )
        
        # Initialize ASC integrator
        self.asc_integrator = ExponentialIntegrator(
            rates=asc_decay, dt=self.dt, scaling=1/self.dt,
            amplitudes=asc_amps, t_ref=self.t_ref
        )
        
        # Initialize synapse processor, pass membrane parameters for coupling calculation
        membrane_params = {
            'tau_m': self.tau_m,
            'C_m': self.C_m
        }
        self.synapse_processor = AlphaSynapseProcessor(
            tau_syn=tau_syn, dt=self.dt, 
            membrane_params=membrane_params
        )
        
    def _init_states(self):
        """Initialize all dynamic state variables to their appropriate starting values."""
        # Initialize membrane potential relative to leak reversal potential
        initial_voltage = self.V_m - self.E_L
        self.U = torch.tensor(initial_voltage, dtype=torch.float32)
        
        # Initialize normalized voltage
        voltage_range = self.V_th - self.E_L
        initial_v_normalized = initial_voltage / voltage_range
        
        # Set v directly (BaseNode already registered it as memory)
        self.v = torch.tensor(initial_v_normalized, dtype=torch.float32)
        
        # Initialize after-spike current state vector
        self.asc = torch.tensor(self.asc_init, dtype=torch.float32)
        
        # Initialize refractory period counter
        self.ref_count = torch.tensor(0.0, dtype=torch.float32)
        
        # Reset synapse processor state
        self.synapse_processor.reset_states()
    
    def neuronal_charge(self, x: torch.Tensor, syn_inputs: Optional[torch.Tensor] = None):
        """
        Update all neuronal state variables for one integration time step.
        
        Now using modular integration components for better code reusability and maintainability.
        
        Parameters:
        -----------
        x : torch.Tensor
            External current injection (pA), can be scalar or tensor
        syn_inputs : torch.Tensor, optional
            Synaptic input currents for each receptor type
             shape should be (num_syn,) or broadcastable
             if None, no synaptic inputs are applied
        """
        # Handle refractory period
        not_refractory = torch.relu(1.0 - self.refractory_smoothness * self.ref_count)
        not_refractory = torch.clamp(not_refractory, 0.0, 1.0)
        
        # Calculate ASC contribution to membrane potential
        asc_contribution = self.asc_integrator.get_membrane_contribution(self.asc)
        
        # Update synaptic state and get membrane contribution
        self.synapse_processor.integrate_step(syn_inputs)
        synaptic_contribution = self.synapse_processor.get_membrane_contribution()
        
        # Calculate total input current
        total_current = x + asc_contribution
        
        # Update voltage using membrane voltage integrator
        self.U = self.membrane_integrator.integrate_step(
            current_state=self.U,
            input_current=total_current,
            modulation_factor=not_refractory
        )
        
        # Add synaptic contribution
        self.U += synaptic_contribution
        
        # Update ASC state using ASC integrator
        self.asc = self.asc_integrator.integrate_step(
            current_state=self.asc,
            modulation_factor=not_refractory
        )
        
        # Update normalized voltage
        voltage_range = self.V_th - self.E_L
        self.v = self.U / voltage_range
        
        # Update refractory period counter
        self.ref_count = torch.clamp(self.ref_count - self.dt, min=0.0)
    
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
        
        # Reset voltage
        if self.v_reset is None:
            # Soft reset: subtract threshold
            self.v = self.jit_soft_reset(self.v, spike_d, self.v_threshold)
        else:
            # Hard reset: set to reset value
            self.v = self.jit_hard_reset(self.v, spike_d, self.v_reset)
        
        # Update absolute voltage
        voltage_range = self.V_th - self.E_L
        self.U = self.v * voltage_range
        
        # Use ASC integrator to handle adaptation current updates after spike
        spike_continuous = spike  # Use continuous values directly
        self.asc = self.asc_integrator.apply_spike_increment(self.asc, spike_continuous)
        
        # Initiate refractory period
        spike_continuous = spike
        refractory_increment = self.t_ref * spike_continuous
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
            - 'modular_states': States from individual modular components
        """
        return {
            'v_mV': (self.U + self.E_L).item(),
            'U_rel_mV': self.U.item(),
            'asc_total': self.asc.sum().item(),
            'syn_total': self.synapse_processor.get_current_contribution().item(),
            'ref_count': self.ref_count.item(),
            'detailed': {
                'asc_components': self.asc.detach().cpu().numpy(),
                'synaptic_y1': self.synapse_processor.y1.detach().cpu().numpy(),
                'synaptic_y2': self.synapse_processor.y2.detach().cpu().numpy(),
            },
            'modular_states': {
                'synapse_processor': self.synapse_processor.get_state(),
                'asc_membrane_contribution': self.asc_integrator.get_membrane_contribution(self.asc).item(),
                'synaptic_membrane_contribution': self.synapse_processor.get_membrane_contribution().item()
            }
        }


# ============================================================================
# Utility Functions and Analysis Tools
# ============================================================================


def create_glif3_from_params(params_file, dt=0.1):
    """Load neuron parameters from Allen Institute JSON parameter file."""
    with open(params_file, 'r') as f:
        params = json.load(f)
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
    """Demonstrate the GLIF3 neuron implementation."""
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