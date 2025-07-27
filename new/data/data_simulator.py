"""
Ship Steering Gear Hydraulic System Data Simulator
Simulates AMESim-like data for normal and fault conditions
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class SteeringGearSimulator:
    """Simulates hydraulic steering gear system data with various fault conditions"""
    
    def __init__(self, sampling_rate: float = 100.0, duration: float = 10.0):
        """
        Initialize simulator
        
        Args:
            sampling_rate: Sampling frequency in Hz
            duration: Simulation duration in seconds
        """
        self.sampling_rate = sampling_rate
        self.duration = duration
        self.time_steps = int(sampling_rate * duration)
        self.time = np.linspace(0, duration, self.time_steps)
        
        # System parameters
        self.base_pressure = 150.0  # bar
        self.base_displacement = 0.05  # m
        self.base_current = 2.0  # A
        
    def generate_normal_condition(self, noise_level: float = 0.02) -> Dict[str, np.ndarray]:
        """Generate normal operating condition data"""
        
        # Pump outlet pressure (relatively stable with small fluctuations)
        pump_pressure = self.base_pressure + 10 * np.sin(2 * np.pi * 0.5 * self.time) + \
                       5 * np.sin(2 * np.pi * 2 * self.time) + \
                       noise_level * self.base_pressure * np.random.randn(self.time_steps)
        
        # Steering cylinder A chamber pressure
        cylinder_a_pressure = self.base_pressure * 0.8 + \
                             15 * np.sin(2 * np.pi * 0.3 * self.time) + \
                             noise_level * self.base_pressure * 0.8 * np.random.randn(self.time_steps)
        
        # Steering cylinder B chamber pressure
        cylinder_b_pressure = self.base_pressure * 0.7 + \
                             12 * np.sin(2 * np.pi * 0.4 * self.time + np.pi/4) + \
                             noise_level * self.base_pressure * 0.7 * np.random.randn(self.time_steps)
        
        # Steering cylinder piston displacement
        displacement = self.base_displacement * np.sin(2 * np.pi * 0.2 * self.time) + \
                      noise_level * self.base_displacement * np.random.randn(self.time_steps)
        
        # Servo valve control current
        control_current = self.base_current + 0.5 * np.sin(2 * np.pi * 0.3 * self.time) + \
                         noise_level * self.base_current * np.random.randn(self.time_steps)
        
        return {
            'pump_outlet_pressure': pump_pressure,
            'cylinder_a_pressure': cylinder_a_pressure,
            'cylinder_b_pressure': cylinder_b_pressure,
            'piston_displacement': displacement,
            'servo_valve_current': control_current,
            'label': np.zeros(self.time_steps)  # 0 for normal
        }
    
    def generate_air_contamination_fault(self, severity: float = 0.3, noise_level: float = 0.02) -> Dict[str, np.ndarray]:
        """Generate hydraulic oil air contamination fault data"""
        
        # Air contamination causes pressure fluctuations and reduced efficiency
        contamination_factor = 1 - severity * 0.5
        fluctuation_amplitude = severity * 20
        
        pump_pressure = self.base_pressure * contamination_factor + \
                       fluctuation_amplitude * np.sin(2 * np.pi * 3 * self.time) + \
                       fluctuation_amplitude * 0.5 * np.sin(2 * np.pi * 8 * self.time) + \
                       noise_level * self.base_pressure * np.random.randn(self.time_steps)
        
        cylinder_a_pressure = self.base_pressure * 0.8 * contamination_factor + \
                             fluctuation_amplitude * 0.8 * np.sin(2 * np.pi * 2.5 * self.time) + \
                             noise_level * self.base_pressure * 0.8 * np.random.randn(self.time_steps)
        
        cylinder_b_pressure = self.base_pressure * 0.7 * contamination_factor + \
                             fluctuation_amplitude * 0.7 * np.sin(2 * np.pi * 3.2 * self.time) + \
                             noise_level * self.base_pressure * 0.7 * np.random.randn(self.time_steps)
        
        displacement = self.base_displacement * np.sin(2 * np.pi * 0.2 * self.time) * contamination_factor + \
                      noise_level * self.base_displacement * np.random.randn(self.time_steps)
        
        control_current = self.base_current * (1 + severity * 0.3) + \
                         0.5 * np.sin(2 * np.pi * 0.3 * self.time) + \
                         noise_level * self.base_current * np.random.randn(self.time_steps)
        
        return {
            'pump_outlet_pressure': pump_pressure,
            'cylinder_a_pressure': cylinder_a_pressure,
            'cylinder_b_pressure': cylinder_b_pressure,
            'piston_displacement': displacement,
            'servo_valve_current': control_current,
            'label': np.ones(self.time_steps)  # 1 for air contamination
        }
    
    def generate_internal_leakage_fault(self, severity: float = 0.4, noise_level: float = 0.02) -> Dict[str, np.ndarray]:
        """Generate hydraulic cylinder internal leakage fault data"""
        
        # Internal leakage causes pressure drop and reduced displacement efficiency
        leakage_factor = 1 - severity * 0.6
        pressure_drop = severity * 30
        
        pump_pressure = self.base_pressure + 10 * np.sin(2 * np.pi * 0.5 * self.time) + \
                       noise_level * self.base_pressure * np.random.randn(self.time_steps)
        
        cylinder_a_pressure = (self.base_pressure * 0.8 - pressure_drop) + \
                             10 * np.sin(2 * np.pi * 0.3 * self.time) * leakage_factor + \
                             noise_level * self.base_pressure * 0.8 * np.random.randn(self.time_steps)
        
        cylinder_b_pressure = (self.base_pressure * 0.7 - pressure_drop * 0.8) + \
                             8 * np.sin(2 * np.pi * 0.4 * self.time) * leakage_factor + \
                             noise_level * self.base_pressure * 0.7 * np.random.randn(self.time_steps)
        
        displacement = self.base_displacement * np.sin(2 * np.pi * 0.2 * self.time) * leakage_factor + \
                      noise_level * self.base_displacement * np.random.randn(self.time_steps)
        
        control_current = self.base_current * (1 + severity * 0.4) + \
                         0.5 * np.sin(2 * np.pi * 0.3 * self.time) + \
                         noise_level * self.base_current * np.random.randn(self.time_steps)
        
        return {
            'pump_outlet_pressure': pump_pressure,
            'cylinder_a_pressure': cylinder_a_pressure,
            'cylinder_b_pressure': cylinder_b_pressure,
            'piston_displacement': displacement,
            'servo_valve_current': control_current,
            'label': np.full(self.time_steps, 2)  # 2 for internal leakage
        }
    
    def generate_valve_sticking_fault(self, severity: float = 0.5, noise_level: float = 0.02) -> Dict[str, np.ndarray]:
        """Generate valve core sticking fault data"""
        
        # Valve sticking causes irregular pressure patterns and delayed response
        sticking_events = np.random.poisson(severity * 5, self.time_steps)
        sticking_mask = sticking_events > 0
        
        pump_pressure = self.base_pressure + 10 * np.sin(2 * np.pi * 0.5 * self.time) + \
                       noise_level * self.base_pressure * np.random.randn(self.time_steps)
        
        # Add sudden pressure spikes due to valve sticking
        pump_pressure[sticking_mask] += severity * 40 * np.random.randn(np.sum(sticking_mask))
        
        cylinder_a_pressure = self.base_pressure * 0.8 + \
                             15 * np.sin(2 * np.pi * 0.3 * self.time) + \
                             noise_level * self.base_pressure * 0.8 * np.random.randn(self.time_steps)
        cylinder_a_pressure[sticking_mask] += severity * 25 * np.random.randn(np.sum(sticking_mask))
        
        cylinder_b_pressure = self.base_pressure * 0.7 + \
                             12 * np.sin(2 * np.pi * 0.4 * self.time) + \
                             noise_level * self.base_pressure * 0.7 * np.random.randn(self.time_steps)
        cylinder_b_pressure[sticking_mask] += severity * 20 * np.random.randn(np.sum(sticking_mask))
        
        displacement = self.base_displacement * np.sin(2 * np.pi * 0.2 * self.time) + \
                      noise_level * self.base_displacement * np.random.randn(self.time_steps)
        
        control_current = self.base_current + 0.5 * np.sin(2 * np.pi * 0.3 * self.time) + \
                         noise_level * self.base_current * np.random.randn(self.time_steps)
        control_current[sticking_mask] += severity * 1.5 * np.random.randn(np.sum(sticking_mask))
        
        return {
            'pump_outlet_pressure': pump_pressure,
            'cylinder_a_pressure': cylinder_a_pressure,
            'cylinder_b_pressure': cylinder_b_pressure,
            'piston_displacement': displacement,
            'servo_valve_current': control_current,
            'label': np.full(self.time_steps, 3)  # 3 for valve sticking
        }
    
    def generate_filter_blockage_fault(self, severity: float = 0.6, noise_level: float = 0.02) -> Dict[str, np.ndarray]:
        """Generate oil filter blockage fault data"""
        
        # Filter blockage causes overall pressure reduction and flow restrictions
        blockage_factor = 1 - severity * 0.4
        pressure_reduction = severity * 25
        
        pump_pressure = (self.base_pressure - pressure_reduction) * blockage_factor + \
                       8 * np.sin(2 * np.pi * 0.5 * self.time) + \
                       noise_level * self.base_pressure * np.random.randn(self.time_steps)
        
        cylinder_a_pressure = (self.base_pressure * 0.8 - pressure_reduction * 0.8) * blockage_factor + \
                             10 * np.sin(2 * np.pi * 0.3 * self.time) + \
                             noise_level * self.base_pressure * 0.8 * np.random.randn(self.time_steps)
        
        cylinder_b_pressure = (self.base_pressure * 0.7 - pressure_reduction * 0.7) * blockage_factor + \
                             8 * np.sin(2 * np.pi * 0.4 * self.time) + \
                             noise_level * self.base_pressure * 0.7 * np.random.randn(self.time_steps)
        
        displacement = self.base_displacement * np.sin(2 * np.pi * 0.2 * self.time) * blockage_factor + \
                      noise_level * self.base_displacement * np.random.randn(self.time_steps)
        
        control_current = self.base_current * (1 + severity * 0.5) + \
                         0.5 * np.sin(2 * np.pi * 0.3 * self.time) + \
                         noise_level * self.base_current * np.random.randn(self.time_steps)
        
        return {
            'pump_outlet_pressure': pump_pressure,
            'cylinder_a_pressure': cylinder_a_pressure,
            'cylinder_b_pressure': cylinder_b_pressure,
            'piston_displacement': displacement,
            'servo_valve_current': control_current,
            'label': np.full(self.time_steps, 4)  # 4 for filter blockage
        }
    
    def generate_dataset(self, samples_per_condition: int = 100, 
                        noise_levels: List[float] = [0.02]) -> pd.DataFrame:
        """
        Generate complete dataset with all conditions
        
        Args:
            samples_per_condition: Number of samples per fault condition
            noise_levels: List of noise levels to simulate
            
        Returns:
            DataFrame with all simulated data
        """
        all_data = []
        condition_names = ['Normal', 'Air Contamination', 'Internal Leakage', 
                          'Valve Sticking', 'Filter Blockage']
        
        for noise_level in noise_levels:
            for i in range(samples_per_condition):
                # Generate data for each condition
                conditions_data = [
                    self.generate_normal_condition(noise_level),
                    self.generate_air_contamination_fault(noise_level=noise_level),
                    self.generate_internal_leakage_fault(noise_level=noise_level),
                    self.generate_valve_sticking_fault(noise_level=noise_level),
                    self.generate_filter_blockage_fault(noise_level=noise_level)
                ]
                
                for condition_data in conditions_data:
                    # Create sample record
                    sample_data = {
                        'sample_id': len(all_data),
                        'condition': condition_names[int(condition_data['label'][0])],
                        'noise_level': noise_level,
                        **{key: value for key, value in condition_data.items() if key != 'label'},
                        'label': condition_data['label'][0]
                    }
                    all_data.append(sample_data)
        
        return pd.DataFrame(all_data)


def add_noise_to_signal(signal: np.ndarray, snr_db: float) -> np.ndarray:
    """Add noise to signal based on SNR"""
    signal_power = np.mean(signal ** 2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = np.sqrt(noise_power) * np.random.randn(len(signal))
    return signal + noise
