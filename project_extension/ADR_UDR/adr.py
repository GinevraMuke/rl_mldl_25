import random
import math
from collections import deque
import numpy as np

class AutomaticDomainRandomizer:
    """
    ADR version that operates directly on NumPy arrays instead of dictionaries,
    to interface with environments requiring a vector format.
    """
    def __init__(self, param_names: list, initial_values: np.ndarray, 
                 perf_threshold_low: float, perf_threshold_high: float, 
                 buffer_size: int = 5, step_size: float = 0.01, max_range_percentage: float = 0.5):
        
        if not perf_threshold_low < perf_threshold_high:
            raise ValueError("Low performance threshold must be less than the high threshold.")

        self.param_names = list(param_names)
        self.initial_values = np.copy(initial_values).astype(np.float32)
        self.low_bounds = np.copy(initial_values).astype(np.float32)
        self.high_bounds = np.copy(initial_values).astype(np.float32)
        
        self.perf_threshold_low = perf_threshold_low
        self.perf_threshold_high = perf_threshold_high
        self.step_size = step_size
        
        # Performance buffers still use parameter names as keys
        self.perf_buffers = {name: {'low': deque(maxlen=buffer_size), 'high': deque(maxlen=buffer_size)} 
                             for name in self.param_names}
        
        self.max_range_percentage = max_range_percentage
        # Calculate absolute min/max bounds based on a percentage of the initial values
        self.abs_min_bounds = self.initial_values * (1 - self.max_range_percentage)
        self.abs_max_bounds = self.initial_values * (1 + self.max_range_percentage)
        # Ensure minimum bounds are not negative (important for physical properties like mass)
        self.abs_min_bounds = np.maximum(self.abs_min_bounds, 0)
        
        print("ADR (Array-based, with range limits) initialized.")
        print(f"Max range set to +/- {self.max_range_percentage*100}%.")
        self.__repr__()

    def sample_env(self) -> np.ndarray:
        """Sample a parameter array from the current range."""
        return np.random.uniform(low=self.low_bounds, high=self.high_bounds)

    def boundary_sample(self):
        """Sample a parameter array from a specific boundary to test performance."""
        # Choose a random parameter and boundary to test
        param_idx = random.randrange(len(self.param_names))
        param_name = self.param_names[param_idx]
        boundary_type = random.choice(['low', 'high'])
        
        # Sample a base configuration
        env_config_array = self.sample_env()
        
        # Overwrite the value at the chosen boundary
        if boundary_type == 'low':
            env_config_array[param_idx] = self.low_bounds[param_idx]
        else:
            env_config_array[param_idx] = self.high_bounds[param_idx]
        
        return env_config_array, param_name, boundary_type

    def update(self, param_name: str, boundary_type: str, performance_score: float):
        """Update bounds based on performance, applying all constraints."""
        buffer = self.perf_buffers[param_name][boundary_type]
        buffer.append(performance_score)
        
        if len(buffer) == buffer.maxlen:
            avg_perf = sum(buffer) / len(buffer)
            param_idx = self.param_names.index(param_name)
            
            # 1. Compute the proposed new bound
            if avg_perf > self.perf_threshold_high:
                print(f"INFO: High performance ({avg_perf:.2f}) on {param_name}/{boundary_type}. Attempting to expand.")
                new_bound = self.low_bounds[param_idx] - self.step_size if boundary_type == 'low' else self.high_bounds[param_idx] + self.step_size
            
            elif avg_perf < self.perf_threshold_low:
                print(f"INFO: Low performance ({avg_perf:.2f}) on {param_name}/{boundary_type}. Attempting to shrink.")
                new_bound = self.low_bounds[param_idx] + self.step_size if boundary_type == 'low' else self.high_bounds[param_idx] - self.step_size
            else:
                # Performance is in the intermediate range, do nothing
                buffer.clear()
                return
                
            # 2. Apply constraints to the new bound
            # First constraint: Clip to absolute min/max limits
            clipped_bound = np.clip(
                new_bound,
                self.abs_min_bounds[param_idx],
                self.abs_max_bounds[param_idx]
            )

            # Second constraint: Prevent low and high bounds from crossing
            if boundary_type == 'low':
                final_bound = min(clipped_bound, self.high_bounds[param_idx])
                self.low_bounds[param_idx] = final_bound
            else: # high
                final_bound = max(clipped_bound, self.low_bounds[param_idx])
                self.high_bounds[param_idx] = final_bound
                
            buffer.clear()

    def get_entropy(self) -> float:
        """Calculate entropy based on the logarithmic width of the ranges."""
        range_widths = self.high_bounds - self.low_bounds
        # Add epsilon to avoid log(0) for zero-width ranges
        total_log_range = np.sum(np.log(range_widths + 1e-9))
        return total_log_range / len(self.param_names)

    def __repr__(self):
        print("--- Current ADR State ---")
        for i, name in enumerate(self.param_names):
            print(f"- {name}: low={self.low_bounds[i]:.4f}, high={self.high_bounds[i]:.4f}")
        print(f"ADR Entropy: {self.get_entropy():.4f} nats/dim")
        print("-----------------------")
        return "" # __repr__ must return a string