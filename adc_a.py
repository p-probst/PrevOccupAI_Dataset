# --- ADC to ACCELERATION CONVERSION ---
import numpy as np

def adc_to_acceleration(adc_values, n_bits=16, g_range=8):
    """
    Convert raw ADC accelerometer values to acceleration in m/s².
    Parameters:
        adc_values: pd.Series, np.ndarray, or list of raw ADC values
        n_bits: Number of ADC bits (e.g., 16)
        g_range: ±g_range is the accelerometer range (e.g., 8 for ±8g)
    Returns:
        Converted acceleration in m/s²
    """
    # Convert to numpy array to ensure compatibility
    adc_values = np.asarray(adc_values)
    
    midpoint = 2 ** n_bits / 2
    full_span = 2 * g_range
    g_acceleration = (adc_values - midpoint) * (full_span / (2 ** n_bits))
    return g_acceleration * 9.80665  # Convert from g to m/s²