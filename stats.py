"""Core calculation logic: statistics"""

import logging
import pandas as pd
import numpy as np

# Assuming ConfigModel and exceptions are importable from sibling modules
from .exceptions import CalculationError

# Get the logger instance configured in utils
logger = logging.getLogger("sdc_checker.stats")


# --- Weighted Median Function (from initial_plan_specs/sdc.md) ---

### Weighted median function following the ONS Annual Survey of Hours and Earnings method (cumulative weighted median)
def weighted_median_interpolation(values, weights):
    """
    Calculate weighted median using interpolation between values.
    Handles NaN values by removing pairs where either value or weight is NaN.

    This function implements an interpolation approach that mirrors the ASHE median method.
    Linear interpolation is applied when the cumulative weight crosses the halfway point.
    If the halfway point lands exactly at a value and it's the first in the list, that value is returned.
    This can differ to other median functions that use the next value method.

    Args:
        values: Array-like of values
        weights: Array-like of weights

    Returns:
        The weighted median or np.nan if no valid data
    """
    # Ensure numpy is imported within the function's scope if it wasn't at the top
    # import numpy as np # Already imported at module level

    # Convert to numpy arrays for easier NaN handling
    values = np.array(values)
    weights = np.array(weights)

    # Filter out NaN values from both arrays
    mask = ~(np.isnan(values) | np.isnan(weights))
    filtered_values = values[mask]
    filtered_weights = weights[mask]

    # New check for negative weights
    if np.any(filtered_weights < 0):
        # Raise CalculationError consistent with module exceptions
        raise CalculationError("Negative weights are not allowed in weighted median calculation.")

    # Check if we have data left after filtering
    if len(filtered_values) == 0:
        logger.debug("Weighted median: No valid data after filtering NaNs.")
        return np.nan

    # Sort values and corresponding weights
    indices = np.argsort(filtered_values)
    sorted_values = filtered_values[indices]
    sorted_weights = filtered_weights[indices]

    # Calculate total weight and midpoint
    total_weight = np.sum(sorted_weights)
    if total_weight == 0:  # new check for zero total weight
        logger.debug("Weighted median: Total weight is zero after filtering.")
        return np.nan
    half_weight = total_weight / 2.0

    # Find the weighted median with interpolation
    cumulative_weight = 0
    for i, (value, weight) in enumerate(zip(sorted_values, sorted_weights)):
        previous_cumulative = cumulative_weight
        cumulative_weight += weight
        # Check if the midpoint falls within the current weight's range
        # Use >= for half_weight to handle exact matches correctly
        if cumulative_weight >= half_weight and previous_cumulative < half_weight:
            # If the half_weight is exactly the cumulative weight AND it's not the first element,
            # the median is the current value (as per ONS ASHE method description implicitly).
            # The interpolation handles this case correctly too, but being explicit might be clearer.
            # Let's stick to the provided interpolation logic which covers this.

            if i == 0:
                # If the first element's weight alone meets/exceeds half_weight
                logger.debug(f"Weighted median: Crossover at first element. Value: {value}")
                return value
            else:
                prev_value = sorted_values[i - 1]
                # Avoid division by zero if cumulative_weight == previous_cumulative (shouldn't happen with positive weights)
                weight_diff = cumulative_weight - previous_cumulative
                if weight_diff <= 0:
                    # This case implies zero or negative weight, which should have been filtered or raised error
                    logger.warning(f"Weighted median: Encountered non-positive weight difference ({weight_diff}) during interpolation. Returning current value {value}.")
                    return value # Fallback, though should ideally not be reached

                proportion = (half_weight - previous_cumulative) / weight_diff
                interpolated_median = prev_value + proportion * (value - prev_value)
                logger.debug(f"Weighted median: Interpolating between {prev_value} and {value}. Proportion: {proportion:.4f}. Result: {interpolated_median:.4f}")
                return interpolated_median

    # Fallback error if logic fails (e.g., empty data somehow bypassed checks)
    # This should theoretically not be reached if input validation and filtering work.
    logger.error("Weighted median: Failed to find weighted median despite valid data checks. This indicates a logic error.")
    raise CalculationError("Failed to find weighted median. This should not happen with valid data.")


# --- Individual SDC Indicator Helper Functions ---

def _calculate_sdc_count(abs_values: pd.Series) -> int:
    """Calculates the count of non-NaN absolute values."""
    return len(abs_values)

def _calculate_sdc_max(abs_values: pd.Series) -> float:
    """Calculates the maximum of non-NaN absolute values."""
    return abs_values.max() if not abs_values.empty else np.nan

def _calculate_sdc_second_max(abs_values: pd.Series) -> float:
    """Calculates the second maximum of non-NaN absolute values."""
    count = len(abs_values)
    if count > 1:
        return abs_values.nlargest(2).iloc[-1]
    else:
        return np.nan

def _calculate_sdc_sum_third_to_nth(abs_values: pd.Series) -> float:
    """Calculates the sum of the 3rd to nth largest non-NaN absolute values."""
    count = len(abs_values)
    if count > 2:
        # Sum all values excluding the top 2 largest
        return abs_values.sum() - abs_values.nlargest(2).sum()
    else:
        return 0.0 # Return 0 if count <= 2

def _calculate_sdc_total(abs_values: pd.Series) -> float:
    """Calculates the sum (total) of non-NaN absolute values."""
    return abs_values.sum()

def _calculate_sdc_dominance_1k(abs_values: pd.Series) -> float:
    """Calculates the Dominance (1,k) check ratio (Max / Total)."""
    max_val = _calculate_sdc_max(abs_values)
    total = _calculate_sdc_total(abs_values)
    # Handle division by zero or NaN max/total
    if total != 0 and not pd.isna(total) and not pd.isna(max_val):
        return max_val / total
    else:
        return np.nan

def _calculate_sdc_dominance_p_percent(abs_values: pd.Series) -> float:
    """Calculates the Dominance p% check: Sum(3rd..nth) / Max."""
    # Check for empty or all-NaN input early
    if abs_values.empty or abs_values.isna().all():
        logger.debug("Dominance p%: Input empty or all NaN.")
        return np.nan

    # Calculate necessary components using existing helpers
    max_val = _calculate_sdc_max(abs_values)
    sum_3_nth = _calculate_sdc_sum_third_to_nth(abs_values)

    # Check if max_val is NaN (can happen if input was empty/all NaN) or zero
    if pd.isna(max_val) or max_val == 0:
        logger.debug(f"Dominance p%: Max value is {max_val}. Cannot calculate ratio, returning NaN.")
        # Return NaN if max is 0 or NaN to avoid division by zero or invalid result.
        # If sum_3_nth is also 0, 0/0 is undefined -> NaN.
        # If sum_3_nth is non-zero, non-zero/0 is undefined -> NaN.
        return np.nan

    # If max_val is valid and non-zero, perform the division.
    # sum_3_nth can be 0.0 (e.g., if count <= 2), resulting in 0.0 / max_val = 0.0, which is valid.
    result = sum_3_nth / max_val
    logger.debug(f"Dominance p%: Sum(3rd..nth)={sum_3_nth}, Max={max_val}, Result={result}")
    return result