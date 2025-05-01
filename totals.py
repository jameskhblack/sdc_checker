"""Handles the creation of "Aggregated" columns and rows and supports aggregation"""

import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Callable, Optional, Tuple, Set, Union # Added Tuple, Set, Union
from itertools import product # For generating grouping combinations

# Assuming ConfigModel and exceptions are importable from sibling modules
from .config import ConfigModel
from .exceptions import CalculationError
from .stats import (
    weighted_median_interpolation, _calculate_sdc_count, _calculate_sdc_max,
    _calculate_sdc_second_max, _calculate_sdc_sum_third_to_nth, _calculate_sdc_total,
    _calculate_sdc_dominance_1k, _calculate_sdc_dominance_p_percent
)

# Get the logger instance configured in utils
logger = logging.getLogger("sdc_checker.calculator")
from .utils import get_sort_key # For hierarchical sorting

# --- Helper Functions for Aggregation ---

def _replace_nan_in_multiindex(index: pd.MultiIndex) -> pd.MultiIndex:
    """Replaces NaN values with 'Missing' string in each level of a MultiIndex."""
    new_levels = []
    for level in index.levels:
        if level.isna().any():
            # Ensure object dtype before fillna if mixed types or NaN present
            new_level = level.astype(object).fillna('Missing')
        else:
            new_level = level
        new_levels.append(new_level)
    return index.set_levels(new_levels)


def _agg_wrapper(group: pd.DataFrame, value_col: str, weight_col: Optional[str], statistic: str) -> Any:
    """
    Internal wrapper to apply the correct aggregation based on config.
    Handles weighted/unweighted sum/median/count and all-NaN cases.

    Args:
        group: The DataFrame group from groupby().
        value_col: The name of the value column.
        weight_col: The name of the weight column, if any.
        statistic: 'sum', 'median', or 'count'.

    Returns:
        The calculated aggregate value, or np.nan.
    """
    values = group[value_col]
    weights = group[weight_col] if weight_col and weight_col in group.columns else None

    # Ensure weights is a Series/array if weight_col is valid
    if weight_col and weights is None:
        logger.warning(f"Weight column '{weight_col}' specified but not found in group, proceeding unweighted.")


    if statistic == 'sum':
        if weight_col is not None and weights is not None:
            # Weighted Sum: Handle NaNs carefully as per spec
            # Combine masks for valid values and valid weights
            valid_mask = ~values.isna() & ~weights.isna()
            if not valid_mask.any():
                return np.nan
            # Ensure multiplication only happens on valid pairs
            weighted_values = values[valid_mask] * weights[valid_mask]
            # np.sum returns 0 for an empty array, which is correct if all inputs were NaN/filtered out
            return np.sum(weighted_values)
        else:
            # Unweighted Sum: Return NaN if all values in the group are NaN
            if values.isna().all():
                return np.nan
            # Use pandas sum with skipna=True, but the check above handles the all-NaN case returning NaN.
            # Ensure the result is float if input contains NaN, otherwise pandas might return 0 for all-NaN integer sum
            if values.dtype == 'object' or pd.api.types.is_numeric_dtype(values):
                 # Check if all non-NaN values are zero
                 non_nan_values = values.dropna()
                 if not non_nan_values.empty and (non_nan_values == 0).all():
                     return 0.0 # Return float 0.0
                 # Pandas sum handles mixed types and NaN correctly for numeric/object
                 result = values.sum(skipna=True)
                 # Ensure float output if NaNs were present or result is 0
                 if values.isna().any() or result == 0:
                     return float(result)
                 return result
            else:
                 # Fallback for unexpected types, though primary use is numeric
                 return values.sum(skipna=True)


    elif statistic == 'median':
        if weight_col is not None and weights is not None:
            # Weighted Median: Use the provided interpolation function
            return weighted_median_interpolation(values, weights)
        else:
            # Unweighted Median: Pandas median handles skipna and returns NaN for all-NaN
            return values.median(skipna=True)
        
    elif statistic == 'count':
        # Identify non-NaN values
        valid_values_mask = values.notna()

        # All values are NaN, return count 0
        if not valid_values_mask.any():
            return 0
        
        # Weighted Count: Count non-NaN values in the weight column
        if weight_col is not None and weights is not None:
            # Combine masks for valid values and valid weights
            valid_weights_mask = weights.notna()
            combined_mask = valid_values_mask & valid_weights_mask

            # No observations have both a valid value and a valid weight
            if not combined_mask.any():
                return 0

            # WEIGHTED COUNT of non-NaN values.
            else:
                return weights[combined_mask].sum()
        # RETURN UNWEIGHTED COUNT of non-NaN values.
        else:
            return valid_values_mask.sum()

    else:
        # Should be caught by config validation, but raise error just in case
        raise CalculationError(f"Unsupported statistic type '{statistic}' in _agg_wrapper.")

def _add_hierarchical_aggregates(
    base_df: pd.DataFrame,
    original_df: pd.DataFrame,
    config: ConfigModel, # Added config parameter
    axis: int, # Added axis parameter (0 for rows, 1 for columns)
    vars_on_axis: List[str], # Added vars_on_axis parameter
    vars_on_other_axis: List[str], # e.g., config.col_var or config.row_var
    agg_label: str = 'Aggregated'
) -> pd.DataFrame:
    """
    Adds hierarchical 'Aggregated' entries along a specified axis for the PRIMARY statistic table.

    Calculates aggregate values based on the original input data and the
    primary config.statistic for each level of the hierarchy.

    Args:
        base_df: The pivot table DataFrame to add aggregates to.
        original_df: The original, unpivoted DataFrame.
        config: The validated configuration object.
        axis: 0 to add aggregated rows (index), 1 to add aggregated columns.
        vars_on_axis: List of variable names on the axis being aggregated.
        vars_on_other_axis: List of variable names on the other axis.
        agg_label: Label to use for the aggregated entries.

    Returns:
        A new DataFrame with hierarchical aggregates added and sorted.

    Raises:
        CalculationError: If errors occur during aggregation or processing.
    """
    logger.info(f"Adding primary statistic hierarchical aggregates on axis {axis} for vars: {vars_on_axis}")

    try:
        # 1. Select Target Axis
        if axis == 0:
            target_axis_obj = base_df.index
            other_axis_obj = base_df.columns
            logger.debug("Target axis: index (rows)")
        elif axis == 1:
            target_axis_obj = base_df.columns
            other_axis_obj = base_df.index
            logger.debug("Target axis: columns")
        else:
            raise ValueError("Axis must be 0 (index) or 1 (columns)")

        # 2. Handle Simple Index (Non-MultiIndex)
        if not isinstance(target_axis_obj, pd.MultiIndex):
            logger.debug(f"Target axis '{target_axis_obj.name or 'index/columns'}' is not a MultiIndex. Calculating overall total.")
            # Calculate overall total Series using the *other* axis vars for grouping
            overall_total_series = _calculate_aggregate_totals(
                original_df, vars_on_other_axis, config, agg_label
            )
            # The result of _calculate_aggregate_totals is indexed by vars_on_other_axis
            # We need to align it correctly for appending

            if axis == 0: # Append as a new row
                # The series index should match the base_df columns
                if not isinstance(other_axis_obj, pd.MultiIndex):
                     # If columns are simple index, series index should also be simple
                     # _calculate_aggregate_totals might return MultiIndex if vars_on_other_axis has >1 item
                     # Need to ensure alignment. If other axis is simple, group_vars should have len 1.
                     if len(vars_on_other_axis) == 1:
                         # Check if the index is already simple
                         if isinstance(overall_total_series.index, pd.MultiIndex):
                             overall_total_series.index = overall_total_series.index.get_level_values(0)
                     elif not vars_on_other_axis and len(overall_total_series) == 1: # Grand total case
                         pass # Index is already simple ['Aggregated']
                     else:
                         # This case might be complex - simple columns but multiple grouping vars for total?
                         logger.warning("Potential index mismatch: Simple columns but multiple vars for total calculation.")
                # Create a DataFrame row with the correct index name
                total_df_row = overall_total_series.to_frame(name=agg_label).T
                total_df_row.index = pd.Index([agg_label], name=target_axis_obj.name)
                result_df = pd.concat([base_df, total_df_row], axis=0)
            else: # Append as a new column
                # The series index should match the base_df index
                if not isinstance(other_axis_obj, pd.MultiIndex):
                    if len(vars_on_other_axis) == 1:
                         # Check if the index is already simple
                         if isinstance(overall_total_series.index, pd.MultiIndex):
                             overall_total_series.index = overall_total_series.index.get_level_values(0)
                    elif not vars_on_other_axis and len(overall_total_series) == 1: # Grand total case
                         pass # Index is already simple ['Aggregated']
                    else:
                         logger.warning("Potential index mismatch: Simple index but multiple vars for total calculation.")
                # Add as a new column
                result_df = base_df.copy()
                result_df[agg_label] = overall_total_series

            logger.debug(f"Simple index aggregation complete. Result shape: {result_df.shape}")
            # No complex sorting needed for simple index aggregate
            return result_df

        # --- MultiIndex Logic ---
        logger.debug(f"Target axis '{target_axis_obj.names}' is a MultiIndex.")
        n_levels = target_axis_obj.nlevels

        # 3. Generate Target Tuples (MultiIndex)
        all_tuples: Set[Tuple] = set(target_axis_obj)
        logger.debug(f"Initial tuples from target axis: {len(all_tuples)}")

        for level_idx in range(n_levels): # Iterate from 0 to n_levels - 1
            # Get unique prefixes of length level_idx
            # For level_idx = 0, prefixes are empty tuples ()
            # For level_idx = 1, prefixes are tuples of length 1 (first level values)
            # ...
            # For level_idx = n_levels - 1, prefixes are tuples of length n_levels - 1
            unique_prefixes = set(t[:level_idx] for t in target_axis_obj)
            logger.debug(f"Level {level_idx}: Found {len(unique_prefixes)} unique prefixes.")
            for prefix in unique_prefixes:
                # Create the aggregated tuple: prefix + ('Aggregated', 'Aggregated', ...)
                # The number of 'Aggregated' labels is n_levels - level_idx
                agg_tuple = prefix + (agg_label,) * (n_levels - level_idx)
                all_tuples.add(agg_tuple)
                # logger.debug(f"  Added aggregate tuple: {agg_tuple}") # Can be very verbose


        # Add the overall aggregate tuple ('Aggregated', 'Aggregated', ...) - This should be covered by level_idx=0 loop
        # overall_agg_tuple = (agg_label,) * n_levels
        # all_tuples.add(overall_agg_tuple)
        logger.debug(f"Total unique tuples including aggregates: {len(all_tuples)}")

        # 4. Create Target Structure
        target_multiindex = pd.MultiIndex.from_tuples(list(all_tuples), names=target_axis_obj.names)
        logger.debug(f"Created target MultiIndex with {len(target_multiindex)} entries.")

        # 5. Initialize Result DataFrame
        if axis == 0:
            result_df = pd.DataFrame(np.nan, index=target_multiindex, columns=other_axis_obj)
            logger.debug(f"Initialized result DataFrame with shape: ({len(target_multiindex)}, {len(other_axis_obj) if other_axis_obj is not None else 0}) index: {target_multiindex.names}, columns: {other_axis_obj.names if other_axis_obj is not None else None}")
        else: # axis == 1
            result_df = pd.DataFrame(np.nan, index=other_axis_obj, columns=target_multiindex)
            logger.debug(f"Initialized result DataFrame with shape: ({len(other_axis_obj) if other_axis_obj is not None else 0}, {len(target_multiindex)}) index: {other_axis_obj.names if other_axis_obj is not None else None}, columns: {target_multiindex.names}")

        # 6. Populate Original Values
        logger.debug("Populating result DataFrame with original values from base_df...")
        # Update aligns on index and columns, fills NaNs where labels don't match
        result_df.update(base_df)
        logger.debug("Original values populated.")
        # Add check for NaNs introduced if needed: logger.debug(f"NaN count after update: {result_df.isna().sum().sum()}")

        # 7. Calculate & Populate Aggregates
        new_agg_tuples = all_tuples - set(target_axis_obj)
        logger.info(f"Calculating {len(new_agg_tuples)} new primary aggregate values...")

        for target_tuple in new_agg_tuples:
            # logger.debug(f"  Calculating aggregates for target tuple: {target_tuple}")

            # Determine the actual grouping variables for this aggregate tuple
            # These are the variables corresponding to the non-'Aggregated' parts of the tuple
            grouping_vars_target = []
            target_filters = {}
            for i in range(n_levels):
                level_val = target_tuple[i]
                level_name = vars_on_axis[i]
                if level_val != agg_label:
                    grouping_vars_target.append(level_name)
                    target_filters[level_name] = level_val # Store filter value

            # Calculate aggregates across the *other* axis, grouped by the other axis variables
            # but filtered by the non-aggregated parts of the *target* axis
            # logger.debug(f"    Target filters: {target_filters}")
            # logger.debug(f"    Grouping by other axis vars: {vars_on_other_axis}")

            # Filter the original DataFrame based on the target axis non-aggregated parts
            filtered_orig_df = original_df.copy()
            for var, val in target_filters.items():
                 # Handle potential 'Missing' placeholder if used
                 if val == 'Missing':
                     filtered_orig_df = filtered_orig_df[filtered_orig_df[var].isna()]
                 else:
                     filtered_orig_df = filtered_orig_df[filtered_orig_df[var] == val]

            # Now calculate the totals across the other axis for this filtered data
            agg_series = _calculate_aggregate_totals(
                df=filtered_orig_df,
                group_vars=vars_on_other_axis, # Group by the variables on the other axis
                config=config,
                agg_label=agg_label # This label isn't really used in the output series index here
            )
            # logger.debug(f"    Calculated aggregate series (len {len(agg_series)}): \n{agg_series.head()}")


            # Assign the calculated series to the correct location in the result_df
            try:
                if axis == 0: # Assign series as a row
                    # Ensure index alignment (agg_series index should match result_df columns)
                    if not result_df.columns.equals(agg_series.index):
                        # Attempt reindexing if other_axis_obj exists
                        if other_axis_obj is not None and not other_axis_obj.empty:
                            logger.debug(f"Reindexing calculated agg series for row {target_tuple} to match columns {result_df.columns.names}")
                            agg_series = agg_series.reindex(result_df.columns, fill_value=np.nan)
                        else:
                             logger.warning(f"Column mismatch for row {target_tuple} and cannot reindex. Columns: {result_df.columns}, Series Index: {agg_series.index}")
                             # Skip assignment if indices don't match and reindexing failed/not possible
                             continue
                    result_df.loc[target_tuple] = agg_series.values # Assign values directly after alignment
                else: # axis == 1, assign series as a column
                    # Ensure index alignment (agg_series index should match result_df index)
                     if not result_df.index.equals(agg_series.index):
                        # Attempt reindexing if other_axis_obj exists
                        if other_axis_obj is not None and not other_axis_obj.empty:
                            logger.debug(f"Reindexing calculated agg series for column {target_tuple} to match index {result_df.index.names}")
                            agg_series = agg_series.reindex(result_df.index, fill_value=np.nan)
                        else:
                            logger.warning(f"Index mismatch for column {target_tuple} and cannot reindex. Index: {result_df.index}, Series Index: {agg_series.index}")
                            # Skip assignment if indices don't match and reindexing failed/not possible
                            continue
                     result_df[target_tuple] = agg_series # Assign as column

            except KeyError as e:
                # This might happen with unexpected index/column labels
                logger.error(f"KeyError assigning aggregate series for target={target_tuple}. This indicates an issue with index/column alignment. Error: {e}", exc_info=True)
                # Don't raise, just log, as NaNs are expected if data is missing
            except Exception as e:
                logger.error(f"Unexpected error assigning aggregate series for target={target_tuple}. Error: {e}", exc_info=True)
                raise CalculationError(f"Unexpected error assigning aggregate series for target={target_tuple}. Error: {e}") from e


        logger.info("Primary aggregate value calculation complete.")

        # 8. Sort the DataFrame
        logger.debug("Sorting the resulting DataFrame with hierarchical aggregates...")
        sort_axis_obj = result_df.index if axis == 0 else result_df.columns
        if isinstance(sort_axis_obj, pd.MultiIndex):
            # Create temporary columns for sorting keys
            temp_sort_cols = []
            df_to_sort = result_df.copy() # Work on a copy to add columns

            if axis == 0: # Sorting rows (index)
                df_to_sort = df_to_sort.reset_index()
                original_index_names = list(target_multiindex.names)
                for i, level_name in enumerate(original_index_names):
                    sort_col_name = f"_sort_key_{level_name}"
                    # Apply get_sort_key to the corresponding level column
                    target_series = df_to_sort[level_name]
                    logger.debug(f"Axis 0 Sorting: Applying get_sort_key to level '{level_name}' (Series head):\n{target_series.head()}")
                    try:
                        df_to_sort[sort_col_name] = target_series.apply(get_sort_key) # Pass function directly
                        logger.debug(f"Axis 0 Sorting: Applied get_sort_key successfully for level '{level_name}'.")
                    except Exception as e_apply:
                        logger.error(f"Axis 0 Sorting: Error applying get_sort_key for level '{level_name}': {e_apply}", exc_info=True)
                        # Add NaN column to allow sorting to proceed, highlighting the error location
                        df_to_sort[sort_col_name] = np.nan
                    temp_sort_cols.append(sort_col_name)
                # Sort by the key columns
                df_to_sort = df_to_sort.sort_values(by=temp_sort_cols)
                # Drop the temporary sort key columns
                df_to_sort = df_to_sort.drop(columns=temp_sort_cols)
                # Set the index back
                result_df_sorted = df_to_sort.set_index(original_index_names)
                # Ensure column order is preserved if columns were MultiIndex
                if isinstance(other_axis_obj, pd.MultiIndex):
                     result_df_sorted = result_df_sorted.reindex(columns=other_axis_obj)

            else: # axis == 1, sorting columns
                # Transpose, sort index, then transpose back
                df_to_sort = df_to_sort.T
                df_to_sort = df_to_sort.reset_index()
                original_column_names = list(target_multiindex.names)
                for i, level_name in enumerate(original_column_names):
                    sort_col_name = f"_sort_key_{level_name}"
                    target_series = df_to_sort[level_name]
                    logger.debug(f"Axis 1 Sorting: Applying get_sort_key to level '{level_name}' (Series head):\n{target_series.head()}")
                    try:
                        df_to_sort[sort_col_name] = target_series.apply(get_sort_key) # Pass function directly
                        logger.debug(f"Axis 1 Sorting: Applied get_sort_key successfully for level '{level_name}'.")
                    except Exception as e_apply:
                        logger.error(f"Axis 1 Sorting: Error applying get_sort_key for level '{level_name}': {e_apply}", exc_info=True)
                        # Add NaN column to allow sorting to proceed, highlighting the error location
                        df_to_sort[sort_col_name] = np.nan
                    temp_sort_cols.append(sort_col_name)
                df_to_sort = df_to_sort.sort_values(by=temp_sort_cols)
                # Drop the temporary sort key columns
                df_to_sort = df_to_sort.drop(columns=temp_sort_cols)
                df_to_sort = df_to_sort.set_index(original_column_names)
                result_df_sorted = df_to_sort.T
                # Ensure index order is preserved if index was MultiIndex
                if isinstance(other_axis_obj, pd.MultiIndex):
                    result_df_sorted = result_df_sorted.reindex(index=other_axis_obj)

            logger.debug("MultiIndex sorting complete.")
            return result_df_sorted
        else:
            # Simple index sorting (usually just place 'Aggregated' last)
            logger.debug("Simple index sorting applied (Aggregated typically last).")
            # The concat/assignment in step 2 usually handles this, but explicit sort can ensure it
            if axis == 0:
                # Sort index: Use get_sort_key
                sorted_index = result_df.index.to_series().apply(get_sort_key).sort_values().index
                return result_df.loc[sorted_index]
            else: # axis == 1
                # Sort columns: Use get_sort_key
                sorted_columns = result_df.columns.to_series().apply(get_sort_key).sort_values().index
                return result_df[sorted_columns]

    except CalculationError: # Re-raise CalculationErrors directly
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred during primary hierarchical aggregation on axis {axis}: {e}", exc_info=True)
        raise CalculationError(f"Failed to add primary hierarchical aggregates on axis {axis}. Error: {e}") from e


def _calculate_aggregate_totals(df: pd.DataFrame,
                                group_vars: List[str],
                                config: ConfigModel,
                                agg_label: str = 'Aggregated') -> pd.Series:
    """
    Calculates aggregated totals (sum or median, weighted or unweighted)
    for the specified grouping variables based on the primary statistic.

    Args:
        df: The input DataFrame (should be the original data).
        group_vars: List of columns to group by. If empty, calculates grand total.
        config: The validated configuration object.
        agg_label: The label to use for the aggregated result index/column (only used for grand total).

    Returns:
        A Pandas Series containing the aggregated results, indexed by group_vars
        (or a single value Series if group_vars is empty). Returns empty Series if no data.
    """
    if df.empty:
        logger.warning(f"Input DataFrame for calculating '{agg_label}' totals is empty.")
        # Return an empty series with appropriate index type if possible
        if group_vars:
            index = pd.MultiIndex(levels=[[]]*len(group_vars), codes=[[]]*len(group_vars), names=group_vars)
            return pd.Series([], index=index, dtype=float)
        else:
            # Provide data matching the index length (1) for the grand total case
            return pd.Series([np.nan], index=pd.Index(['GrandTotal']), dtype=float)

    if not group_vars: # Calculate grand total
        logger.debug(f"Calculating grand total ({config.statistic})...")
        # Apply wrapper to the whole DataFrame treated as one group
        total = _agg_wrapper(df, config.value_var, config.pweight, config.statistic)
        return pd.Series([total], index=pd.Index(['GrandTotal'])) # Use GrandTotal as name, let dtype be inferred
    else:
        logger.debug(f"Calculating primary '{agg_label}' totals for groups: {group_vars} ({config.statistic})...")
        # Check if group_vars exist in df
        missing_vars = [v for v in group_vars if v not in df.columns]
        if missing_vars:
            raise CalculationError(f"Grouping variables for aggregate totals not found in DataFrame: {missing_vars}")
        # Check value/weight vars exist
        if config.value_var not in df.columns:
             raise CalculationError(f"Value variable '{config.value_var}' not found in DataFrame for aggregate totals.")
        if config.pweight and config.pweight not in df.columns:
             raise CalculationError(f"Weight variable '{config.pweight}' not found in DataFrame for aggregate totals.")

        try:
            # Use dropna=False to include groups for NaN keys if they exist
            grouped = df.groupby(group_vars, observed=True, dropna=False)
            agg_series = grouped.apply(_agg_wrapper,
                                       value_col=config.value_var,
                                       weight_col=config.pweight,
                                       statistic=config.statistic,
                                       include_groups=False) # Avoid DeprecationWarning
            return agg_series
        except Exception as e:
            raise CalculationError(f"Failed to calculate primary aggregate totals for groups {group_vars}. Error: {e}") from e

def _fill_aggregate_intersections(
df: pd.DataFrame,
    df_orig: pd.DataFrame,
    config: ConfigModel,
    agg_label: str = "Aggregated"
) -> pd.DataFrame:
    """
    Fills NaN values at the intersections of aggregated rows and columns for the PRIMARY statistic table.

    Calculates the correct aggregate value for cells where both the row
    and column index contain the aggregation label (e.g., 'Aggregated'),
    based on filtering the original DataFrame.

    Args:
        df: The DataFrame potentially containing NaN intersections after
            applying _add_hierarchical_aggregates for both axes.
        df_orig: The original, unfiltered input DataFrame.
        config: The validated configuration object.
        agg_label: The label used to denote aggregated rows/columns.

    Returns:
        The DataFrame with NaN intersections filled.
    """
    logger.info("Filling NaN values at primary aggregate intersections...")
    df_filled = df.copy() # Work on a copy

    # Identify aggregated indices
    agg_rows = [idx for idx in df_filled.index if _contains_agg_label(idx, agg_label)]
    agg_cols = [col for col in df_filled.columns if _contains_agg_label(col, agg_label)]

    if not agg_rows or not agg_cols:
        logger.debug("No primary aggregate intersections found to fill.")
        return df_filled # Return early if no intersections possible

    # Iterate through intersections
    for agg_row in agg_rows:
        for agg_col in agg_cols:
            try:
                # Check if the cell is NaN using .at for scalar access
                if pd.isna(df_filled.at[agg_row, agg_col]):
                    # Determine filters based on non-agg parts
                    current_filter = pd.Series(True, index=df_orig.index)

                    # Row filters
                    if isinstance(agg_row, tuple):
                        for i, level_val in enumerate(agg_row):
                            if level_val != agg_label and i < len(config.row_var):
                                level_name = config.row_var[i]
                                # Handle 'Missing' placeholder if used
                                if level_val == 'Missing':
                                    current_filter &= df_orig[level_name].isna()
                                else:
                                    current_filter &= (df_orig[level_name] == level_val)
                    elif agg_row != agg_label and config.row_var: # Single level index
                         level_name = config.row_var[0]
                         if agg_row == 'Missing':
                             current_filter &= df_orig[level_name].isna()
                         else:
                             current_filter &= (df_orig[level_name] == agg_row)

                    # Column filters
                    if isinstance(agg_col, tuple):
                        for i, level_val in enumerate(agg_col):
                            if level_val != agg_label and i < len(config.col_var):
                                level_name = config.col_var[i]
                                # Handle 'Missing' placeholder if used
                                if level_val == 'Missing':
                                    current_filter &= df_orig[level_name].isna()
                                else:
                                    current_filter &= (df_orig[level_name] == level_val)
                    elif agg_col != agg_label and config.col_var: # Single level columns
                         level_name = config.col_var[0]
                         if agg_col == 'Missing':
                             current_filter &= df_orig[level_name].isna()
                         else:
                             current_filter &= (df_orig[level_name] == agg_col)

                    # Filter original data and calculate statistic using the existing wrapper
                    df_filtered = df_orig.loc[current_filter]
                    if df_filtered.empty:
                         calculated_value = np.nan # No data matches filters
                    else:
                         calculated_value = _agg_wrapper(
                             df_filtered, config.value_var, config.pweight, config.statistic
                         )
                    df_filled.at[agg_row, agg_col] = calculated_value # Fill the NaN

            except KeyError as e:
                 # This might happen if agg_row/agg_col somehow doesn't exist after checks
                 logger.warning(f"KeyError accessing primary intersection row={agg_row}, col={agg_col}. Skipping. Error: {e}")
            except Exception as e:
                 logger.error(f"Error processing primary intersection row={agg_row}, col={agg_col}: {e}", exc_info=True)
                 # Raising is safer to indicate a problem
                 raise CalculationError(f"Failed to fill primary aggregate intersection for row={agg_row}, col={agg_col}. Error: {e}") from e

    logger.info("Finished filling primary aggregate intersections.")
    return df_filled

# --- SDC Aggregation ---
# Map Indicator Names to Helper Functions (used in multiple places now)
SDC_INDICATOR_FUNCTIONS: Dict[str, Callable] = {
    'Count': _calculate_sdc_count,
    'Max': _calculate_sdc_max,
    '2nd Max': _calculate_sdc_second_max,
    'Sum of 3rd to nth': _calculate_sdc_sum_third_to_nth,
    'Total': _calculate_sdc_total,
    'Dominance (1,k) check': _calculate_sdc_dominance_1k,
    'Dominance p% check': _calculate_sdc_dominance_p_percent
}
SDC_INDICATOR_NAMES = list(SDC_INDICATOR_FUNCTIONS.keys())

def _apply_sdc_indicators_to_group(group: pd.DataFrame, value_col: str) -> pd.Series:
    """
    Calculates all SDC indicators for a given group based on the absolute value column.
    Handles empty or all-NaN groups by returning NaNs.
    (Adapted from calculator._calculate_indicators_for_group)

    Args:
        group: The DataFrame group from groupby().
        value_col: The name of the column containing the ABSOLUTE values to analyze.

    Returns:
        A Pandas Series containing the calculated indicators for the group.
    """
    # Work with the absolute values passed in value_col & Drop NaNs before calculation as per original SDC logic
    abs_values = group[value_col].dropna()
    if abs_values.empty:
        return pd.Series(np.nan, index=SDC_INDICATOR_NAMES) # Return a series of NaNs with the expected index

    # Call helper functions using the global map
    indicators = {name: func(abs_values) for name, func in SDC_INDICATOR_FUNCTIONS.items()}
    return pd.Series(indicators)

def _calculate_all_sdc_aggregates(
    df_orig: pd.DataFrame,
    config: ConfigModel,
    abs_value_col: str
) -> Dict[Tuple[Optional[Tuple[str,...]], Optional[Tuple[str,...]]], pd.DataFrame]:
    """
    Pre-calculates all SDC indicators for all required aggregation levels.

    Groups the original data by various combinations of row and column variables
    (including partial aggregations and grand total) and calculates SDC indicators
    for each group.

    Args:
        df_orig: The original, unpivoted input DataFrame.
        config: The validated configuration object.
        abs_value_col: The name of the column in df_orig containing absolute values.

    Returns:
        A dictionary where:
            - Keys are tuples: (row_grouping_vars, col_grouping_vars).
              An empty tuple () indicates aggregation over that dimension's variables.
              (tuple(), tuple()) represents the grand total.
            - Values are DataFrames containing the calculated SDC indicators for that
              aggregation level. The DataFrame index corresponds to the grouping variables,
              and columns are the SDC indicators.
    """
    logger.info("Pre-calculating all SDC aggregate indicators...")
    all_aggregates: Dict[Tuple[Tuple[str,...], Tuple[str,...]], pd.DataFrame] = {}
    row_vars = tuple(config.row_var)
    col_vars = tuple(config.col_var)

    # Generate all combinations of grouping keys - We need to group by subsets of row_vars AND subsets of col_vars simultaneously
    # 0. Base level (group by all row and col vars) - useful for structure
    grouping_keys_base = list(row_vars + col_vars)
    if grouping_keys_base:
        key_tuple = (row_vars, col_vars)
        logger.debug(f"Calculating SDC indicators for base grouping: {grouping_keys_base}")
        try:
            grouped = df_orig.groupby(grouping_keys_base, observed=True, dropna=False)
            result = grouped.apply(_apply_sdc_indicators_to_group, value_col=abs_value_col, include_groups=False) # Apply the SDC calculation function
            # Result might be Series (if single indicator) or DataFrame
            if isinstance(result, pd.Series) and not result.empty:
                if isinstance(result.index, pd.MultiIndex):
                    # Check if the last level is the indicator level before unstacking
                    if result.index.names[-1] is None or result.index.names[-1] == 'Indicator':
                        result = result.unstack() # Unstack the indicator level
                    else: # If not, convert to frame carefully
                        result = result.to_frame()
                else: # Simple index, likely single group result
                    result = result.to_frame(name=result.name or 0).T # Convert Series to DataFrame row
            if not result.empty:
                if isinstance(result.columns, pd.RangeIndex): result.columns = SDC_INDICATOR_NAMES # Ensure columns are named correctly
                # Replace NaN in index with 'Missing' BEFORE storing
                if isinstance(result.index, pd.MultiIndex):
                    result.index = _replace_nan_in_multiindex(result.index)
                elif pd.isna(result.index).any():
                    result.index = result.index.fillna('Missing')
                all_aggregates[key_tuple] = result.astype(float) # Ensure float type
        except Exception as e:
            logger.error(f"Error calculating base SDC indicators for groups {grouping_keys_base}: {e}", exc_info=True)
            all_aggregates[key_tuple] = pd.DataFrame(columns=SDC_INDICATOR_NAMES) # Store empty DF or handle error as needed

    # 1. Iterate through levels of row aggregation (keeping all col_vars)
    for i in range(len(row_vars)): # 0 to len(row_vars)-1
        row_group_vars = row_vars[:i] # Group by first i row vars
        grouping_keys = list(row_group_vars + col_vars)
        key_tuple = (row_group_vars, col_vars) # Key: (row_subset, all_cols)
        if not grouping_keys: continue # Skip if no grouping vars left
        logger.debug(f"Calculating SDC indicators for row agg level {i}: {grouping_keys}")
        try:
            grouped = df_orig.groupby(grouping_keys, observed=True, dropna=False)
            result = grouped.apply(_apply_sdc_indicators_to_group, value_col=abs_value_col, include_groups=False)
            if isinstance(result, pd.Series) and not result.empty:
                if isinstance(result.index, pd.MultiIndex):
                    if result.index.names[-1] is None or result.index.names[-1] == 'Indicator':
                        result = result.unstack()
                    else:
                        result = result.to_frame()
                else:
                    result = result.to_frame(name=result.name or 0).T
            if not result.empty:
                 if isinstance(result.columns, pd.RangeIndex): result.columns = SDC_INDICATOR_NAMES
                 # Replace NaN in index with 'Missing' BEFORE storing
                 if isinstance(result.index, pd.MultiIndex):
                     result.index = _replace_nan_in_multiindex(result.index)
                 elif pd.isna(result.index).any():
                     result.index = result.index.fillna('Missing')
                 all_aggregates[key_tuple] = result.astype(float)
        except Exception as e:
            logger.error(f"Error calculating SDC indicators for groups {grouping_keys}: {e}", exc_info=True)
            all_aggregates[key_tuple] = pd.DataFrame(columns=SDC_INDICATOR_NAMES)

    # 2. Iterate through levels of col aggregation (keeping all row_vars)
    for j in range(len(col_vars)): # 0 to len(col_vars)-1
        col_group_vars = col_vars[:j] # Group by first j col vars
        grouping_keys = list(row_vars + col_group_vars)
        key_tuple = (row_vars, col_group_vars) # Key: (all_rows, col_subset)
        if not grouping_keys: continue
        logger.debug(f"Calculating SDC indicators for col agg level {j}: {grouping_keys}")
        try:
            grouped = df_orig.groupby(grouping_keys, observed=True, dropna=False)
            result = grouped.apply(_apply_sdc_indicators_to_group, value_col=abs_value_col, include_groups=False)
            if isinstance(result, pd.Series) and not result.empty:
                if isinstance(result.index, pd.MultiIndex):
                    if result.index.names[-1] is None or result.index.names[-1] == 'Indicator':
                        result = result.unstack()
                    else:
                        result = result.to_frame()
                else:
                    result = result.to_frame(name=result.name or 0).T
            if not result.empty:
                if isinstance(result.columns, pd.RangeIndex): result.columns = SDC_INDICATOR_NAMES
                # Replace NaN in index with 'Missing' BEFORE storing
                if isinstance(result.index, pd.MultiIndex):
                    result.index = _replace_nan_in_multiindex(result.index)
                elif pd.isna(result.index).any():
                    result.index = result.index.fillna('Missing')
                all_aggregates[key_tuple] = result.astype(float)
        except Exception as e:
            logger.error(f"Error calculating SDC indicators for groups {grouping_keys}: {e}", exc_info=True)
            all_aggregates[key_tuple] = pd.DataFrame(columns=SDC_INDICATOR_NAMES)

    # 3. Iterate through levels of row AND col aggregation (intersections)
    for i in range(len(row_vars)):
        for j in range(len(col_vars)):
            row_group_vars = row_vars[:i]
            col_group_vars = col_vars[:j]
            grouping_keys = list(row_group_vars + col_group_vars)
            key_tuple = (row_group_vars, col_group_vars) # Key: (row_subset, col_subset)
            # Skip if grouping keys are empty (grand total handled later)
            if not grouping_keys: continue
            logger.debug(f"Calculating SDC indicators for intersection agg level R{i}/C{j}: {grouping_keys}")
            try:
                grouped = df_orig.groupby(grouping_keys, observed=True, dropna=False)
                result = grouped.apply(_apply_sdc_indicators_to_group, value_col=abs_value_col, include_groups=False)
                if isinstance(result, pd.Series) and not result.empty:
                    if isinstance(result.index, pd.MultiIndex):
                        if result.index.names[-1] is None or result.index.names[-1] == 'Indicator':
                             result = result.unstack()
                        else:
                             result = result.to_frame()
                    else:
                        result = result.to_frame(name=result.name or 0).T
                if not result.empty:
                    if isinstance(result.columns, pd.RangeIndex): result.columns = SDC_INDICATOR_NAMES
                    # Replace NaN in index with 'Missing' BEFORE storing
                    if isinstance(result.index, pd.MultiIndex):
                        result.index = _replace_nan_in_multiindex(result.index)
                    elif pd.isna(result.index).any():
                        result.index = result.index.fillna('Missing')
                    all_aggregates[key_tuple] = result.astype(float)
            except Exception as e:
                logger.error(f"Error calculating SDC indicators for groups {grouping_keys}: {e}", exc_info=True)
                all_aggregates[key_tuple] = pd.DataFrame(columns=SDC_INDICATOR_NAMES)

    # 4. Calculate Row Total Aggregates (group by all row_vars, aggregate cols)
    if row_vars:
        key_tuple_row_total = (row_vars, tuple()) # Key: (all_rows, empty_cols_tuple)
        logger.debug(f"Calculating SDC indicators for row totals: {list(row_vars)}")
        try:
            grouped = df_orig.groupby(list(row_vars), observed=True, dropna=False)
            result = grouped.apply(_apply_sdc_indicators_to_group, value_col=abs_value_col, include_groups=False)
            if isinstance(result, pd.Series) and not result.empty:
                if isinstance(result.index, pd.MultiIndex):
                    if result.index.names[-1] is None or result.index.names[-1] == 'Indicator':
                        result = result.unstack()
                    else:
                        result = result.to_frame()
                else:
                    result = result.to_frame(name=result.name or 0).T
            if not result.empty:
                if isinstance(result.columns, pd.RangeIndex): result.columns = SDC_INDICATOR_NAMES
                # Replace NaN in index with 'Missing' BEFORE storing
                if isinstance(result.index, pd.MultiIndex):
                    result.index = _replace_nan_in_multiindex(result.index)
                elif pd.isna(result.index).any():
                    result.index = result.index.fillna('Missing')
                all_aggregates[key_tuple_row_total] = result.astype(float)
        except Exception as e:
            logger.error(f"Error calculating SDC row total indicators: {e}", exc_info=True)
            all_aggregates[key_tuple_row_total] = pd.DataFrame(columns=SDC_INDICATOR_NAMES)

    # 5. Calculate Column Total Aggregates (group by all col_vars, aggregate rows)
    if col_vars:
        key_tuple_col_total = (tuple(), col_vars) # Key: (empty_rows_tuple, all_cols)
        logger.debug(f"Calculating SDC indicators for col totals: {list(col_vars)}")
        try:
            grouped = df_orig.groupby(list(col_vars), observed=True, dropna=False)
            result = grouped.apply(_apply_sdc_indicators_to_group, value_col=abs_value_col, include_groups=False)
            if isinstance(result, pd.Series) and not result.empty:
                if isinstance(result.index, pd.MultiIndex):
                    if result.index.names[-1] is None or result.index.names[-1] == 'Indicator':
                        result = result.unstack()
                    else:
                        result = result.to_frame()
                else:
                    result = result.to_frame(name=result.name or 0).T
            if not result.empty:
                if isinstance(result.columns, pd.RangeIndex): result.columns = SDC_INDICATOR_NAMES
                # Replace NaN in index with 'Missing' BEFORE storing
                if isinstance(result.index, pd.MultiIndex):
                    result.index = _replace_nan_in_multiindex(result.index)
                elif pd.isna(result.index).any():
                    result.index = result.index.fillna('Missing')
                all_aggregates[key_tuple_col_total] = result.astype(float)
        except Exception as e:
            logger.error(f"Error calculating SDC col total indicators: {e}", exc_info=True)
            all_aggregates[key_tuple_col_total] = pd.DataFrame(columns=SDC_INDICATOR_NAMES)

    # 6. Calculate Grand Total Aggregate (no grouping)
    key_tuple_grand_total = (tuple(), tuple()) # Key: (empty_rows_tuple, empty_cols_tuple)
    logger.debug("Calculating SDC indicators for grand total")
    try:
        result_series = _apply_sdc_indicators_to_group(df_orig, value_col=abs_value_col) # Apply directly to the whole DataFrame
        result_df = result_series.to_frame(name='GrandTotal').T # Convert Series to DataFrame and transpose to make indicators columns
        result_df.index.name = None # No index name for grand total row
        if not result_df.empty:
            if isinstance(result_df.columns, pd.RangeIndex): result_df.columns = SDC_INDICATOR_NAMES
            # Replace NaN in index with 'Missing' BEFORE storing (though unlikely for grand total)
            if isinstance(result_df.index, pd.MultiIndex):
                 result_df.index = _replace_nan_in_multiindex(result_df.index)
            elif pd.isna(result_df.index).any():
                 result_df.index = result_df.index.fillna('Missing')
            all_aggregates[key_tuple_grand_total] = result_df.astype(float)
    except Exception as e:
        logger.error(f"Error calculating SDC grand total indicators: {e}", exc_info=True)
        all_aggregates[key_tuple_grand_total] = pd.DataFrame(columns=SDC_INDICATOR_NAMES)
    logger.info(f"Pre-calculation of SDC aggregates complete. Found {len(all_aggregates)} aggregation levels.")
    return all_aggregates

def _add_sdc_hierarchical_aggregates(
    sdc_df: pd.DataFrame, # Base SDC table (rows: row_vars+Indicator, cols: col_vars)
    config: ConfigModel,
    axis: int, # 0 for rows (index), 1 for columns
    precalculated_aggregates: Dict[Tuple[Tuple[str,...], Tuple[str,...]], pd.DataFrame],
    agg_label: str = 'Aggregated'
) -> pd.DataFrame:
    """
    Adds hierarchical 'Aggregated' entries (rows or columns) to the SDC indicator table
    using pre-calculated aggregate values.

    Args:
        sdc_df: The base SDC indicator DataFrame (index: row_vars + Indicator, columns: col_vars).
        config: The validated configuration object.
        axis: 0 to add aggregated rows (index), 1 to add aggregated columns.
        precalculated_aggregates: Dictionary containing pre-calculated SDC indicators
                                   for various aggregation levels. Keys are (row_group_vars, col_group_vars).
        agg_label: Label to use for the aggregated entries.

    Returns:
        A new DataFrame with hierarchical aggregates added and sorted.

    Raises:
        CalculationError: If errors occur during processing.
    """
    axis_name = "rows" if axis == 0 else "columns"
    logger.info(f"Adding hierarchical 'Aggregated' {axis_name} to SDC table using precalculated values...")

    # --- Axis-dependent setup ---
    row_vars_tuple = tuple(config.row_var)
    col_vars_tuple = tuple(config.col_var)
    indicator_level_name = 'Indicator' # Expected name

    # Aggregating Rows (Index)
    if axis == 0: 
        if not config.row_var:
            logger.debug("No row_var specified, skipping SDC row aggregation.")
            return sdc_df
        if sdc_df.empty:
            logger.warning("SDC DataFrame is empty, cannot add row aggregates.")
            return sdc_df
        target_axis_obj = sdc_df.index
        other_axis_obj = sdc_df.columns
        vars_on_axis = config.row_var # List of names
        vars_on_other_axis = config.col_var # List of names
        # Row aggregation requires MultiIndex with Indicator level
        if not isinstance(target_axis_obj, pd.MultiIndex):
            logger.warning("SDC table has simple index. Cannot add hierarchical row aggregates.")
            return sdc_df
        # Check if Indicator level exists and find its position
        if indicator_level_name not in target_axis_obj.names:
            logger.error(f"Cannot find '{indicator_level_name}' level in index {target_axis_obj.names}. Aborting row aggregation.")
            return sdc_df
        indicator_level_idx = target_axis_obj.names.index(indicator_level_name)
        if target_axis_obj.nlevels != len(vars_on_axis) + 1:
            logger.warning(f"Expected {len(vars_on_axis) + 1} index levels (row_vars + Indicator), found {target_axis_obj.nlevels}. Row aggregation may be incorrect.")
    # Aggregating Columns
    elif axis == 1:
        if not config.col_var:
            logger.debug("No col_var specified, skipping SDC column aggregation.")
            return sdc_df
        # Allow aggregation even if columns are empty initially (might add grand total col)
        target_axis_obj = sdc_df.columns
        other_axis_obj = sdc_df.index # Note: Index includes Indicator level here
        vars_on_axis = config.col_var # List of names
        vars_on_other_axis = config.row_var # List of names
        # Check if Indicator level exists in the index (other_axis_obj)
        if not isinstance(other_axis_obj, pd.MultiIndex) or indicator_level_name not in other_axis_obj.names:
             logger.warning(f"Cannot add column aggregates: Index is not MultiIndex or '{indicator_level_name}' level is missing.")
             return sdc_df
        indicator_level_idx = other_axis_obj.names.index(indicator_level_name)
    else:
        raise ValueError("Axis must be 0 (rows) or 1 (columns)")

    try: # --- Generate Target Tuples (Index or Columns) ---
        all_tuples = set()
        n_levels = 0
        indicator_values = [] # Only used for axis=0

        if isinstance(target_axis_obj, pd.MultiIndex):
            n_levels = target_axis_obj.nlevels
            all_tuples.update(target_axis_obj) # Add existing tuples
            # Row Aggregation - Special handling for Indicator
            if axis == 0: 
                row_var_levels = len(vars_on_axis)
                indicator_values = target_axis_obj.get_level_values(indicator_level_idx).unique()
                logger.debug(f"Found indicators for row aggregation: {list(indicator_values)}")

                # Extract existing prefixes (excluding indicator)
                existing_prefixes = set(
                    tuple(idx_tuple[i] for i in range(n_levels) if i != indicator_level_idx)
                    for idx_tuple in target_axis_obj
                )
                logger.debug(f"Found {len(existing_prefixes)} unique row prefixes.")

                # Create aggregated tuples for each level, adding indicator back
                for level_idx in range(row_var_levels + 1): # Iterate 0 to row_var_levels
                    unique_prefixes_at_level = set(prefix[:level_idx] for prefix in existing_prefixes) # Prefixes of length level_idx
                    for prefix in unique_prefixes_at_level:
                        for indicator in indicator_values:
                            agg_part = (agg_label,) * (row_var_levels - level_idx) # Agg tuple: prefix + ('Aggregated',) * (remaining row levels) + (indicator,)
                            # Construct the full tuple, inserting the indicator at the correct position
                            full_agg_tuple_list = list(prefix + agg_part)
                            full_agg_tuple_list.insert(indicator_level_idx, indicator)
                            all_tuples.add(tuple(full_agg_tuple_list))
            # axis == 1 (Column Aggregation) - Simpler tuple generation
            else: 
                for level_idx in range(n_levels + 1): # Iterate 0 to n_levels
                    unique_prefixes = set(t[:level_idx] for t in target_axis_obj)
                    for prefix in unique_prefixes:
                        agg_tuple = prefix + (agg_label,) * (n_levels - level_idx)
                        all_tuples.add(agg_tuple)
        # Simple Index/Columns on target axis
        else: # This case is less likely for SDC tables but handle defensively
             if target_axis_obj.empty:
                n_levels = len(vars_on_axis) # Use expected number of levels
                if axis == 0: n_levels += 1 # Add indicator level for rows
             else:
                n_levels = 1
                all_tuples.update((c,) for c in target_axis_obj) # Wrap in tuple

             # Add the overall aggregate tuple
             overall_agg_tuple = (agg_label,) * n_levels
             all_tuples.add(overall_agg_tuple)

             if axis == 0: # Simple index on rows - cannot do hierarchical aggregation if indicator isn't the index
                if target_axis_obj.name != indicator_level_name:
                    logger.warning("SDC table has simple index that is not 'Indicator'. Cannot add hierarchical row aggregates.")
                    return sdc_df
        logger.debug(f"Total unique target tuples including aggregates: {len(all_tuples)}")

        # --- Create Target Structure (Index or Columns) ---
        target_names = list(target_axis_obj.names) # Use list for potential modification
        # Ensure target_names has the correct length, especially if target_axis_obj was empty
        expected_n_levels = len(vars_on_axis)
        if axis == 0: expected_n_levels +=1 # Add indicator level
        if len(target_names) != expected_n_levels:
             logger.warning(f"Target axis names length mismatch. Expected {expected_n_levels}, got {len(target_names)}. Using default names.")
             # Generate default names based on expected structure
             if axis == 0:
                target_names = list(config.row_var)
                target_names.insert(indicator_level_idx, indicator_level_name)
             else: # axis == 1
                target_names = list(config.col_var)

        # Sort the tuples before creating the MultiIndex for consistent order
        if n_levels > 0 and all_tuples:
            if axis == 0: # Rows - Use custom sort key including indicator
                def sdc_row_sort_key(tup):
                    key = []
                    indicator_val = None
                    for i, val in enumerate(tup):
                        if i == indicator_level_idx:
                            indicator_val = val
                        else:
                            key.append(get_sort_key(val))
                    return tuple(key) + (indicator_val,) # Append indicator value at the end for secondary sort within hierarchy
                all_tuples_list = sorted(list(all_tuples), key=sdc_row_sort_key)
            else: # Columns - Standard hierarchical sort key is sufficient
                all_tuples_list = sorted(list(all_tuples), key=lambda tup: tuple(get_sort_key(val) for val in tup))
            target_multiindex = pd.MultiIndex.from_tuples(all_tuples_list, names=target_names)
        else: # Handle case with no levels or no tuples
            target_multiindex = pd.MultiIndex(levels=[[]]*expected_n_levels,codes=[[]]*expected_n_levels,names=target_names)

        # --- Initialize Result DataFrame ---
        if axis == 0: # New rows
            result_df = pd.DataFrame(np.nan, index=target_multiindex, columns=other_axis_obj)
        else: # New columns
            # Ensure other_axis_obj (index) is valid before creating DataFrame
            if other_axis_obj is None or other_axis_obj.empty:
                logger.warning("Cannot initialize result DataFrame for column aggregation: Index is empty.")
                result_df = pd.DataFrame(columns=target_multiindex) # Create empty DF with correct columns if possible
            else:
                result_df = pd.DataFrame(np.nan, index=other_axis_obj, columns=target_multiindex)
        logger.debug(f"Initialized result DataFrame shape: {result_df.shape}")

        # --- Populate Original Values ---
        logger.debug("Populating result DataFrame with original SDC values...")
        if not sdc_df.empty and not result_df.empty:
            # Directly copy data from sdc_df to result_df, handling potential
            # differences in column types (simple vs tuple) caused by aggregation.
            # This avoids issues with reindex/update losing data.
            logger.debug("Directly copying original SDC values into result DataFrame...")
            try:
                # Iterate through the columns of the original sdc_df
                for col in sdc_df.columns:
                    # Determine the corresponding column name in result_df
                    # If axis=1 (column aggregation), result_df columns are tuples
                    # If axis=0 (row aggregation), result_df columns match sdc_df columns (which might already be tuples)
                    target_col = (col,) if axis == 1 and not isinstance(col, tuple) else col

                    # Check if the target column exists in result_df
                    if target_col in result_df.columns:
                        # Assign the data column by column
                        # This preserves data even if column names differ slightly (e.g., 2020 vs (2020,))
                        result_df[target_col] = sdc_df[col]
                    else:
                        logger.warning(f"Target column '{target_col}' not found in result_df columns while copying original values. Skipping column '{col}'.")

                # Verify the specific value after copy
                row_key = ('A', 'Count')
                target_col_key = (2020,) # Expected column format in result_df
                if row_key in result_df.index and target_col_key in result_df.columns:
                    copied_value = result_df.loc[row_key, target_col_key]
                    logger.debug(f"DEBUG: Value in result_df for ({row_key}, {target_col_key}) after direct copy: {copied_value}")
                else:
                    logger.debug(f"DEBUG: Key ({row_key}, {target_col_key}) not found in result_df after direct copy.")

            except Exception as copy_err:
                 logger.error(f"Error during direct copy of sdc_df to result_df: {copy_err}", exc_info=True)
                 # If copy fails, result_df might remain partially filled or with NaNs

        # --- Populate Aggregates using Precalculated Values ---
        # Ensure target_axis_obj is iterable even if simple index
        original_tuples = set(target_axis_obj) if isinstance(target_axis_obj, pd.MultiIndex) else set((t,) for t in target_axis_obj)
        new_agg_tuples = all_tuples - original_tuples
        logger.info(f"Populating {len(new_agg_tuples)} new aggregate {axis_name} using precalculated values...")

        for agg_tuple in new_agg_tuples:
            # Now iterate through the *other* axis to assign values
            for other_label in other_axis_obj:
                # --- Determine grouping keys and lookup_key INSIDE the inner loop ---
                row_group_key_list = []
                col_group_key_list = []
                indicator_name_from_agg = None # Specific indicator for this agg_tuple (if axis=0)

                # 1. Determine row_group_key based on agg_tuple (target axis)
                if axis == 0: # agg_tuple is row index
                    current_row_vars_indices = []
                    for i, level_val in enumerate(agg_tuple):
                        if i != indicator_level_idx and level_val != agg_label:
                            current_row_vars_indices.append(i)
                        if i == indicator_level_idx:
                            indicator_name_from_agg = level_val # Store indicator name
                    original_row_var_index = 0
                    for i in range(len(agg_tuple)):
                        if i == indicator_level_idx: continue
                        if i in current_row_vars_indices:
                            if original_row_var_index < len(config.row_var):
                                row_group_key_list.append(config.row_var[original_row_var_index])
                        if original_row_var_index < len(config.row_var):
                            original_row_var_index += 1
                else: # agg_tuple is column index, so keep all row vars for row_group_key
                    row_group_key_list = list(row_vars_tuple)

                # 2. Determine col_group_key based on other_label (other axis) if axis=0, or agg_tuple if axis=1
                if axis == 0: # other_label is column index
                    other_label_tuple = other_label if isinstance(other_label, tuple) else (other_label,)
                    current_col_vars_indices = []
                    for i, level_val in enumerate(other_label_tuple):
                        if level_val != agg_label and i < len(col_vars_tuple):
                            current_col_vars_indices.append(i)
                    col_group_key_list = [config.col_var[idx] for idx in current_col_vars_indices]
                else: # agg_tuple is column index
                    current_col_vars_indices = []
                    for i, level_val in enumerate(agg_tuple):
                         if level_val != agg_label and i < len(col_vars_tuple):
                              current_col_vars_indices.append(i)
                    col_group_key_list = [config.col_var[idx] for idx in current_col_vars_indices]

                # 3. Construct the lookup key for this specific intersection
                lookup_key = (tuple(row_group_key_list), tuple(col_group_key_list))

                # 4. Retrieve the corresponding precalculated data
                if lookup_key not in precalculated_aggregates:
                    continue # Skip this specific cell

                agg_data_df = precalculated_aggregates[lookup_key]
                if agg_data_df.empty:
                    continue # Skip this specific cell
                # --- DEBUG LOGGING START ---
                key_exists = lookup_key in precalculated_aggregates
                if not key_exists:
                    continue # Skip if key not found
                # --- DEBUG LOGGING END ---
                try:
                    # Determine the specific value needed from agg_data_df
                    indicator_name = None
                    lookup_index_val = None # Value(s) to use for index lookup in agg_data_df

                    if axis == 0: # Target=Rows, Other=Columns (other_label is a column label)
                        indicator_name = indicator_name_from_agg # From the row agg_tuple
                        lookup_index_val = other_label # Column label identifies the group in agg_data_df index
                    else: # Target=Columns, Other=Rows (other_label is a row index label)
                        # Extract indicator and row parts from other_label
                        row_parts_for_lookup = []
                        if isinstance(other_label, tuple):
                            indicator_name = other_label[indicator_level_idx]
                            # Extract row parts (excluding indicator)
                            row_parts_for_lookup = [other_label[i] for i in range(len(other_label)) if i != indicator_level_idx]
                        else: # Simple index (should be just indicator)
                            indicator_name = other_label # No row parts if index is simple
                        
                        # Extract non-aggregated column parts from agg_tuple
                        col_parts_for_lookup = [val for i, val in enumerate(agg_tuple) if val != agg_label and i < len(col_vars_tuple)]

                        # Combine row and column parts based on the lookup_key structure
                        # lookup_key = (row_group_key, col_group_key). The index of agg_data_df is row_group_key + col_group_key
                        lookup_index_val_list = row_parts_for_lookup + col_parts_for_lookup

                        # Convert to tuple, handle empty case (grand total)
                        if lookup_index_val_list:
                             lookup_index_val = tuple(lookup_index_val_list)
                        else:
                            # This case corresponds to the grand total lookup key (tuple(), tuple()) or cases where either row
                            # or col parts are fully aggregated. 
                            lookup_index_val = None # Let the lookup logic handle None index value for grand total

                    if indicator_name not in SDC_INDICATOR_NAMES:
                        continue

                    # Select the value from the precalculated DataFrame
                    agg_value = np.nan # Default to NaN

                    # --- Determine Lookup Index Value ---
                    # The index value depends on the structure of the precalculated agg_data_df's index, which is determined by the
                    # lookup_key = (row_group_key, col_group_key). The index columns are row_group_key + col_group_key.
                    expected_index_vars = lookup_key[0] + lookup_key[1]
                    lookup_index_val = None # Default

                    if not expected_index_vars:
                         # Grand Total case (lookup_key = ((), ())) - index is usually single dummy value
                         if not agg_data_df.index.empty:
                              lookup_index_val = agg_data_df.index[0] # Use the actual index value
                         else:
                              lookup_index_val = float('nan') # Cannot lookup in empty DF
                    else:
                        # Build the index value from agg_tuple (row/target axis) and other_label (column/other axis)
                        lookup_filters = {}
                        valid_lookup = True

                        # 1. Get values from agg_tuple (target axis) corresponding to row_group_key (if axis=0) or col_group_key (if axis=1)
                        target_parts = {}
                        current_target_var_idx = 0
                        target_key_group = lookup_key[0] if axis == 0 else lookup_key[1] # row_group_key or col_group_key

                        for i in range(len(agg_tuple)):
                            if axis == 0 and i == indicator_level_idx: continue # Skip indicator on row axis
                            if current_target_var_idx < len(target_key_group):
                                var_name = target_key_group[current_target_var_idx]
                                if agg_tuple[i] != agg_label:
                                    target_parts[var_name] = agg_tuple[i]
                                # If agg_tuple[i] IS agg_label, we don't add it to the parts needed for lookup
                                current_target_var_idx += 1

                        # 2. Get values from other_label (other axis) corresponding to col_group_key (if axis=0) or row_group_key (if axis=1)
                        other_parts = {}
                        other_label_tuple = other_label if isinstance(other_label, tuple) else (other_label,)
                        current_other_var_idx = 0
                        other_key_group = lookup_key[1] if axis == 0 else lookup_key[0] # col_group_key or row_group_key

                        for i in range(len(other_label_tuple)):
                            if axis == 1 and i == indicator_level_idx: continue # Skip indicator on row axis (other_label)
                            if current_other_var_idx < len(other_key_group):
                                var_name = other_key_group[current_other_var_idx]
                                # We assume other_label never contains agg_label, it's a specific slice label
                                other_parts[var_name] = other_label_tuple[i]
                                current_other_var_idx += 1

                        # 3. Combine parts based on expected_index_vars order
                        lookup_index_val_list = []
                        try:
                            for var in expected_index_vars:
                                if var in target_parts: # Check target axis parts first
                                    lookup_index_val_list.append(target_parts[var])
                                elif var in other_parts: # Then check other axis parts
                                    lookup_index_val_list.append(other_parts[var])
                                else: # This variable was aggregated out on both axes, should not be in expected_index_vars
                                    valid_lookup = False; break
                        except Exception as e:
                            valid_lookup = False

                        # 4. Finalize lookup_index_val
                        if not valid_lookup:
                            lookup_index_val = float('nan') # Indicate error state
                        elif lookup_index_val_list:
                            lookup_index_val = tuple(lookup_index_val_list) if len(lookup_index_val_list) > 1 else lookup_index_val_list[0] # Use tuple for MultiIndex, scalar otherwise
                        else: # If reached here, it's an error state.
                            lookup_index_val = float('nan') # Indicate error state
                    # --- End Lookup Index Value Determination ---
                    try:
                        # Handle grand total case first (index is irrelevant)
                        is_grand_total_lookup = lookup_key == (tuple(), tuple())
                        if is_grand_total_lookup:
                            if indicator_name in agg_data_df.columns:
                                agg_value = agg_data_df.iloc[0][indicator_name]
                        # Handle cases where lookup_index_val determines the row
                        elif lookup_index_val is not None:
                            # Need to handle both simple and multi-index lookup
                            if isinstance(agg_data_df.index, pd.MultiIndex):
                                # Ensure lookup_index_val is a tuple of the correct length
                                if not isinstance(lookup_index_val, tuple):
                                    lookup_index_val = (lookup_index_val,)
                                # Check if tuple exists in index before loc
                                if len(lookup_index_val) == agg_data_df.index.nlevels and agg_data_df.index.isin([lookup_index_val]).any():
                                    agg_value = agg_data_df.loc[lookup_index_val, indicator_name]
                            else: # Simple index
                                # Ensure lookup_index_val is not a tuple for simple index lookup
                                lookup_scalar = lookup_index_val[0] if isinstance(lookup_index_val, tuple) and len(lookup_index_val)==1 else lookup_index_val
                                if not isinstance(lookup_scalar, tuple) and lookup_scalar in agg_data_df.index:
                                    agg_value = agg_data_df.loc[lookup_scalar, indicator_name]

                        # Handle cases where the agg_data_df might only have one row (e.g., total for a single group)
                        elif lookup_index_val is None and len(agg_data_df) == 1:
                            if indicator_name in agg_data_df.columns:
                                agg_value = agg_data_df.iloc[0][indicator_name]

                    except KeyError: # This is expected if the specific combination didn't exist in original data
                        
                        logger.debug(f"    Lookup KeyError for index {lookup_index_val}, indicator {indicator_name}. Assigning NaN.")
                        pass # Keep default NaN
                    except Exception as lookup_err:
                        logger.warning(f"    Error during lookup index {lookup_index_val} for indicator {indicator_name}: {lookup_err}. Assigning NaN.")
                        pass # Keep default NaN

                    #logger.debug(f"    Lookup Result: agg_value={agg_value} (type: {type(agg_value)}) for lookup_key={lookup_key}, indicator={indicator_name}, lookup_index_val={lookup_index_val}")

                    # Assign the value using .at for efficiency
                    try:
                        if axis == 0: # Assign to row=agg_tuple, col=other_label
                            #logger.debug(f"    Assigning (Axis 0): result_df.at[{agg_tuple}, {other_label}] = {agg_value}")
                            result_df.at[agg_tuple, other_label] = agg_value
                        else: # Assign to row=other_label, col=agg_tuple
                            #logger.debug(f"    Assigning (Axis 1): result_df.at[{other_label}, {agg_tuple}] = {agg_value}")
                            result_df.at[other_label, agg_tuple] = agg_value
                    except ValueError as ve:
                         logger.error(f"    ValueError during assignment: {ve}. agg_value type: {type(agg_value)}", exc_info=True)
                         # Attempt to assign NaN if assignment failed
                         try:
                             if axis == 0: result_df.at[agg_tuple, other_label] = np.nan
                             else: result_df.at[other_label, agg_tuple] = np.nan
                         except Exception: pass
                    except Exception as assign_err:
                         logger.error(f"    Unexpected error during assignment: {assign_err}", exc_info=True)
                         # Attempt to assign NaN if assignment failed
                         try:
                             if axis == 0: result_df.at[agg_tuple, other_label] = np.nan
                             else: result_df.at[other_label, agg_tuple] = np.nan
                         except Exception: pass

                except KeyError as e:
                    # Error during assignment itself (less likely) - This except corresponds to the outer try. Attempt to assign NaN defensively
                    try:
                        if axis == 0: result_df.at[agg_tuple, other_label] = np.nan
                        else: result_df.at[other_label, agg_tuple] = np.nan
                    except Exception: pass # Ignore error during defensive assignment
                except Exception as e:
                    # This except corresponds to the outer try. Attempt to assign NaN defensively
                    try:
                        if axis == 0: result_df.at[agg_tuple, other_label] = np.nan
                        else: result_df.at[other_label, agg_tuple] = np.nan
                    except Exception: pass

        logger.info(f"SDC aggregate {axis_name} population complete.")

        # --- Sort the Aggregated Axis ---
        # Use the already sorted target_multiindex created earlier
        if axis == 0:
            final_df = result_df.reindex(index=target_multiindex) # Reindex the DataFrame to match the sorted index
            if isinstance(other_axis_obj, pd.MultiIndex): # Ensure column order is preserved if columns were MultiIndex
                final_df = final_df.reindex(columns=other_axis_obj) # Preserve original column order
        else: # axis == 1
            final_df = result_df.reindex(columns=target_multiindex) # Reindex the DataFrame columns to match the sorted columns
            final_df = final_df.reindex(index=other_axis_obj) # Ensure index order is preserved (original rows)
        return final_df

    except CalculationError: # Re-raise CalculationErrors directly
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred during SDC {axis_name} aggregation: {e}", exc_info=True)
        raise CalculationError(f"Failed to add SDC aggregated {axis_name}. Error: {e}") from e


def _contains_agg_label(label: Union[str, Tuple], agg_label: str) -> bool:
    """Helper to check if a label (str or tuple) contains the agg_label."""
    if isinstance(label, tuple):
        return agg_label in label
    else:
        return label == agg_label # Also handle potential NaN or None comparison safely
