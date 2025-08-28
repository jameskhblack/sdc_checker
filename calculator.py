"""Core calculation logic: pivoting, aggregation, SDC indicators."""

import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Callable, Optional, Tuple, Set # Added Tuple, Set

# Assuming ConfigModel and exceptions are importable from sibling modules
from .config import ConfigModel
from .exceptions import CalculationError
from .stats import _calculate_sdc_count, _calculate_sdc_max, _calculate_sdc_second_max, _calculate_sdc_sum_third_to_nth, _calculate_sdc_total, _calculate_sdc_dominance_1k, _calculate_sdc_dominance_p_percent
# Refactored imports from totals
from .totals import (
    _add_hierarchical_aggregates, _fill_aggregate_intersections, # For primary stat
    _add_sdc_hierarchical_aggregates, _calculate_all_sdc_aggregates, # Refactored SDC aggs
    _agg_wrapper
)


# Get the logger instance configured in utils
logger = logging.getLogger("sdc_checker.calculator")
from .utils import get_sort_key # For hierarchical sorting

# --- Primary Statistic Calculation ---

def calculate_primary_statistic(df: pd.DataFrame, config: ConfigModel) -> pd.DataFrame:
    """
    Generates the primary pivot table based on the configuration.

    Uses groupby().apply().unstack() for robust weighted/unweighted aggregation.
    Optionally includes 'Aggregated' totals.

    Args:
        df: The input DataFrame (loaded by data_loader).
        config: The validated configuration object.

    Returns:
        A Pandas DataFrame representing the pivot table with the primary statistic.

    Raises:
        CalculationError: If errors occur during pivoting or aggregation.
    """
    logger.info(f"Calculating primary statistic '{config.statistic}'...")
    logger.debug(f"  Row vars: {config.row_var}")
    logger.debug(f"  Col vars: {config.col_var}")
    logger.debug(f"  Value var: {config.value_var}")
    logger.debug(f"  Weight var: {config.pweight}")
    logger.debug(f"  Include Aggregated: {config.include_aggregated}")

    try:
        # Pre-aggregate if secondary_ref is specified (mirrors calculate_sdc_indicators)
        df = _pre_aggregate_secondary_ref(df.copy(), config)
        # --- Main Pivot Calculation using Groupby + Apply + Unstack ---
        group_by_vars = config.row_var + config.col_var
        logger.debug(f"Grouping by: {group_by_vars}")

        # Ensure grouping vars exist
        missing_group_vars = [v for v in group_by_vars if v not in df.columns]
        if missing_group_vars:
            raise CalculationError(f"Grouping variables not found in DataFrame: {missing_group_vars}")
        # Ensure value/weight vars exist
        if config.value_var not in df.columns:
            raise CalculationError(f"Value variable '{config.value_var}' not found in DataFrame.")
        if config.pweight and config.pweight not in df.columns:
            raise CalculationError(f"Weight variable '{config.pweight}' not found in DataFrame.")


        grouped = df.groupby(group_by_vars, observed=True, dropna=False)
        # Apply the aggregation wrapper
        result_series = grouped.apply(_agg_wrapper,
                                      value_col=config.value_var,
                                      weight_col=config.pweight,
                                      statistic=config.statistic,
                                      include_groups=False) # Avoid DeprecationWarning

        # Handle cases where the result series might be empty or all NaN
        if result_series.empty:
            logger.warning("Resulting series after grouping and aggregation is empty. Creating empty pivot table.")
            # Create an empty DataFrame with the expected index/columns structure
            index = pd.MultiIndex(levels=[[]]*len(config.row_var), codes=[[]]*len(config.row_var), names=config.row_var)
            columns = pd.MultiIndex(levels=[[]]*len(config.col_var), codes=[[]]*len(config.col_var), names=config.col_var)
            pivot_table = pd.DataFrame(index=index, columns=columns, dtype=float)

        else:
            # Unstack the column variables to create the pivot table structure
            if not config.col_var: # Only row vars specified
                pivot_table = result_series.to_frame(name=config.value_var) # Result is a Series, convert to Frame
            else:
                # Unstack requires the levels to be part of the index
                try:
                    pivot_table = result_series.unstack(level=config.col_var)
                except KeyError as e:
                    logger.error(f"Error during unstacking. Level(s) '{config.col_var}' might not be in the index: {result_series.index}. Error: {e}", exc_info=True)
                    raise CalculationError(f"Failed to unstack columns '{config.col_var}'. Check data for missing combinations. Original error: {e}") from e
                except ValueError as e:
                    # Check if it's the duplicate entry error
                    if "Index contains duplicate entries, cannot reshape" in str(e):
                        logger.error(f"Duplicate index entries found before unstacking: {result_series.index[result_series.index.duplicated()]}")
                        raise CalculationError(f"Failed to unstack columns '{config.col_var}' due to duplicate index entries. Check input data for unexpected combinations. Original error: {e}") from e
                    else:
                        logger.error(f"Error during unstacking: {e}. Index: {result_series.index}", exc_info=True)
                        raise CalculationError(f"Failed to unstack columns '{config.col_var}' due to reshaping issue. Original error: {e}") from e


        logger.debug(f"Base pivot table shape: {pivot_table.shape}")
        logger.debug(f"Base pivot table index:\n{pivot_table.index}")
        logger.debug(f"Base pivot table columns:\n{pivot_table.columns}")

        # --- Add Aggregated Totals (if requested) ---
        if config.include_aggregated:
            logger.info("Calculating and adding hierarchical 'Aggregated' totals...") # Keep this line
            agg_label = 'Aggregated' # Keep this line
            # Start with the base pivot table (before any aggregation was added)
            # Ensure 'pivot_table' variable holds the result of the initial unstacking
            pivot_table_with_agg = pivot_table.copy()

            # Add row aggregates (if row vars exist)
            if config.row_var:
                logger.debug("Calling _add_hierarchical_aggregates for rows (axis=0)...")
                pivot_table_with_agg = _add_hierarchical_aggregates( # This is the primary stat agg func
                    base_df=pivot_table_with_agg,
                    original_df=df, # Pass the potentially pre-aggregated DataFrame
                    config=config,
                    axis=0,
                    vars_on_axis=config.row_var,
                    vars_on_other_axis=config.col_var,
                    agg_label=agg_label
                )
                logger.debug("Returned from _add_hierarchical_aggregates for rows.")
                # Optional: Add log of the table shape/index after row aggregation if needed

            # Add column aggregates (if col vars exist)
            if config.col_var:
                logger.debug("Calling _add_hierarchical_aggregates for columns (axis=1)...")
                pivot_table_with_agg = _add_hierarchical_aggregates( # This is the primary stat agg func
                    base_df=pivot_table_with_agg,
                    original_df=df, # Pass the potentially pre-aggregated DataFrame
                    config=config,
                    axis=1,
                    vars_on_axis=config.col_var,
                    vars_on_other_axis=config.row_var, # Note: row_var is now on the other axis
                    agg_label=agg_label
                )
                logger.debug("Returned from _add_hierarchical_aggregates for columns.")
                # Optional: Add log of the table shape/columns after column aggregation if needed

            # Fill NaN values at the intersections of newly added aggregates
            # This uses the primary stat intersection filler
            logger.debug("Calling _fill_aggregate_intersections...")
            pivot_table_with_agg = _fill_aggregate_intersections(
                df=pivot_table_with_agg,
                df_orig=df, # Pass potentially pre-aggregated df
                config=config,
                agg_label=agg_label
            )
            logger.debug("Returned from _fill_aggregate_intersections.")


            # Assign the final result back to the main variable
            pivot_table = pivot_table_with_agg


        logger.info(f"Primary statistic calculation complete. Pivot table shape: {pivot_table.shape}")
        logger.debug(f"Final pivot table returned:\n{pivot_table}")
        return pivot_table

    except CalculationError: # Re-raise CalculationErrors directly
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred during primary statistic calculation: {e}", exc_info=True)
        raise CalculationError(f"Failed to calculate primary statistic. Error: {e}") from e


# --- Helper Functions for SDC Indicator Calculation ---

def _pre_aggregate_secondary_ref(df: pd.DataFrame, config: ConfigModel) -> pd.DataFrame:
    """
    Performs pre-aggregation based on the secondary_ref variable if specified.

    Sums the value_var within unique combinations of row_var, col_var, and secondary_ref.
    Handles NaNs during summation: NaN is treated as 0 if there's at least one non-NaN
    value in the group; otherwise, the result is NaN.

    Args:
        df: The input DataFrame.
        config: The validated configuration object.

    Returns:
        A DataFrame aggregated to the secondary_ref level, ready for SDC calculations.
        Returns the original DataFrame if secondary_ref is not specified.
    """
    if not config.secondary_ref:
        logger.debug("No secondary_ref specified, skipping pre-aggregation.")
        return df

    logger.info(f"Performing pre-aggregation based on secondary_ref: '{config.secondary_ref}'...")
    grouping_vars = config.row_var + config.col_var + [config.secondary_ref]

    # Check required columns exist
    missing_vars = [v for v in grouping_vars + [config.value_var] if v not in df.columns]
    if missing_vars:
        raise CalculationError(f"Columns required for secondary_ref aggregation not found: {missing_vars}")

    # Custom aggregation function for specific NaN handling
    def sum_with_nan_handling(series):
        if series.isna().all():
            return np.nan
        else:
            # Use fillna(0) before sum to treat NaN as 0 only if non-NaN exists
            return series.fillna(0).sum()

    try:
        # Group and aggregate
        # Use dropna=False to include groups with NaN keys
        grouped = df.groupby(grouping_vars, observed=True, dropna=False)
        # Apply the custom sum to the value_var column
        aggregated_df = grouped[config.value_var].apply(sum_with_nan_handling, include_groups=False).reset_index()
        logger.info(f"Pre-aggregation complete. Aggregated DataFrame shape: {aggregated_df.shape}")
        return aggregated_df

    except Exception as e:
        logger.error(f"Error during secondary_ref pre-aggregation: {e}", exc_info=True)
        raise CalculationError(f"Failed to perform secondary_ref pre-aggregation. Error: {e}") from e


def _calculate_indicators_for_group(group: pd.DataFrame, value_col: str) -> pd.Series:
    """
    Calculates all SDC indicators for a given group using helper functions.
    (This function is now primarily used internally by totals._apply_sdc_indicators_to_group)

    Args:
        group: The DataFrame group (already grouped by row/col vars).
        value_col: The name of the column containing the ABSOLUTE values to analyze.

    Returns:
        A Pandas Series containing the calculated indicators for the group.
        Returns NaNs for indicators if the group is empty or contains only NaNs.
    """
    # Work with the absolute values passed in value_col
    abs_values = group[value_col].dropna()

    if abs_values.empty:
        # Return a series of NaNs with the expected index
        indicator_names = ['Count', 'Max', '2nd Max', 'Sum of 3rd to nth', 'Total', 'Dominance (1,k) check', 'Dominance p% check']
        return pd.Series(np.nan, index=indicator_names)

    # Call helper functions
    count = _calculate_sdc_count(abs_values)
    max_val = _calculate_sdc_max(abs_values)
    second_max = _calculate_sdc_second_max(abs_values)
    sum_3rd_nth = _calculate_sdc_sum_third_to_nth(abs_values)
    total = _calculate_sdc_total(abs_values)
    dom_1k = _calculate_sdc_dominance_1k(abs_values)
    dom_p_percent = _calculate_sdc_dominance_p_percent(abs_values)

    indicators = {
        'Count': count,
        'Max': max_val,
        '2nd Max': second_max,
        'Sum of 3rd to nth': sum_3rd_nth,
        'Total': total,
        'Dominance (1,k) check': dom_1k,
        'Dominance p% check': dom_p_percent
    }

    return pd.Series(indicators)


# --- SDC Indicator Calculation ---

def calculate_sdc_indicators(df: pd.DataFrame, config: ConfigModel) -> pd.DataFrame:
    """
    Calculates detailed SDC indicators based on the configuration.

    Handles optional secondary_ref pre-aggregation.
    Calculates indicators based on the absolute value of value_var.
    Structures the output with row_vars and indicators in the index.
    Uses pre-calculated aggregates for efficiency when adding 'Aggregated' rows/columns.

    Args:
        df: The input DataFrame (loaded by data_loader).
        config: The validated configuration object.

    Returns:
        A Pandas DataFrame with row_vars and SDC indicators in the index,
        and col_vars in the columns.

    Raises:
        CalculationError: If errors occur during processing.
    """
    logger.info("Calculating SDC indicators...")

    try:
        # --- Preparation ---
        # 1. Perform pre-aggregation if secondary_ref is specified
        # Use original df for this step
        df_for_sdc = _pre_aggregate_secondary_ref(df.copy(), config) # Use copy to avoid modifying original df

        # 2. Ensure required columns exist in the (potentially aggregated) df
        group_by_vars = config.row_var + config.col_var
        required_cols = set(group_by_vars + [config.value_var])
        missing_cols = required_cols - set(df_for_sdc.columns)
        if missing_cols:
            raise CalculationError(f"Required columns for SDC calculation not found in DataFrame after potential aggregation: {missing_cols}")

        # 3. Create absolute value column for calculations
        abs_value_col = f"{config.value_var}_abs"
        # Ensure value_var exists before abs()
        if config.value_var not in df_for_sdc.columns:
             raise CalculationError(f"Value variable '{config.value_var}' not found in df_for_sdc.")
        df_for_sdc[abs_value_col] = df_for_sdc[config.value_var].abs()
        logger.debug(f"Created absolute value column: '{abs_value_col}'")

        # --- Calculate Base SDC Table (No Aggregates) ---
        logger.info("Calculating base SDC indicator table...")
        # 4. Group by row and column variables
        logger.debug(f"Grouping for SDC indicators by: {group_by_vars}")
        # Use dropna=False to include groups with NaN keys
        if not group_by_vars: # Handle case with no grouping vars (single cell result)
             # Create a dummy grouper that yields the whole DataFrame
             class SingleGroup:
                 def __init__(self, df_single):
                     self.df_single = df_single
                 def __iter__(self):
                     yield (None, self.df_single) # Yield None as group key
                 def apply(self, func, *args, **kwargs):
                     # Apply function directly to the DataFrame
                     return func(self.df_single, *args, **kwargs)

             grouped = SingleGroup(df_for_sdc)
             index_names = []
        else:
             grouped = df_for_sdc.groupby(group_by_vars, observed=True, dropna=False)
             index_names = group_by_vars


        # 5. Apply indicator calculation function (now imported from totals)
        # Use the internal _apply_sdc_indicators_to_group which handles the logic
        from .totals import _apply_sdc_indicators_to_group # Ensure it's available
        intermediate_result = grouped.apply(_apply_sdc_indicators_to_group,
                                            value_col=abs_value_col,
                                            include_groups=False)

        # Check if intermediate result is empty before reshaping
        if intermediate_result.empty:
            logger.warning("Intermediate result after grouped apply is empty.")
            indicator_series = pd.Series(dtype=float) # Create empty series to handle gracefully
        # Handle case where apply returns a DataFrame directly (e.g., single group)
        elif isinstance(intermediate_result, pd.DataFrame):
             # Stack the DataFrame to get the desired Series structure (index=groups+indicator)
             intermediate_result.columns.name = 'Indicator' # Name the indicator level
             indicator_series = intermediate_result.stack(future_stack=True)
        # Handle case where apply returns a Series but index is not MultiIndex (single group result)
        elif isinstance(intermediate_result, pd.Series) and not isinstance(intermediate_result.index, pd.MultiIndex):
             # This happens when grouped.apply returns a Series for a single group
             # Convert to DataFrame, set index, and stack
             temp_df = intermediate_result.to_frame().T
             # Create index based on group_by_vars (which should be empty in this case)
             if not group_by_vars:
                 temp_df.index = pd.Index([None] * len(temp_df)) # No grouping levels
             else:
                 # This case should ideally be handled by the DataFrame path above, but handle defensively
                 temp_df.index = pd.MultiIndex.from_tuples([intermediate_result.name], names=group_by_vars)

             temp_df.columns.name = 'Indicator'
             indicator_series = temp_df.stack(dropna=False)

        elif isinstance(intermediate_result.iloc[0], pd.Series):
            # Reshape: Turn the Series values into columns, then stack the columns into the index
            # Use stack(dropna=False) to keep groups that might have all NaN indicators
            indicator_series = intermediate_result.apply(pd.Series).stack(dropna=False)
            # logger.debug(f"Reshaped indicator series index: {indicator_series.index}") # Can be verbose
        else:
            # Handle unexpected case where apply didn't return Series (e.g., scalar)
            logger.warning(f"Unexpected intermediate result type: {type(intermediate_result)}. Assigning directly.")
            indicator_series = intermediate_result # Assign directly, might cause issues later


        # Handle empty result series after potential reshape
        if indicator_series.empty:
            logger.warning("SDC indicator calculation resulted in an empty series. Returning empty DataFrame.")
            # Create empty DF with appropriate structure
            indicator_level_name = 'Indicator' # Define expected name
            index_names = config.row_var + [indicator_level_name]
            index = pd.MultiIndex(levels=[[]]*len(index_names), codes=[[]]*len(index_names), names=index_names)
            # Handle columns based on whether col_var exists
            columns = pd.MultiIndex(levels=[[]]*len(config.col_var), codes=[[]]*len(config.col_var), names=config.col_var) if config.col_var else pd.Index(['Value'])
            return pd.DataFrame(index=index, columns=columns, dtype=float)

        # Name the index levels robustly BEFORE unstacking
        indicator_level_name = 'Indicator' # Define expected name
        # The index should now be (group_by_vars..., Indicator)
        expected_index_names = group_by_vars + [indicator_level_name]
        # Check if the last level name needs setting
        current_names = list(indicator_series.index.names)

        # Ensure the number of names matches the number of levels
        if len(current_names) != indicator_series.index.nlevels:
             logger.warning(f"Mismatch between number of index names ({len(current_names)}) and levels ({indicator_series.index.nlevels}). Resetting names.")
             current_names = [None] * indicator_series.index.nlevels # Reset names

        # Set names if they don't match expected
        if list(current_names) != expected_index_names:
             if len(current_names) == len(expected_index_names):
                 try:
                     # Set the expected names
                     indicator_series.index.names = expected_index_names
                     logger.debug(f"Set stacked index names to: {expected_index_names}")
                 except Exception as e:
                     logger.warning(f"Could not set stacked index names: {e}")
             else:
                  logger.warning(f"Cannot set index names: Length mismatch. Index names: {current_names}, Expected names: {expected_index_names}")

        logger.debug(f"DEBUG: Index before unstack: names={indicator_series.index.names}, levels={indicator_series.index.levels}") # DEBUG ADDED

        # 7. Restructure the output: Unstack column variables if they exist
        if config.col_var:
            try:
                # Ensure the levels to unstack actually exist in the index
                levels_to_unstack = [name for name in config.col_var if name in indicator_series.index.names]
                if not levels_to_unstack:
                    logger.warning(f"Column variables {config.col_var} not found in indicator series index {indicator_series.index.names}. Cannot unstack.")
                    # If we can't unstack, treat as if col_var was empty for structure
                    # The index should be (row_var..., Indicator)
                    sdc_table_base = indicator_series.to_frame(name='Value')
                elif len(levels_to_unstack) != len(config.col_var):
                    logger.warning(f"Only found subset {levels_to_unstack} of column variables {config.col_var} in index {indicator_series.index.names}. Unstacking only found levels.")
                    sdc_table_base = indicator_series.unstack(level=levels_to_unstack)
                else:
                    # Use original config.col_var if all found and names match
                    sdc_table_base = indicator_series.unstack(level=config.col_var)

            except Exception as e:
                logger.error(f"Error unstacking SDC indicator table: {e}", exc_info=True)
                logger.debug(f"Indicator Series before unstack:\n{indicator_series}")
                raise CalculationError(f"Failed to unstack SDC indicator table. Error: {e}") from e
        else: # No col_vars, result should have row_vars + Indicator in index
             # Index should be (row_var..., Indicator)
             sdc_table_base = indicator_series.to_frame(name='Value')

        # Ensure sdc_table_base is a DataFrame
        if isinstance(sdc_table_base, pd.Series):
            sdc_table_base = sdc_table_base.to_frame(name='Value') # Use a default name

        logger.info(f"Base SDC indicator table calculated. Shape: {sdc_table_base.shape}")
        logger.debug(f"DEBUG: Base SDC table columns after unstack: {sdc_table_base.columns}") # DEBUG ADDED
        # logger.debug(f"Base SDC table index:\n{sdc_table_base.index}") # Verbose
        # logger.debug(f"Base SDC table columns:\n{sdc_table_base.columns}") # Verbose - Keep original commented out


        # --- Post-Unstacking Adjustments (Applied to Base Table) ---

        # 8. Ensure all expected columns are present (reindex based on original unique col_var values)
        if config.col_var:
            expected_cols_index = None
            try:
                # Use unique values from df_for_sdc (which might include secondary_ref effect)
                if len(config.col_var) == 1:
                    col_name = config.col_var[0]
                    unique_vals = df_for_sdc[col_name].unique()
                    expected_cols_index = pd.Index(unique_vals, name=col_name)
                else:
                    unique_combos_df = df_for_sdc[config.col_var].drop_duplicates()
                    expected_cols_index = pd.MultiIndex.from_frame(unique_combos_df, names=config.col_var)

                # Sort expected columns lexicographically if possible
                if not expected_cols_index.empty:
                    try:
                        # Sorting might fail with mixed types or NaNs
                        expected_cols_index = expected_cols_index.sort_values()
                    except TypeError as e:
                        logger.warning(f"Could not sort expected columns due to mixed types or NaNs: {e}. Proceeding with unsorted unique columns.")

            except Exception as e:
                logger.warning(f"Could not create expected columns index: {e}. Column reindexing might be incomplete.")

            # Reindex only if expected_cols_index was successfully created
            if expected_cols_index is not None:
                logger.debug(f"Reindexing base SDC table columns to ensure all are present: {expected_cols_index.names}")
                if not sdc_table_base.empty:
                    try:
                        # Align column names if both are MultiIndex and names differ
                        if isinstance(sdc_table_base.columns, pd.MultiIndex) and isinstance(expected_cols_index, pd.MultiIndex):
                            if sdc_table_base.columns.nlevels == expected_cols_index.nlevels and list(sdc_table_base.columns.names) != list(expected_cols_index.names):
                                    logger.debug(f"Aligning sdc_table_base column names {sdc_table_base.columns.names} to expected {expected_cols_index.names}")
                                    sdc_table_base.columns.names = expected_cols_index.names

                        # Reindex using the expected index
                        sdc_table_base = sdc_table_base.reindex(columns=expected_cols_index, fill_value=np.nan)
                    except Exception as reindex_e:
                        logger.error(f"Failed to reindex base SDC table columns: {reindex_e}", exc_info=True)
                        logger.debug(f"Base SDC Table columns: {sdc_table_base.columns}")
                        logger.debug(f"Expected columns: {expected_cols_index}")

                elif not expected_cols_index.empty: # If sdc_table_base is empty but expected cols exist
                    # Ensure the index of the empty table matches the expected final index structure
                    final_index_names = config.row_var + [indicator_level_name]
                    current_index = sdc_table_base.index
                    if not isinstance(current_index, pd.MultiIndex) or list(current_index.names) != final_index_names:
                        logger.debug("Creating empty base SDC DataFrame with corrected empty index structure.")
                        empty_index = pd.MultiIndex(levels=[[]]*len(final_index_names), codes=[[]]*len(final_index_names), names=final_index_names)
                        sdc_table_base = pd.DataFrame(index=empty_index, columns=expected_cols_index, dtype=float)
                    else:
                        # Index structure is already okay, just assign columns
                        sdc_table_base = pd.DataFrame(index=current_index, columns=expected_cols_index, dtype=float)


        # 9. Handle 'median' and 'count' statistic case AFTER unstacking and reindexing columns
        if config.statistic == "median" or config.statistic == "count":
            logger.info(f"Statistic is '{config.statistic}', keeping only 'Count' indicator row.")
            # Check if index is MultiIndex and has the indicator level
            if isinstance(sdc_table_base.index, pd.MultiIndex) and indicator_level_name in sdc_table_base.index.names:
                try:
                    # Check if 'Count' exists before trying to filter
                    indicator_values = sdc_table_base.index.get_level_values(indicator_level_name)
                    if 'Count' in indicator_values:
                        # Use boolean indexing which is generally safer than .loc with potential duplicate indices
                        sdc_table_base = sdc_table_base[indicator_values == 'Count']
                    else:
                        logger.warning("Median statistic selected, but 'Count' indicator not found in index. Returning unfiltered table.")
                except KeyError:
                    logger.warning(f"Could not filter median table: '{indicator_level_name}' level key error.")

                if sdc_table_base.empty:
                    logger.warning("SDC table became empty after filtering for 'Count' indicator.")
            # Handle case where index might be simple (e.g., only 'Indicator' level if no row_vars)
            elif sdc_table_base.index.name == indicator_level_name:
                if 'Count' in sdc_table_base.index:
                    sdc_table_base = sdc_table_base.loc[['Count']] # .loc is okay for simple index
                else:
                    logger.warning("Median statistic selected, but 'Count' indicator not found in index. Returning unfiltered table.")
                if sdc_table_base.empty:
                    logger.warning("SDC table became empty after filtering for 'Count' indicator.")
            else:
                logger.warning(f"Cannot filter for 'Count' indicator row: Index structure not recognized or '{indicator_level_name}' level missing/unnamed.")

        # Replace NaN in index with 'Missing' placeholder before aggregation helper
        # to avoid .loc issues with NaN keys. Styler handles final output name.
        if isinstance(sdc_table_base.index, pd.MultiIndex):
            new_levels = []
            for level in sdc_table_base.index.levels:
                # Use fillna only if NaN is present to avoid dtype issues
                if level.isna().any():
                     # Ensure level is object dtype before fillna if it contains mixed types or NaN
                     new_level = level.astype(object).fillna('Missing')
                else:
                     new_level = level
                new_levels.append(new_level)
            sdc_table_base.index = sdc_table_base.index.set_levels(new_levels)
            logger.debug("Replaced NaN with 'Missing' in SDC table base MultiIndex levels.")
        elif pd.isna(sdc_table_base.index).any(): # Handle simple index with NaN
             sdc_table_base.index = sdc_table_base.index.fillna('Missing')
             logger.debug("Replaced NaN with 'Missing' in SDC table base simple index.")

        # --- Pre-calculate SDC Aggregates ---
        precalculated_sdc_aggregates = {}
        if config.include_aggregated:
             # Use the potentially pre-aggregated df_for_sdc (which already has abs_value_col)
             # for calculating SDC aggregates to ensure consistency when secondary_ref is used.
             precalculated_sdc_aggregates = _calculate_all_sdc_aggregates(
                 df_orig=df_for_sdc, # Use potentially pre-aggregated data with abs value
                 config=config,
                 abs_value_col=abs_value_col
             )

        # --- Add Aggregated Rows/Columns using Precalculated Values ---
        sdc_table_final = sdc_table_base.copy() # Start with the processed base table

        # 10. Add Aggregated Columns (if requested)
        if config.include_aggregated and config.col_var:
            logger.info("Calling helper to add SDC aggregated columns using precalculated values...")
            sdc_table_final = _add_sdc_hierarchical_aggregates( # Use refactored SDC agg function
                sdc_df=sdc_table_final, # Pass current state of the table
                config=config,
                axis=1, # Specify axis=1 for columns
                precalculated_aggregates=precalculated_sdc_aggregates
                # agg_label defaults to 'Aggregated'
            )
            logger.info("Returned from adding SDC aggregated columns.")
            logger.debug(f"DEBUG: Final SDC table columns after adding column aggregates: {sdc_table_final.columns}") # DEBUG ADDED
        elif config.include_aggregated and not config.col_var:
             logger.debug("include_aggregated is True, but col_var is empty. Skipping SDC column aggregation.")

        # Add Aggregated Rows (if requested)
        if config.include_aggregated and config.row_var:
            logger.info("Calling helper to add SDC aggregated rows using precalculated values...")
            sdc_table_final = _add_sdc_hierarchical_aggregates( # Use refactored SDC agg function
                sdc_df=sdc_table_final, # Pass table possibly with agg columns already added
                config=config,
                axis=0, # Specify axis=0 for rows
                precalculated_aggregates=precalculated_sdc_aggregates
                # agg_label defaults to 'Aggregated'
            )
            logger.info("Returned from adding SDC aggregated rows.")
        elif config.include_aggregated and not config.row_var:
             logger.debug("include_aggregated is True, but row_var is empty. Skipping SDC row aggregation.")

        # REMOVED: Call to _fill_aggregate_intersections_sdc
        # Intersections are now populated directly by _add_sdc_hierarchical_aggregates using precalculated values.

        # 11. Reorder indicator rows if desired_order is specified (and statistic is 'sum')
        if config.statistic == 'sum' and config.desired_order:
            # Check if index is MultiIndex and has the indicator level
            if isinstance(sdc_table_final.index, pd.MultiIndex) and indicator_level_name in sdc_table_final.index.names:
                try:
                    logger.debug(f"Reordering indicator level '{indicator_level_name}' based on desired_order.")
                    current_indicators = sdc_table_final.index.get_level_values(indicator_level_name).unique()
                    # Filter desired_order to only include indicators actually present
                    ordered_indicators = [ind for ind in config.desired_order if ind in current_indicators]
                    # Find any indicators present but not in desired_order
                    missing_from_desired = [ind for ind in current_indicators if ind not in ordered_indicators]
                    # Combine them: desired order first, then the rest
                    final_order = ordered_indicators + missing_from_desired

                    if final_order and set(final_order) == set(current_indicators):
                        # Use reindex for ordering
                        sdc_table_final = sdc_table_final.reindex(final_order, level=indicator_level_name)
                        logger.debug("Indicator rows reordered successfully.")
                    elif not final_order and not current_indicators.empty:
                        logger.warning("Final order for indicators is empty, but indicators exist. Skipping reorder.")
                    elif set(final_order) != set(current_indicators):
                        logger.warning(f"Constructed final order {final_order} does not match current indicators {current_indicators}. Skipping reorder.")

                except Exception as e:
                    logger.warning(f"Could not reorder indicator rows: {e}", exc_info=True)
                    # Proceed with the default order
            elif config.desired_order: # Log warning if reorder requested but index structure is wrong
                logger.warning(f"Cannot reorder indicator rows: Index is not MultiIndex or '{indicator_level_name}' level missing.")

        # Final check on index names before returning (if MultiIndex)
        # This ensures the remaining levels (row_var + Indicator) have the correct names
        if isinstance(sdc_table_final.index, pd.MultiIndex):
            expected_final_names = config.row_var + [indicator_level_name]
            if list(sdc_table_final.index.names) != expected_final_names:
                try:
                    # Ensure correct number of levels before renaming
                    if len(sdc_table_final.index.levels) == len(expected_final_names):
                        sdc_table_final.index.names = expected_final_names
                        logger.debug(f"Set final index names to: {expected_final_names}")
                    else:
                        logger.warning(f"Final index level count ({len(sdc_table_final.index.levels)}) mismatch with expected ({len(expected_final_names)}). Cannot set names.")
                except Exception as final_name_e:
                    logger.warning(f"Failed to set final index names: {final_name_e}")
        # Also check if the index is simple and should be named 'Indicator' (case with no row_var)
        elif not config.row_var and sdc_table_final.index.name != indicator_level_name:
             # Ensure index is not empty before naming
             if not sdc_table_final.index.empty:
                 sdc_table_final.index.name = indicator_level_name


        logger.info(f"SDC indicator calculation complete. Result table shape: {sdc_table_final.shape}")
        # logger.debug(f"Final SDC table index:\n{sdc_table_final.index}") # Verbose
        # logger.debug(f"Final SDC table columns:\n{sdc_table_final.columns}") # Verbose
        return sdc_table_final

    except CalculationError: # Re-raise CalculationErrors directly
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred during SDC indicator calculation: {e}", exc_info=True)
        raise CalculationError(f"Failed to calculate SDC indicators. Error: {e}") from e