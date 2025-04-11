"""Handles the logic for determining Pass/Fail status based on SDC indicators."""

import logging
import pandas as pd
import numpy as np

# Assuming ConfigModel and exceptions are importable from sibling modules
from .config import ConfigModel
from .exceptions import SDCLogicError

# Get the logger instance configured in utils
logger = logging.getLogger("sdc_checker.sdc_logic")

def generate_pass_fail_table(sdc_indicators_df: pd.DataFrame, config: ConfigModel) -> pd.DataFrame:
    """
    Generates a DataFrame indicating Pass ('P') or Fail ('F') for each cell
    based on the calculated SDC indicators and configured rules.

    Args:
        sdc_indicators_df: DataFrame containing detailed SDC indicators,
                           with row_vars and 'Indicator' in the index, and
                           col_vars in the columns.
        config: The validated configuration object.

    Returns:
        A DataFrame with the same index (row_vars) and columns (col_vars)
        as the primary statistic table, containing 'P' or 'F' values.

    Raises:
        SDCLogicError: If required indicators are missing or errors occur.
    """
    logger.info("Generating Pass/Fail table based on SDC rules...")
    logger.debug(f"Input SDC indicators table shape: {sdc_indicators_df.shape}")
    logger.debug(f"SDC Rules: Primary={config.sdc_rules.primary_threshold}, Dom(1,k)={config.sdc_rules.dominance_1k}, Dom(p%)={config.sdc_rules.dominance_p_percent}")

    try:
        # Check if input DataFrame is empty
        if sdc_indicators_df.empty:
            logger.warning("Input SDC indicators DataFrame is empty. Returning empty Pass/Fail table.")
            # Create an empty DataFrame with the expected primary table structure
            index = pd.MultiIndex(levels=[[]]*len(config.row_var), codes=[[]]*len(config.row_var), names=config.row_var)
            columns = pd.MultiIndex(levels=[[]]*len(config.col_var), codes=[[]]*len(config.col_var), names=config.col_var)
            return pd.DataFrame(index=index, columns=columns, dtype=str)


        # --- Extract required indicators ---
        # Assume 'Indicator' is the last level of the index
        indicator_level_name = sdc_indicators_df.index.names[-1]
        if indicator_level_name != 'Indicator':
             # If apply in calculator returned df with indicators as columns, adjust logic
             # For now, assume indicators are in the index as planned
             logger.warning(f"Expected 'Indicator' as the last index level, found '{indicator_level_name}'. Attempting to proceed.")
             # Potential fallback: Try finding 'Indicator' level by name if not last?

        def get_indicator(name):
            try:
                # Select the indicator data, dropping the 'Indicator' level from index
                indicator_df = sdc_indicators_df.xs(name, level=indicator_level_name)
                return indicator_df
            except KeyError:
                # If only 'Count' exists (median case), other indicators won't be present
                if name == 'Count' and name in sdc_indicators_df.index.get_level_values(indicator_level_name):
                     return sdc_indicators_df.xs(name, level=indicator_level_name)
                logger.info(f"Required SDC indicator '{name}' not found in the input DataFrame. Skipping related checks.")
                # Return an empty DataFrame or Series with the same structure but all NaNs?
                # Let's return None, and handle it in the rule application.
                return None
            except Exception as e:
                 logger.error(f"Error extracting indicator '{name}': {e}", exc_info=True)
                 raise SDCLogicError(f"Failed to extract SDC indicator '{name}'. Error: {e}") from e

        count_indicator = get_indicator('Count')
        dom1k_indicator = get_indicator('Dominance (1,k) check')
        domp_indicator = get_indicator('Dominance p% check')

        # Debug: Log missing values in indicators
        if count_indicator is not None:
            logger.debug(f"Count indicator missing values:\n{count_indicator.isna().astype(int).to_string()}")
        if dom1k_indicator is not None:
            logger.debug(f"Dominance (1,k) indicator missing values:\n{dom1k_indicator.isna().astype(int).to_string()}")
        if domp_indicator is not None:
            logger.debug(f"Dominance p% indicator missing values:\n{domp_indicator.isna().astype(int).to_string()}")

        # If count_indicator is None (e.g., empty input), we can't proceed meaningfully
        if count_indicator is None:
             logger.error("Could not extract 'Count' indicator. Cannot generate Pass/Fail table.")
             raise SDCLogicError("Essential 'Count' indicator missing from SDC indicators table.")

        # Initialize the combined failure mask based on the structure of count_indicator
        # Start with all False (Pass)
        fail_combined = pd.DataFrame(False, index=count_indicator.index, columns=count_indicator.columns)

        # --- Apply SDC Rules ---

        # 1. Primary Threshold (Count)
        primary_threshold = config.sdc_rules.primary_threshold
        # Fail if count < threshold
        fail_count_raw = (count_indicator < primary_threshold)
        logger.debug(f"Primary threshold fail mask before missing handling:\n{fail_count_raw.astype('Int8').to_string()}")
        # Explicitly treat missing as Fail
        missing_count_mask = count_indicator.isna()
        logger.debug(f"Primary threshold missing mask:\n{missing_count_mask.astype(int).to_string()}")
        fail_count = fail_count_raw | missing_count_mask
        fail_combined |= fail_count
        logger.debug(f"Applied Primary Threshold (<{primary_threshold}) with missing treated as Fail. Failures:\n{fail_count.astype(int).to_string()}")


        # 2. Dominance (1,k)
        if dom1k_indicator is not None:
            dom1k_threshold = config.sdc_rules.dominance_1k
            # Fail if dom1k > threshold
            fail_dom1k_raw = (dom1k_indicator > dom1k_threshold)
            logger.debug(f"Dominance (1,k) fail mask before missing handling:\n{fail_dom1k_raw.astype('Int8').to_string()}")
            # Explicitly treat missing as Fail
            missing_dom1k_mask = dom1k_indicator.isna()
            logger.debug(f"Dominance (1,k) missing mask:\n{missing_dom1k_mask.astype(int).to_string()}")
            fail_dom1k = fail_dom1k_raw | missing_dom1k_mask
            fail_combined |= fail_dom1k
            logger.debug(f"Applied Dominance (1,k) (> {dom1k_threshold}) with missing treated as Fail. Failures:\n{fail_dom1k.astype(int).to_string()}")
        else:
            logger.debug("Dominance (1,k) indicator not found, skipping check.")


        # 3. Dominance p%
        if domp_indicator is not None:
            domp_threshold = config.sdc_rules.dominance_p_percent
            # Fail if dom_p < threshold (as per plan spec - seems unusual, usually >). Handle NaN (treat as Pass).
            # Adding a specific log message about this rule direction.
            logger.debug(f"Applying Dominance p% check: Fail if calculated p% < threshold ({domp_threshold}).")
            fail_domp_raw = (domp_indicator < domp_threshold)
            logger.debug(f"Dominance p% fail mask before missing handling:\n{fail_domp_raw.astype('Int8').to_string()}")
            # Explicitly treat missing as Fail
            missing_domp_mask = domp_indicator.isna()
            logger.debug(f"Dominance p% missing mask:\n{missing_domp_mask.astype(int).to_string()}")
            fail_domp = fail_domp_raw | missing_domp_mask
            fail_combined |= fail_domp
            logger.debug(f"Applied Dominance p% (< {domp_threshold}) with missing treated as Fail. Failures:\n{fail_domp.astype(int).to_string()}")
        else:
            logger.debug("Dominance p% indicator not found, skipping check.")


        # --- Create Pass/Fail DataFrame ---
        # Map True (Fail) to 'F' and False (Pass) to 'P'
        pass_fail_table = fail_combined.map(lambda x: 'F' if x else 'P')

        logger.info(f"Pass/Fail table generated successfully. Shape: {pass_fail_table.shape}")
        return pass_fail_table

    except Exception as e:
        logger.error(f"An unexpected error occurred during Pass/Fail table generation: {e}", exc_info=True)
        raise SDCLogicError(f"Failed to generate Pass/Fail table. Error: {e}") from e