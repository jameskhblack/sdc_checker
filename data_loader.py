"""Handles data loading from a provided DataFrame"""

import sys
import os
import logging
import pandas as pd
import numpy as np
from typing import Optional, List

# Assuming ConfigModel and exceptions are importable from sibling modules
from .config import ConfigModel
from .exceptions import DataLoaderError

# Get the logger instance configured in utils
logger = logging.getLogger("sdc_checker.data_loader")

# --- Data Loading Function ---

def load_data(config: ConfigModel, input_df: pd.DataFrame) -> pd.DataFrame:
    """
    Loads data from a provided Pandas DataFrame.

    Args:
        config: The validated configuration object (ConfigModel).
        input_df: A Pandas DataFrame.

    Returns:
        A Pandas DataFrame containing the required data.

    Raises:
        DataLoaderError: If required columns are missing in the input_df.
    """
    required_vars = set(config.row_var + config.col_var + [config.value_var])
    if config.pweight:
        required_vars.add(config.pweight)
    if config.secondary_ref:
        required_vars.add(config.secondary_ref)

    if input_df is not None:
        logger.info("Using provided Pandas DataFrame as input.")

        # Validate required columns exist in the provided DataFrame
        missing_cols = required_vars - set(input_df.columns)
        if missing_cols:
            msg = f"Provided DataFrame is missing required columns specified in config: {missing_cols}"
            logger.error(msg)
            raise DataLoaderError(msg)

        # Only keep the columns we need
        df = input_df[list(required_vars)].copy()

        # Convert row_var and col_var columns to string, replacing NaN with "Missing"
        convert_vars = config.row_var + config.col_var
        for var in convert_vars:
            if var in df.columns:
                 logger.debug(f"Converting column '{var}' to string and filling NaN with 'Missing'.")
                 # Fill NaN first, then convert to string to handle potential mixed types
                 df[var] = df[var].fillna('Missing').astype(str)

        logger.debug(f"Provided DataFrame columns validated and potentially converted: {list(df.columns)}")
        return df
