"""
SDC Checker Package

Provides functions to perform Statistical Disclosure Control checks and generate
styled Excel reports.
"""
import logging
from typing import Dict, Any, Optional
import pandas as pd
import warnings

# Import core components from submodules
from . import utils
from . import config as config_module
from . import data_loader
from . import calculator
from . import sdc_logic
from . import excel_writer
from . import styler
from . import stats
from . import totals
from .exceptions import SDCCheckerBaseError

# Suppress lexsorted performance warning in totals.py
warnings.filterwarnings("ignore", message="dropping on a non-lexsorted multi-index without a level parameter may impact performance.")

# Define package-level attributes if needed (e.g., version)
__version__ = "0.1.0"

# --- Public API Functions ---

def generate_data(config_dict: Dict[str, Any], input_df: pd.DataFrame = None) -> None:
    """
    Generates the primary statistic table and writes it to an Excel sheet.

    Orchestrates configuration validation, data loading, calculation,
    Excel writing, and styling for the primary output table.

    Args:
        config_dict: A dictionary containing the configuration parameters.
        input_df: An Pandas DataFrame to use as input.

    Raises:
        SDCCheckerBaseError: If any validation, loading, calculation, or
                             writing error occurs within the package.
    """
    # Logging should be configured by the calling application, not the library function.
    # Get the logger instance configured elsewhere.
    logger = logging.getLogger("sdc_checker.generate_data") # Or just "sdc_checker" if less granularity needed
    logger.info(f"--- Starting generate_data (SDC Checker v{__version__}) ---")
    workbook = None # Initialize workbook variable

    try:
        # 1. Validate Configuration
        logger.info("Validating configuration...")
        config = config_module.validate_config(config_dict)
        logger.info("Configuration validated successfully.")

        # Apply default sheet name before using it
        config = config_module.set_default_sheet_name(config, "generate_data")

        # 2. Load Data
        logger.info("Loading data...")
        df = data_loader.load_data(config, input_df=input_df)
        logger.info("Data loaded successfully.")

        # 3. Calculate Primary Statistic
        logger.info("Calculating primary statistic...")
        primary_table = calculator.calculate_primary_statistic(df, config)
        logger.info("Primary statistic calculated successfully.")

        # 4. Write to Excel (without saving yet)
        logger.info("Writing primary statistic table to Excel sheet...")

        # 4a. Construct Title
        stat_desc = config.statistic.capitalize()
        primary_title_text = f"Raw Output: {stat_desc} of {config.value_var}, row_var: {', '.join(config.row_var)}, col_var: {', '.join(config.col_var)}"
        if config.pweight: # Check if pweight is not None or empty
            primary_title_text += f", weighted by {config.pweight}"

        # 4b. Write to Excel (without saving yet)
        workbook, worksheet = excel_writer.write_excel_sheet(
            df=primary_table,
            config=config,
            title_text=primary_title_text, # Pass the constructed title
            sheet_name_suffix="", # No suffix for primary sheet
            existing_workbook=None # Pass None initially
        )
        logger.info(f"Sheet '{worksheet.title}' written.")

        # 5. Apply Styles
        logger.info(f"Applying styles to sheet '{worksheet.title}'...")
        styler.apply_styles(
            workbook=workbook,
            worksheet=worksheet,
            original_df=primary_table, # Pass the calculated table for structure ref
            config=config,
            table_type='primary'
        )
        logger.info("Styles applied successfully.")

        # 6. Save Workbook
        logger.info(f"Saving workbook to '{config.output}'...")
        workbook.save(config.output)
        logger.info("Workbook saved successfully.")
        logger.info("--- generate_data completed successfully ---")
        print(f"Workbook saved to '{config.output}'")

    except SDCCheckerBaseError as e:
        logger.error(f"generate_data failed: {e}", exc_info=True)
        # Optionally save workbook even if styling failed? For now, only save on full success.
        # if workbook and config and config.output:
        #     try:
        #         logger.warning(f"Attempting to save workbook to '{config.output}' despite earlier error...")
        #         workbook.save(config.output)
        #         logger.warning("Workbook saved with potentially incomplete styling.")
        #     except Exception as save_e:
        #         logger.error(f"Failed to save workbook after error: {save_e}", exc_info=True)
        raise # Re-raise the original SDC Checker error
    except Exception as e:
        logger.error(f"An unexpected error occurred in generate_data: {e}", exc_info=True)
        raise # Re-raise unexpected errors


def generate_sdc(config_dict: Dict[str, Any], input_df: pd.DataFrame = None) -> None:
    """
    Generates the detailed SDC indicator table and the Pass/Fail summary table,
    writing them to separate sheets in an Excel file.

    Orchestrates configuration validation, data loading, SDC calculations,
    Pass/Fail logic, Excel writing, and styling.

    Args:
        config_dict: A dictionary containing the configuration parameters.
        input_df: An Pandas DataFrame to use as input.

    Raises:
        SDCCheckerBaseError: If any validation, loading, calculation, or
                             writing error occurs within the package.
    """
    # Logging should be configured by the calling application, not the library function.
    # Get the logger instance configured elsewhere.
    logger = logging.getLogger("sdc_checker.generate_sdc") # Or just "sdc_checker" if less granularity needed
    logger.info(f"--- Starting generate_sdc (SDC Checker v{__version__}) ---")
    workbook = None # Initialize workbook variable

    try:
        # 1. Validate Configuration
        logger.info("Validating configuration...")
        config = config_module.validate_config(config_dict)
        logger.info("Configuration validated successfully.")

        # Apply default sheet name before using it
        config = config_module.set_default_sheet_name(config, "generate_sdc")

        # 2. Load Data
        logger.info("Loading data...")
        df = data_loader.load_data(config, input_df=input_df)
        logger.info("Data loaded successfully.")

        # 3. Calculate SDC Indicators
        logger.info("Calculating SDC indicators...")
        sdc_indicators_table = calculator.calculate_sdc_indicators(df, config)
        logger.info("SDC indicators calculated successfully.")

        # 4. Generate Pass/Fail Table
        logger.info("Generating Pass/Fail table...")
        pass_fail_table = sdc_logic.generate_pass_fail_table(sdc_indicators_table, config)
        logger.info("Pass/Fail table generated successfully.")

        # 5. Write SDC Indicators Table to Excel (Get initial workbook)
        logger.info("Writing SDC indicators table to Excel sheet...")
        # Initialize workbook to None before the first call
        workbook = None

        # 5a. Construct Title for Detailed SDC Table
        sdc_detailed_title_text = f"Full SDC Working: Absolute value of {config.value_var}, row_var: {', '.join(config.row_var)}, col_var: {', '.join(config.col_var)}, unweighted"
        if config.secondary_ref: # Check if secondary_ref is not None or empty
            sdc_detailed_title_text += f", over reference {config.secondary_ref}"

        # 5b. Write SDC Indicators Table to Excel (Get initial workbook)
        workbook, sdc_worksheet = excel_writer.write_excel_sheet(
            df=sdc_indicators_table,
            config=config,
            title_text=sdc_detailed_title_text, # Pass the constructed title
            sheet_name_suffix="", # No suffix for detailed SDC sheet
            existing_workbook=workbook # Pass None initially
        )
        logger.info(f"Sheet '{sdc_worksheet.title}' written to workbook object.")

        # 6. Apply Styles to SDC Indicators Sheet (Modifies the workbook in memory)
        logger.info(f"Applying styles to sheet '{sdc_worksheet.title}'...")
        styler.apply_styles(
            workbook=workbook, # Use the workbook object from the previous step
            worksheet=sdc_worksheet,
            original_df=sdc_indicators_table, # Pass the calculated table
            config=config,
            table_type='sdc_detailed'
        )
        logger.info("Styles applied successfully to SDC indicators sheet.")

        # 7. Write Pass/Fail Table to Excel (Pass the existing workbook)
        logger.info("Writing Pass/Fail table to Excel sheet...")
        # Pass the workbook object obtained and potentially modified from the first write/style

        # 7a. Construct Title for Pass/Fail Table
        sdc_pass_fail_title_text = f"SDC Cell Pass (P) / Fail (F): Absolute value of {config.value_var}, row_var: {', '.join(config.row_var)}, col_var: {', '.join(config.col_var)}, unweighted"
        if config.secondary_ref: # Check if secondary_ref is not None or empty
            sdc_pass_fail_title_text += f", over reference {config.secondary_ref}"

        # 7b. Write Pass/Fail Table to Excel (Pass the existing workbook)
        workbook, pf_worksheet = excel_writer.write_excel_sheet(
            df=pass_fail_table,
            config=config,
            title_text=sdc_pass_fail_title_text, # Pass the constructed title
            sheet_name_suffix=" Pass-Fail", # Add suffix
            existing_workbook=workbook # Pass the SAME workbook object
        )
        logger.info(f"Sheet '{pf_worksheet.title}' written to workbook object.")

        # 8. Apply Styles to Pass/Fail Sheet (Modifies the same workbook in memory)
        logger.info(f"Applying styles to sheet '{pf_worksheet.title}'...")
        styler.apply_styles(
            workbook=workbook, # Use the SAME workbook object
            worksheet=pf_worksheet,
            original_df=pass_fail_table, # Pass the calculated table
            config=config,
            table_type='pass_fail'
        )
        logger.info("Styles applied successfully to Pass/Fail sheet.")

        # 9. Save the single Workbook object containing both styled sheets
        logger.info(f"Saving workbook to '{config.output}'...")
        workbook.save(config.output)
        logger.info("Workbook saved successfully.")
        logger.info("--- generate_sdc completed successfully ---")
        print(f"Workbook saved to '{config.output}'")

    except SDCCheckerBaseError as e:
        logger.error(f"generate_sdc failed: {e}", exc_info=True)
        # Optionally save workbook even if styling failed?
        # if workbook and config and config.output:
        #     try:
        #         logger.warning(f"Attempting to save workbook to '{config.output}' despite earlier error...")
        #         workbook.save(config.output)
        #         logger.warning("Workbook saved with potentially incomplete styling.")
        #     except Exception as save_e:
        #         logger.error(f"Failed to save workbook after error: {save_e}", exc_info=True)
        raise # Re-raise the original SDC Checker error
    except Exception as e:
        logger.error(f"An unexpected error occurred in generate_sdc: {e}", exc_info=True)
        raise # Re-raise unexpected errors