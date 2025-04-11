"""Utility functions for the SDC Checker application."""

import logging
import re
import sys
import pandas as pd
from typing import Tuple

try:
    import colorama
    colorama.init(autoreset=True) # Initialize colorama
    COLORAMA_AVAILABLE = True
except ImportError:
    COLORAMA_AVAILABLE = False

# --- Logging Setup ---

# Define custom formatter with colors if colorama is available
class ColorFormatter(logging.Formatter):
    """Custom logging formatter that adds color based on log level."""

    # Base format string
    _base_fmt = "%(asctime)s - %(levelname)-8s - %(message)s (%(filename)s:%(lineno)d)"

    # Color mapping (using names to avoid holding colorama objects if not needed)
    _LEVEL_COLOR_MAP = {
        logging.DEBUG: 'CYAN',
        logging.INFO: 'GREEN',
        logging.WARNING: 'YELLOW',
        logging.ERROR: 'RED',
        logging.CRITICAL: 'MAGENTA',
    }

    def __init__(self, datefmt='%Y-%m-%d %H:%M:%S'):
        # Initialize the base formatter with the core format string and date format
        super().__init__(fmt=self._base_fmt, datefmt=datefmt)

    def format(self, record):
        # Check the current state of COLORAMA_AVAILABLE from the utils module
        # This ensures the patch applied during tests is respected
        use_colors = COLORAMA_AVAILABLE

        # Get the base formatted message first
        log_message = super().format(record)

        if use_colors:
            color_name = self._LEVEL_COLOR_MAP.get(record.levelno)
            if color_name:
                try:
                    # Dynamically get colorama attributes ONLY if needed
                    color_prefix = getattr(colorama.Fore, color_name, '')
                    reset_suffix = getattr(colorama.Style, 'RESET_ALL', '')
                    # Return message wrapped in colors
                    return f"{color_prefix}{log_message}{reset_suffix}"
                except (ImportError, AttributeError):
                    # Fallback if colorama is somehow unavailable despite check
                    # or if attributes are missing
                    pass # Fall through to return uncolored message

        # Return the uncolored message if colors are disabled or lookup failed
        return log_message

# The setup_logging function remains largely the same, just instantiates the updated ColorFormatter
def setup_logging(handler_level=logging.DEBUG, default_package_level=logging.WARNING, force=False) -> logging.Logger:
    """
    Configures the main 'sdc_checker' logger and its handler.

    Allows setting separate levels for the handler (what gets output) and
    the logger itself (the default minimum level for the package).

    Args:
        handler_level: The minimum level the console handler will output.
                       (default: logging.DEBUG to allow fine-grained control).
        default_package_level: The default minimum level for the 'sdc_checker'
                               logger and its children. (default: logging.WARNING).
        force: If True, removes existing handlers and reconfigures.
               If False, skips if handlers already exist.

    Returns:
        The configured 'sdc_checker' logger instance.
    """
    logger = logging.getLogger("sdc_checker") # Get the specific package logger

    # Skip configuration if handlers already exist and force=False
    if logger.hasHandlers() and not force:
        # Optionally log that setup is being skipped
        # logger.debug("Logger 'sdc_checker' already has handlers. Skipping setup.")
        return logger

    # Prevent messages from propagating to the root logger if it's configured elsewhere
    logger.propagate = False

    # Remove existing handlers if force=True or if reconfiguring
    if logger.hasHandlers():
        # logger.debug("Removing existing handlers from 'sdc_checker'.")
        logger.handlers.clear()

    # Set the default level for the package logger
    logger.setLevel(default_package_level)
    # logger.debug(f"Set 'sdc_checker' logger level to {logging.getLevelName(default_package_level)}")

    # Create console handler
    ch = logging.StreamHandler(sys.stdout) # Use stdout for console output
    # Set the level for the handler (determines what messages make it to the console)
    ch.setLevel(handler_level)
    # logger.debug(f"Set console handler level to {logging.getLevelName(handler_level)}")


    # Create formatter and add it to the handler
    # The formatter now handles color logic internally based on COLORAMA_AVAILABLE
    formatter = ColorFormatter()
    ch.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(ch)

    # Child loggers (e.g., sdc_checker.calculator) will inherit the package level
    # and propagate messages to this handler by default. No need to configure them here.

    # logger.info(f"Logging for 'sdc_checker' configured. Handler Level: {logging.getLevelName(handler_level)}, Package Default Level: {logging.getLevelName(default_package_level)}")
    return logger

# --- Sheet Name Sanitization ---

# Characters invalid in Excel sheet names: []*/\?
INVALID_SHEET_NAME_CHARS = re.compile(r'[\[\]\*\\/\?]')
MAX_SHEET_NAME_LENGTH = 31

def sanitize_sheet_name(name: str) -> str:
    """
    Sanitizes a string to be a valid Excel sheet name.

    Removes invalid characters ([]*/\\?) and truncates to 31 characters.

    Args:
        name: The proposed sheet name.

    Returns:
        A sanitized, valid Excel sheet name.
    """
    if not isinstance(name, str):
        name = str(name) # Attempt to convert non-strings
    sanitized = INVALID_SHEET_NAME_CHARS.sub('', name)
    # Truncate if necessary
    if len(sanitized) > MAX_SHEET_NAME_LENGTH:
        # TODO: Add logging warning here when logger is available
        # logger = logging.getLogger("sdc_checker")
        # logger.warning(f"Sheet name '{name}' truncated to '{sanitized[:MAX_SHEET_NAME_LENGTH]}'.")
        sanitized = sanitized[:MAX_SHEET_NAME_LENGTH]
    # Handle potential empty string after sanitization
    if not sanitized:
        # logger.warning(f"Sheet name '{name}' became empty after sanitization, using default 'Sheet'.")
        sanitized = "Sheet" # Provide a default name
    return sanitized

# --- Header Sorting Key ---

# Define constants for sorting keys to improve readability
SORT_KEY_NORMAL = 0
SORT_KEY_MISSING = 1
SORT_KEY_AGGREGATED = 2

def get_sort_key(header_value: any) -> Tuple[int, any]:
    """
    Generates a sort key tuple for DataFrame headers (index or columns).

    Ensures 'Aggregated' appears last and 'Missing' appears second-to-last
    within peer groups during sorting. Handles various potential representations
    of missing values (None, np.nan, empty string).

    Args:
        header_value: The value of the header (index level or column level).

    Returns:
        A tuple (sort_priority, original_value) where sort_priority is
        an integer (0 for normal, 1 for missing, 2 for aggregated) and
        original_value is the header_value itself for secondary sorting.
    """
    # Check for 'Aggregated' first
    if isinstance(header_value, str) and header_value.strip().lower() == 'aggregated':
        return (SORT_KEY_AGGREGATED, header_value)

    # Check for various forms of missing/empty values
    # Using pd.isna() is robust for None, np.nan
    # Also check for empty strings explicitly
    if pd.isna(header_value) or header_value == '':
        # Represent missing consistently for sorting, e.g., using a specific string
        # or ensuring the sort priority handles it. Returning the original value
        # might lead to inconsistent sorting if None and '' are treated differently.
        # Let's return a consistent representation like 'Missing' for the secondary key.
        return (SORT_KEY_MISSING, 'Missing') # Use 'Missing' for consistent secondary sort

    # Otherwise, it's a normal value
    return (SORT_KEY_NORMAL, header_value)

# --- Logging Level Control ---
def set_module_log_level(module_name: str, level: int) -> None:
    """
    Sets the logging level for a specific sdc_checker submodule.

    This allows overriding the default package level for targeted debugging,
    without affecting handlers or propagation.

    Args:
        module_name: The simple name of the module (e.g., 'calculator', 'styler').
        level: The desired logging level (e.g., logging.DEBUG, logging.INFO).
    """
    if not isinstance(level, int):
        print(f"Warning: Invalid level '{level}' provided for module '{module_name}'. Level must be an integer (e.g., logging.DEBUG).")
        return

    full_module_name = f"sdc_checker.{module_name}"
    module_logger = logging.getLogger(full_module_name)
    module_logger.setLevel(level)
    # Optional: Log the change using the *parent* logger to avoid being filtered by the new level
    # logging.getLogger("sdc_checker").debug(f"Set log level for '{full_module_name}' to {logging.getLevelName(level)}")