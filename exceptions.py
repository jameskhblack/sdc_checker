"""Custom exception classes for the SDC Checker application."""

class SDCCheckerBaseError(Exception):
    """Base class for all custom exceptions in the SDC Checker application."""
    pass

class ConfigValidationError(SDCCheckerBaseError):
    """Exception raised for errors during configuration validation."""
    pass

class DataLoaderError(SDCCheckerBaseError):
    """Exception raised for errors during data loading (e.g., pystata issues, missing variables)."""
    pass

class CalculationError(SDCCheckerBaseError):
    """Exception raised for errors during core calculations (pivoting, aggregation, statistics)."""
    pass

class SDCLogicError(SDCCheckerBaseError):
    """Exception raised for errors during SDC indicator calculation or pass/fail logic."""
    pass

class ExcelWriterError(SDCCheckerBaseError):
    """Exception raised for errors during Excel file writing (e.g., I/O errors, openpyxl issues)."""
    pass

class StylingError(SDCCheckerBaseError):
    """Exception raised for errors during Excel styling application."""
    pass