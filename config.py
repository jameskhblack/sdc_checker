"""Configuration validation using Pydantic."""

from typing import List, Dict, Optional, Literal, Any
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from .exceptions import ConfigValidationError
from .utils import sanitize_sheet_name # Import for sheet name validation

# Define the structure for SDC rules
class SDCRules(BaseModel):
    """Pydantic model for SDC rules configuration."""
    primary_threshold: int = Field(..., alias='Primary Threshold', gt=0) # Must be positive integer
    dominance_1k: float = Field(..., alias='Dominance (1,k)', ge=0, le=1) # Must be between 0 and 1
    dominance_p_percent: float = Field(..., alias='Dominance p%', ge=0, le=1) # Must be between 0 and 1

# Define the main configuration model
class ConfigModel(BaseModel):
    """Pydantic model for the main application configuration."""
    # Required fields
    row_var: List[str] = Field(..., min_length=1)
    col_var: List[str] = Field(..., min_length=1)
    value_var: str
    statistic: Literal["sum", "median", "count"]
    sdc_rules: SDCRules
    output: str # Basic validation, more complex path validation might be OS-dependent
    
    # Make sheet_name optional with default None
    sheet_name: Optional[str] = None

    # Optional fields with defaults
    pweight: Optional[str] = None
    secondary_ref: Optional[str] = None
    include_aggregated: bool = True
    # Default order matches the example if not provided
    desired_order: Optional[List[str]] = Field(default=[
        'Count',
        'Max',
        '2nd Max',
        'Sum of 3rd to nth',
        'Total',
        'Dominance (1,k) check',
        'Dominance p% check'
    ])

    # --- Model Validators ---

    @field_validator('sheet_name')
    @classmethod
    def validate_and_sanitize_sheet_name(cls, v: str) -> Optional[str]:
        """Validate and sanitize the sheet name if provided."""
        if v is None:
            return None
            
        sanitized = sanitize_sheet_name(v)
        if not sanitized: # Should be handled by sanitize_sheet_name, but double-check
             raise ValueError("Sheet name cannot be empty after sanitization.")
        # Note: We sanitize here, but the final truncation might depend on suffixes
        # added later (e.g., " Pass-Fail"). The ExcelWriter should handle final checks.
        return sanitized # Return the initially sanitized version

    @model_validator(mode='after')
    def check_variable_names_distinct(self) -> 'ConfigModel':
        """Ensure core variable names do not overlap."""
        core_vars = {self.value_var}
        if self.pweight:
            core_vars.add(self.pweight)
        if self.secondary_ref:
            core_vars.add(self.secondary_ref)

        all_index_vars = set(self.row_var + self.col_var)

        overlap = core_vars.intersection(all_index_vars)
        if overlap:
            raise ValueError(f"Overlap detected between index variables (row_var, col_var) and core variables (value_var, pweight, secondary_ref): {overlap}")

        # Check for duplicates within row_var or col_var themselves
        if len(self.row_var) != len(set(self.row_var)):
            raise ValueError(f"Duplicate variable names found within row_var: {self.row_var}")
        if len(self.col_var) != len(set(self.col_var)):
            raise ValueError(f"Duplicate variable names found within col_var: {self.col_var}")

        return self

    @model_validator(mode='after')
    def check_sdc_order_for_median(self) -> 'ConfigModel':
        """
        Ensure desired_order only contains 'Count' if statistic is 'median'.
        If desired_order is not provided for median, default it to ['Count'].
        If desired_order IS provided for median, validate it only contains 'Count'.
        """
        # Get the default value defined in the model field
        default_order = ConfigModel.model_fields['desired_order'].default

        if self.statistic == "median":
            allowed_order = {'Count'}
            # Check if the current desired_order is different from the default
            # This implies it was explicitly provided by the user.
            if self.desired_order != default_order:
                current_order_set = set(self.desired_order) if self.desired_order else set()
                if not current_order_set.issubset(allowed_order):
                    raise ValueError("If statistic is 'median', desired_order can only contain 'Count'.")
                # If explicitly provided and valid (i.e., ['Count']), keep it.
                # No need to reset here as it must be ['Count'] to pass the check.
            else:
                # If desired_order was not provided (or provided as the default),
                # set it to ['Count'] for median statistic.
                self.desired_order = ['Count']

        elif self.statistic == "sum":
            # For 'sum', ensure the default is set if None was somehow passed
            if self.desired_order is None:
                self.desired_order = default_order
            # Could add validation here to ensure all expected 'sum' indicators are present if needed
            pass
        return self


# --- Function to set default sheet name ---

def set_default_sheet_name(config: ConfigModel, function_name: str) -> ConfigModel:
    """
    Sets a default sheet name based on the calling function if none was provided.
    
    Args:
        config: The configuration model
        function_name: The name of the function using the config
        
    Returns:
        Config with appropriate default sheet name set if needed
    """
    # If sheet_name is None or empty string, set a default
    if config.sheet_name is None or config.sheet_name == "":
        # Dictionary mapping function names to default sheet names
        default_names = {
            "generate_data": "Raw Output",
            "generate_sdc": "SDC"
        }
        
        # Set default sheet name based on function
        default_name = default_names.get(function_name, "Output")  # Fallback to "Output" if function not recognized
        
        # Set the sheet_name directly on the config object
        # This preserves all other fields and avoids re-validation issues
        config.sheet_name = default_name
    
    return config


# --- Validation Function ---

def validate_config(config_dict: Dict[str, Any]) -> ConfigModel:
    """
    Validates the raw configuration dictionary using the Pydantic model.

    Args:
        config_dict: The raw configuration dictionary.

    Returns:
        A validated ConfigModel instance.

    Raises:
        ConfigValidationError: If validation fails.
    """
    try:
        # Handle potential alias usage for sdc_rules keys
        if 'sdc_rules' in config_dict and isinstance(config_dict['sdc_rules'], dict):
            rules = config_dict['sdc_rules']
            # Map potential user-facing keys to internal Pydantic field names if needed
            # Pydantic handles alias mapping automatically if alias is set in Field
            pass # Pydantic should handle aliases defined in SDCRules

        validated_config = ConfigModel.model_validate(config_dict)
        return validated_config
    except ValidationError as e:
        # Raise a custom exception with a more user-friendly message potentially
        error_messages = "\n".join([f"- {err['loc']}: {err['msg']}" for err in e.errors()])
        raise ConfigValidationError(f"Configuration validation failed:\n{error_messages}") from e
    except Exception as e: # Catch other potential errors during validation
        raise ConfigValidationError(f"An unexpected error occurred during configuration validation: {e}") from e