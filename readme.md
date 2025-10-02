# SDC Checker

This code helps produce tabular data and conducts some statistical disclosure control (SDC) checks. The following SDC checks take place for each cell:

* A minimum primary threshold rule (e.g. minimum sample size).
* The p% dominance rule, where the total of the third to nth largest observation is at least p% of the largest observation.
* The 1, K dominance rule, where the largest observation must contribute less than K% of the cell total.

Checking for other forms of statistical disclosure, such as secondary disclosure and class disclosure, are not covered by this script.

This code is experimental so please verify your results.

## Functions

The `generate_data()` function produces a sheet with tabular data.

The `generate_sdc()` function creates two sheets:

1. A Pass-Fail sheet which indicates whether each cell passes (indicated with `P`) or fails (`F`). Any cells indicating `F` imply that these cells are disclosive and need to be addressed using appropriate SDC methods.
2. A detailed SDC sheet that shows all the working used to calculate the Pass-Fail sheet. For each cell, it includes the observation count, maximum value, 2nd maximum value, sum of the 3rd to nth largest observations, sum, value for the 1,K dominance rule and value for the p% dominance rule. Cells that fail the three SDC checks mentioned earlier are highlighted in yellow.

Both functions can handle tables with multi-level columns and rows.

## Config

`generate_data()` and `generate_sdc()` take a configuration dictionary. Example:

```
config = {
    "value_var": "var_of_interest",
    "row_var": ["industry"],
    "col_var": ["year","export_status"],
    "secondary_ref": "",
    "statistic": "sum",
    "pweight": "",
    "include_aggregated": True,
    "output": r"<full_filepath>\stata_1x2_sum.xlsx",
    "sdc_rules": {
        "Primary Threshold": 3,
        "Dominance (1,k)": 0.4375,
        "Dominance p%": 0.125
    }
}
```

Required fields are:

* `value_var` (string): the name of the variable of interest that makes up the cells of the output table. The underlying data must be numeric.
* `row_var` (list of strings): the variable names that make up the rows of the output table. Multiple variables can be specified in the list for multiple row levels.
* `col_var` (list of strings): the variable names that make up the columns of the output table. Multiple variables are permitted.
* `statistic` (string): the selected statistic applied to the data. Must be one of `"sum"`, `"count"` or `"median"`.
* `sdc_rules` (dictionary): Sets the rules to dictate which cells pass/fail in the Pass-Fail output sheet and conditional formatting.
  * `Primary Threshold` (numeric): The minimum primary threshold for SDC checks.
  * `Dominance (1,k)` (numeric): The `k` for the `1,k` dominance rule, specified as a number between 0 and 1. This is the proportion of a cell's total which the largest observation cannot exceed.
  * `Dominance p%` (numeric): The `p` for the `p%` dominance rule, specified as a number between 0 and 1. This checks the total of the third to nth largest observations is at least p% of the largest observation.

Note: `generate_data` currently requires `sdc_rules` to be specified despite not being used. This will be fixed in future versions.

Optional fields are:

* `pweight` (string): the name of the weighting variable.
* `secondary_ref` (string): Refers to a variable name and used in cases where multiple observations can count as a single entity during statistical disclosure checks. The variable should be a unique entity identifier. [1]
* `include_aggregated` (boolean): Setting to `true` includes subtotals for tables with multiple level columns/rows. Default is `true`.
* `output` (string): The output Excel file name and path. Has only been tested using the full filepath. Recommended to use raw string literals (prefix string with `r`) to avoid issues with backslashes in Windows environments.
* `sheet_name` (string): The output sheet name. Default for `generate_data()` is `Raw Output`. Default for the detailed sheet in `generate_sdc()` is `SDC` and the pass-fail sheet simply suffixes `Pass-Fail` to the detailed sheet name. If sheet name exceeds the Excel limit then it is truncated. [2]

[1] Where possible, it's recommended that you aggregat to the level of units that the SDC is conducted at before running this code. However, in some cases this is not possible. For example, your data output might be based on employee data but one set of SDC checks you are conducting apply at the business level. In this case, you would set `secondary_ref` to be the unique business identifier and the script will attempt to aggregate over the unique identifier before producing the `generate_sdc()` outputs.

[2] If you set a sheet name then you cannot run `generate_data()` and `generate_sdc()` on the same config file

Note that all variable names are case sensitive.

## How to use in Stata

### Loading data from an active Stata IDE

The [stata_to_df](https://github.com/jameskhblack/stata_to_df) package is designed to help load data from Stata IDE into SDC Checker.

To use Stata with SDC Checker, you will need to set:

* The location of the pystata package using PYSTATA_PATH. This is typically the Stata18 utilities folder located in Program Files.
* The edition of Stata you are using, e.g. `'mp'`.
* The location of the parent folder that includes the SDC Checker and stata_to_df folders, i.e. parent folder has folder `sdc_checker_python` which contains the scripts like `__init__.py`.

```stata
python:
import os
import sys

# Set Stata Initialization variables
os.environ["PYSTATA_PATH"] = r"<stata_path>/utilities" # Add full path to Stata18 utilities folder
os.environ["STATA_EDITION"] = 'mp'

# Import python scripts
sys.path.append(r"<python_script_folder>") # Location of stata_to_df script folder
sys.path.append(r"<python_script_folder>") # Location of sdc_checker script folder
sys.path.append(r"<python_library>") # Location of pydantic and any other requirements that are not installed
import pydantic
from stata_to_df import stata_to_df
from sdc_checker import generate_data, generate_sdc
end
```

### Creating data and SDC outputs using Stata

Once you have loaded the data using the method above, you can then specify config files and start creating output. The example below generates a table of data using `generate_data()` and then conducts SDC over the data using `generate_sdc()`.

```stata
python:
config = {
    "value_var": "var_of_interest",
    "row_var": ["industry"],
    "col_var": ["year","export_status"],
    "secondary_ref": "",
    "statistic": "sum",
    "pweight": "",
    "include_aggregated": True,
    "output": r"<full_filepath>\stata_1x2_sum.xlsx",
    "sdc_rules": {
        "Primary Threshold": 3,
        "Dominance (1,k)": 0.4375,
        "Dominance p%": 0.125
    }
}

df=stata_to_df(config,valuelabel=True)
generate_data(config,df)
generate_sdc(config,df)
end
```

## Notes

1x0 or 0x1 dimension tables are not currently supported. If you wish to use these, try creating a new variable with a single value and using that as the extra dimension.

### Weighted Medians

Weighted medians follow an interpolation method. Where there is an even number of observations in a group, the results can differ from other software which selects the next number up rather than interpolating between the two middle numbers.

Our approach should be consistent with the methods used by ONS Annual Survey of Hours and Earnings.

### Tested on

* Stata18 MP
* Python 3.11.9
* colorama 0.4.6
* numpy 2.2.3
* openpyxl 3.1.5
* pandas 2.2.3
* pydantic 2.11.1
