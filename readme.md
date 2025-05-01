# SDC Checker

Two packages are included:
* `stata_to_df` extracts the Stata data for variables value_var, row_var, col_var, pweight and secondary_ref specified in the config and creates a dataframe.
    * config: A dictionary specifying table characteristics. See below for an example.
    * 
* `sdc_checker_python` 


## Config
Add an example here
And say what is optional.
Mention that "sum", "median" and "count" are the only supported statistics.

* 1x0 or 0x1 dimension tables are not supported. If you wish to use these, perhaps try creating a new variable with a single value and using that as the extra dimension.


## How to use in Stata
### Loading data from an active Stata IDE
You will need to set:
* The location of the pystata package using PYSTATA_PATH. This is typically the Stata18 utilities folder, in Program Files.
* The edition of Stata you are using, e.g. `'mp'`.
* The location of the parent folder that includes the scripts, i.e. parent folder has folder `sdc_checker_python` which contains the scripts like `__init__.py`.

```stata
python:
import os
import sys

# Set Stata Initialization variables
os.environ["PYSTATA_PATH"] = r"<insert_full_filepath>" # Add full path to Stata18 utilities folder
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

### Creating SDC outputs using Stata
Once you have loaded the data using the method above, you can then specify config files and start creating output.

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