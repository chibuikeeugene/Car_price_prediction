from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, ValidationError

from regression_model.config.core import config


def drop_na_inputs(*, input_data: pd.DataFrame) -> pd.DataFrame:
    """check model inputs for na values and filter."""
    validated_data = input_data.copy()
    new_vars_with_na = [
        var for var in config.model_config.features if var not in 
        config.model_config.categorical_vars_with_na_frequent
        and config.model_config.numerical_vars_with_na
        and validated_data[var].isnull().sum() > 0
    ]
    validated_data.dropna(subset=new_vars_with_na, inplace=True)

    return validated_data


def validate_inputs(*, input_data: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[dict]]:
    """check model inputs for unprocessable values."""

    # remove the spaces and format the space in each title
    input_data.columns = input_data.columns.str.lower().str.replace(' ', '_')

    #Let's create our cat vars which will enable us to cast some features to categorical type subsequently
    cat_var = [var for var in input_data.columns if input_data[var].dtype == 'object']

    #remove the spaces and format the case of each column values

    for col in cat_var:
        try:
            if input_data[col].dtype == 'object':
                input_data[col] = input_data[col].str.lower().str.replace(' ', '_')
            else:
                input_data[col] = input_data[col]
        except TypeError:
            print('wrong data type')

    #replace na or nan with 0 in the number_of_doors variable
    input_data['number_of_doors'] = input_data['number_of_doors'].fillna(0)


    #number_of_doors is in actual sense supposed to be a cat var, so we first round the decimal to whole number.
    #also we round the values of engine_cylinder to whole number
    for i in input_data.index:
        input_data['number_of_doors'] = round(input_data['number_of_doors'][i])
        input_data['engine_cylinders'] = round(input_data['engine_cylinders'][i])

    #update the data type for number of doors, because in reality it should be a categorical variable and not numerical
    input_data['number_of_doors'] = input_data['number_of_doors'].astype('object')
    input_data['engine_cylinders'] = input_data['engine_cylinders'].astype('object')


    relevant_data = input_data[config.model_config.features].copy()
    validated_data = drop_na_inputs(input_data=relevant_data)
    errors =None

    try:
        # replace numpy nans so that pydantic can validate input
        MultipleCarDataInputs(
            inputs=validated_data.replace({np.nan: None}).to_dict(orient="records")
        )
    except ValidationError as error:
        errors = error.json()
    
    return validated_data, errors



class CarDataInputSchema(BaseModel):
    make:              Optional[str]
    model:             Optional[str]
    year:              Optional[int]
    engine_fuel_type:  Optional[str]
    engine_hp:         Optional[float]
    engine_cylinders:  Optional[str]
    transmission_type: Optional[str]
    driven_wheels:     Optional[str]
    number_of_doors:   Optional[str]
    market_category:   Optional[str]
    vehicle_size:      Optional[str]
    vehicle_style:     Optional[str]
    highway_mpg:       Optional[int]
    city_mpg:          Optional[int]
    popularity:        Optional[int]
    msrp:              Optional[int]


class MultipleCarDataInputs(BaseModel):
    inputs: List[CarDataInputSchema]

