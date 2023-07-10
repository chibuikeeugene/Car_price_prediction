from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, ValidationError

from regression_model.config.core import config


def drop_na_inputs(*, input_data: pd.DataFrame) -> pd.DataFrame:
    """check model inputs for na values and filter."""
    validated_data = input_data.copy()
    new_vars_with_na = [
        var
        for var in config.model_config.features
        if var not in config.model_config.categorical_vars_with_na_frequent
        and config.model_config.numerical_vars_with_na
        and validated_data[var].isnull().sum() > 0
    ]
    validated_data.dropna(subset=new_vars_with_na, inplace=True)

    return validated_data


def validate_inputs(*, input_data: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[dict]]:
    """check model inputs for unprocessable values."""

    relevant_data = input_data[config.model_config.features].copy()
    validated_data = drop_na_inputs(input_data=relevant_data)
    errors = None

    try:
        # replace numpy nans so that pydantic can validate input
        MultipleCarDataInputs(
            inputs=validated_data.replace({np.nan: None}).to_dict(orient="records")
        )
    except ValidationError as error:
        errors = error.json()

    return validated_data, errors


class CarDataInputSchema(BaseModel):
    make: Optional[str]
    model: Optional[str]
    year: Optional[int]
    engine_fuel_type: Optional[str]
    engine_hp: Optional[float]
    engine_cylinders: Optional[str]
    transmission_type: Optional[str]
    driven_wheels: Optional[str]
    number_of_doors: Optional[str]
    market_category: Optional[str]
    vehicle_size: Optional[str]
    vehicle_style: Optional[str]
    highway_mpg: Optional[int]
    city_mpg: Optional[int]
    popularity: Optional[int]


class MultipleCarDataInputs(BaseModel):
    inputs: List[CarDataInputSchema]
