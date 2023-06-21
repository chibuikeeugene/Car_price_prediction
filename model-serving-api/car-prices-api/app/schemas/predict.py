from typing import Any, List, Optional

from pydantic import BaseModel
from regression_model.processing.validation import CarDataInputSchema


class PredictionResults(BaseModel):
    errors: Optional[Any]
    version: str
    predictions: Optional[List[float]]


class MultipleCarDataInputs(BaseModel):
    inputs: List[CarDataInputSchema]

    class Config:
        schema_extra = {
            "example": {
                "inputs": [
                    {
                        "make": "BMW",
                        "model": "1 Series M",
                        "year": 2011,
                        "engine_fuel_type": "premium unleaded (required)",
                        "engine_hp": 335.0,
                        "engine_cylinders": 6.0,
                        "transmission_type": "MANUAL",
                        "driven_wheels": "rear wheel drive",
                        "number_of_doors": 2.0,
                        "market_category": "Factory Tuner,Luxury,High-Performance",
                        "vehicle_size": "Compact",
                        "vehicle_style": "Coupe",
                        "highway_mpg": 26,
                        "city_mpg": 19,
                        "popularity": 3916,
                    }
                ]
            }
        }
