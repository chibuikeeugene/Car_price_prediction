import typing as t
from pathlib import Path

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from regression_model import __version__ as _version
from regression_model.config.core import DATASET_DIR, TRAINED_MODEL_DIR, config


def load_dataset(*, file_name: str) -> pd.DataFrame:
    dataframe = pd.read_csv(Path(f"{DATASET_DIR}/{file_name}"))

    # remove the spaces and format the space in each title
    dataframe.columns = dataframe.columns.str.lower().str.replace(" ", "_")

    # Let's create our cat vars which will enable us to
    # cast some features to categorical type subsequently
    cat_var = [var for var in dataframe.columns if dataframe[var].dtype == "object"]

    # remove the spaces and format the case of each column values

    for col in cat_var:
        try:
            if dataframe[col].dtype == "object":
                dataframe[col] = dataframe[col].str.lower().str.replace(" ", "_")
            else:
                dataframe[col] = dataframe[col]
        except TypeError:
            print("wrong data type")

    # replace na or nan with 0 in the number_of_doors variable
    dataframe["number_of_doors"] = dataframe["number_of_doors"].fillna(0)

    # number_of_doors is in actual sense supposed to be a cat var,
    # so we first round the decimal to whole number.
    # also we round the values of engine_cylinder to whole number
    for i in dataframe.index:
        dataframe["number_of_doors"] = round(dataframe["number_of_doors"][i])
        dataframe["engine_cylinders"] = round(dataframe["engine_cylinders"][i])

    # update the data type for number of doors,
    # because in reality it should be a categorical variable and not numerical
    dataframe["number_of_doors"] = dataframe["number_of_doors"].astype("object")
    dataframe["engine_cylinders"] = dataframe["engine_cylinders"].astype("object")

    return dataframe


def save_pipeline(*, pipeline_to_persist: Pipeline) -> None:
    """Persist the pipeline.
    Saves the versioned model, and overwrites any previous
    saved models. This ensures that when the package is
    published, there is only one trained model that can be
    called, and we know exactly how it was built.
    """

    # prepare the versioned save file name
    save_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
    save_path = TRAINED_MODEL_DIR / save_file_name

    remove_old_pipeline(files_to_keep=[save_file_name])
    joblib.dump(pipeline_to_persist, save_path)


def load_pipeline(*, file_name: str) -> Pipeline:
    """load a persisted pipeline."""
    file_path = TRAINED_MODEL_DIR / file_name
    trained_model = joblib.load(filename=file_path)
    return trained_model


def remove_old_pipeline(*, files_to_keep: t.List[str]) -> None:
    """
    Remove old model pipelines.
    This is to ensure there is a simple one-to-one
    mapping between the package version and the model
    version to be imported and used by other applications.
    """

    do_not_delete = files_to_keep + ["__init__.py"]
    for model_file in TRAINED_MODEL_DIR.iterdir():
        if model_file.name not in do_not_delete:
            model_file.unlink()
