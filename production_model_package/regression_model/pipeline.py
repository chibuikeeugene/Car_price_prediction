# for saving the pipeline
# import joblib
from feature_engine.encoding import OrdinalEncoder, RareLabelEncoder

# packages from feature-engine
from feature_engine.imputation import (
    CategoricalImputer,
    MeanMedianImputer,
)  # AddMissingIndicator,
from feature_engine.selection import DropFeatures

# packages from scikit learn
from sklearn.ensemble import AdaBoostRegressor as abr
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from regression_model.config.core import config
from regression_model.processing import feature as f

# from feature_engine.transformation import YeoJohnsonTransformer
# from feature_engine.wrappers import SklearnTransformerWrapper


msrp_pipeline = Pipeline(
    [
        # ==== IMPUTATION ====
        (
            "frequent_imputation",
            CategoricalImputer(
                imputation_method="frequent",
                variables=config.model_config.categorical_vars_with_na_frequent,
            ),
        ),
        # (
        # 'missing_indicator',
        # AddMissingIndicator(
        #       variables=config.model_config.numerical_vars_with_na)
        # ),
        (
            "median_imputation",
            MeanMedianImputer(
                imputation_method="median",
                variables=config.model_config.numerical_vars_with_na,
            ),
        ),
        # ==== TEMPORAL VARIABLE - CREATING NEW CAR AGE VAR ====
        ("elapsed_time", f.CarAge(variables=config.model_config.temporal_var)),
        (
            "drop_features",
            DropFeatures(features_to_drop=[config.model_config.dropped_var]),
        ),
        # === CATEGORIAL ENCODING ====
        (
            "rare_label_encoder",
            RareLabelEncoder(
                tol=0.01, n_categories=1, variables=config.model_config.categorical_var
            ),
        ),
        # encoding the categorical and discrete variables using the target mean
        (
            "categorical_encoder",
            OrdinalEncoder(
                encoding_method="ordered", variables=config.model_config.categorical_var
            ),
        ),
        # scaling our feature parameters
        ("scalar", MinMaxScaler()),
        # final estimator
        (
            "abr",
            abr(
                random_state=config.model_config.random_state,
                n_estimators=config.model_config.n_estimators,
            ),
        ),
    ]
)
