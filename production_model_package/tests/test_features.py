from regression_model.config.core import config
from regression_model.processing.feature import CarAge


def test_Car_Age(sample_input_data):
    # Given
    transformer = CarAge(variables=config.model_config.temporal_var)
    assert sample_input_data["year"].iat[0] == 2017

    # When
    subject = transformer.fit_transform(sample_input_data)

    # Then
    assert subject["age"].iat[0] == 0
