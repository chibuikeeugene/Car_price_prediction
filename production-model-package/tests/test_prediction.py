import math

import numpy as np

from regression_model.predict import make_prediction

def test_make_prediction(sample_input_data):
    # Given
    expected_first_prediction_value = 91898.71122988
    expected_no_prediction = 3574


    # When
    result = make_prediction(input_data=sample_input_data)

    #Then
    predictions = result.get('predictions')
    assert isinstance(predictions, list)
    assert isinstance(predictions[0], np.int)
    assert result.get('errors') is None
    assert len(predictions) == expected_no_prediction
    assert math.isclose(predictions[0], expected_first_prediction_value, abs_tol=1000)