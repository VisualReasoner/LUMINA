from lumina.schemas.states import LocalComparison, ModalityObservation, ReferenceCalibration


def test_string_false_is_not_treated_as_present() -> None:
    observation = ModalityObservation.from_dict("Image", {"present": "false"})
    assert observation.present is False


def test_invalid_structured_choices_fall_back_to_uncertainty() -> None:
    calibration = ReferenceCalibration.from_dict(
        "Image",
        "reference",
        {"direction": "unsupported", "magnitude": "unsupported"},
    )
    comparison = LocalComparison.from_dict(
        "Image",
        "V1",
        "2020-01-01",
        365,
        {"direction": "unsupported", "magnitude": "unsupported", "comparison_quality": "unsupported"},
    )
    assert calibration.direction == "uncertain"
    assert calibration.magnitude == "uncertain"
    assert comparison.direction == "uncertain"
    assert comparison.magnitude == "uncertain"
    assert comparison.comparison_quality == "unknown"
