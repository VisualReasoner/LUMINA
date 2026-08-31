from __future__ import annotations

from pathlib import Path

import pandas as pd

from lumina.data.labels import LabelBuildConfig, build_label_table


def test_label_builder_carries_prior_amyloid_and_uses_post_target_horizon(tmp_path: Path) -> None:
    visits = pd.DataFrame(
        [
            {
                "subject_id": "s1",
                "visit_id": "s1_v1",
                "visit_date": "2020-01-01",
                "amyloid_pet_date": "2020-01-10",
            }
        ]
    )
    diagnosis = pd.DataFrame(
        [
            {"PTID": "s1", "EXAMDATE": "2019-12-15", "DX": "CN"},
            {"PTID": "s1", "EXAMDATE": "2022-12-20", "DX": "AD"},
        ]
    )
    cdr = pd.DataFrame([{"PTID": "s1", "VISDATE": "2019-12-20", "CDGLOBAL": 0.0}])
    amyloid = pd.DataFrame([{"PTID": "s1", "SCANDATE": "2018-01-01", "AMYLOID_STATUS": 1}])
    paths = {}
    for name, table in (("visits", visits), ("diagnosis", diagnosis), ("cdr", cdr), ("amyloid", amyloid)):
        path = tmp_path / f"{name}.csv"
        table.to_csv(path, index=False)
        paths[name] = path
    labels, _ = build_label_table(
        visit_index_csv=paths["visits"],
        diagnosis_csv=paths["diagnosis"],
        cdr_csv=paths["cdr"],
        amyloid_csv=paths["amyloid"],
        config=LabelBuildConfig(),
    )
    assert labels.iloc[0]["ad_continuum_3way"] == "Preclinical_AD"
    assert labels.iloc[0]["progression_forecasting_3way"] == "clear_progression"
