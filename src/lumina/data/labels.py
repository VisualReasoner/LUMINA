from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from lumina.tasks.ad_continuum import assign_ad_continuum_label
from lumina.tasks.progression import DEFINITION as PROGRESSION_DEFINITION
from lumina.tasks.progression import assign_progression_label, normalize_dx_3way


@dataclass(frozen=True)
class LabelBuildConfig:
    max_cdr_gap_days: int = 180
    max_dx_gap_days: int = 365
    progression_horizon_months: int = PROGRESSION_DEFINITION.horizon_months


def _to_float(value: object) -> Optional[float]:
    try:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _nearest_row(subject_rows: pd.DataFrame, target_date: pd.Timestamp, date_col: str, max_days: int) -> Optional[pd.Series]:
    if subject_rows.empty or pd.isna(target_date):
        return None
    prior_rows = subject_rows[subject_rows[date_col] <= target_date].copy()
    if prior_rows.empty:
        return None
    deltas = (target_date - prior_rows[date_col]).dt.days
    idx = deltas.idxmin()
    if pd.isna(idx):
        return None
    best = deltas.loc[idx]
    if pd.isna(best) or int(best) > max_days:
        return None
    return prior_rows.loc[idx]


def _latest_prior_row(subject_rows: pd.DataFrame, target_date: pd.Timestamp, date_col: str) -> Optional[pd.Series]:
    if subject_rows.empty or pd.isna(target_date):
        return None
    prior_rows = subject_rows[subject_rows[date_col] <= target_date].copy()
    if prior_rows.empty:
        return None
    return prior_rows.sort_values(date_col).iloc[-1]


def _clinical_stage_from_cdr(cdglobal: object) -> str:
    global_score = _to_float(cdglobal)
    if global_score is None:
        return "UNLABELED"
    if global_score == 0.0:
        return "CU"
    if global_score >= 1.0:
        return "Dementia"
    if global_score == 0.5:
        return "MCI"
    return "UNLABELED"


def _amyloid_flag_from_annotation(value: object) -> str:
    numeric = _to_float(value)
    if numeric is None:
        return "unknown"
    if numeric >= 1.0:
        return "positive"
    if numeric <= 0.0:
        return "negative"
    return "unknown"


def build_label_table(
    *,
    visit_index_csv: Path,
    diagnosis_csv: Path,
    cdr_csv: Path,
    amyloid_csv: Path,
    config: LabelBuildConfig | None = None,
) -> tuple[pd.DataFrame, dict]:
    cfg = config or LabelBuildConfig()
    visits = pd.read_csv(visit_index_csv, low_memory=False)
    dx = pd.read_csv(diagnosis_csv, low_memory=False)
    cdr = pd.read_csv(cdr_csv, low_memory=False)
    amyloid = pd.read_csv(amyloid_csv, low_memory=False)

    required_visit_cols = {"subject_id", "visit_id", "visit_date"}
    missing_visit = sorted(required_visit_cols - set(visits.columns))
    if missing_visit:
        raise ValueError(f"Visit index missing required columns: {missing_visit}")

    visits = visits.copy()
    visits["visit_date"] = pd.to_datetime(visits["visit_date"], errors="coerce")
    dx = dx.copy()
    cdr = cdr.copy()
    amyloid = amyloid.copy()

    for frame, date_col in ((dx, "EXAMDATE"), (cdr, "VISDATE"), (amyloid, "SCANDATE")):
        if date_col not in frame.columns:
            raise ValueError(f"Missing required date column: {date_col}")
        frame[date_col] = pd.to_datetime(frame[date_col], errors="coerce")

    if "PTID" not in dx.columns or "DX" not in dx.columns:
        raise ValueError("Diagnosis CSV must contain PTID and DX columns.")
    if "PTID" not in cdr.columns or "CDGLOBAL" not in cdr.columns:
        raise ValueError("CDR CSV must contain PTID and CDGLOBAL columns.")
    if "PTID" not in amyloid.columns or "AMYLOID_STATUS" not in amyloid.columns:
        raise ValueError("Amyloid CSV must contain PTID and AMYLOID_STATUS columns.")

    dx_groups = {
        str(subject_id): group.sort_values("EXAMDATE").reset_index(drop=True)
        for subject_id, group in dx[dx["EXAMDATE"].notna()].groupby("PTID")
    }
    cdr_groups = {
        str(subject_id): group.sort_values("VISDATE").reset_index(drop=True)
        for subject_id, group in cdr[cdr["VISDATE"].notna()].groupby("PTID")
    }
    amyloid_groups = {
        str(subject_id): group.sort_values("SCANDATE").reset_index(drop=True)
        for subject_id, group in amyloid[amyloid["SCANDATE"].notna()].groupby("PTID")
    }

    output_rows: list[dict[str, object]] = []
    for row in visits.itertuples(index=False):
        subject_id = str(row.subject_id)
        visit_date = pd.Timestamp(row.visit_date) if not pd.isna(row.visit_date) else pd.NaT

        cdr_match = _nearest_row(cdr_groups.get(subject_id, pd.DataFrame()), visit_date, "VISDATE", cfg.max_cdr_gap_days)
        dx_match = _nearest_row(dx_groups.get(subject_id, pd.DataFrame()), visit_date, "EXAMDATE", cfg.max_dx_gap_days)
        amyloid_target_date = visit_date
        for candidate_column in ("amyloid_pet_date", "ab_pet_date"):
            candidate_value = getattr(row, candidate_column, None)
            candidate_date = pd.to_datetime(candidate_value, errors="coerce")
            if not pd.isna(candidate_date):
                amyloid_target_date = pd.Timestamp(candidate_date)
                break
        amyloid_match = _latest_prior_row(
            amyloid_groups.get(subject_id, pd.DataFrame()), amyloid_target_date, "SCANDATE"
        )

        clinical_stage = _clinical_stage_from_cdr(None if cdr_match is None else cdr_match.get("CDGLOBAL"))
        amyloid_flag = _amyloid_flag_from_annotation(None if amyloid_match is None else amyloid_match.get("AMYLOID_STATUS"))
        ad_label = assign_ad_continuum_label(clinical_stage, amyloid_flag)

        current_dx = normalize_dx_3way(None if dx_match is None else dx_match.get("DX"))
        future_dx_values: list[str] = []
        future_followup_count = 0
        future_worst_dx = "UNLABELED"
        future_last_date = None
        current_dx_date = None
        if dx_match is not None:
            current_dx_date = pd.Timestamp(dx_match["EXAMDATE"])
            horizon_end = visit_date + pd.DateOffset(months=cfg.progression_horizon_months)
            future_rows = dx_groups.get(subject_id, pd.DataFrame())
            future_rows = future_rows[
                (future_rows["EXAMDATE"] > visit_date) & (future_rows["EXAMDATE"] <= horizon_end)
            ].copy()
            if not future_rows.empty:
                future_dx_values = [normalize_dx_3way(value) for value in future_rows["DX"].tolist()]
                ranked = [dx_value for dx_value in future_dx_values if dx_value in {"CN", "MCI", "AD"}]
                future_followup_count = len(ranked)
                if ranked:
                    future_last_date = pd.Timestamp(future_rows["EXAMDATE"].max())
                    if "AD" in ranked:
                        future_worst_dx = "AD"
                    elif "MCI" in ranked:
                        future_worst_dx = "MCI"
                    else:
                        future_worst_dx = "CN"
        progression_label = assign_progression_label(current_dx, future_dx_values)

        output_rows.append(
            {
                "subject_id": subject_id,
                "visit_id": str(row.visit_id),
                "visit_date": visit_date,
                "clinical_stage_cdr": clinical_stage,
                "amyloid_flag": amyloid_flag,
                "amyloid_annotation_date": None if amyloid_match is None else amyloid_match.get("SCANDATE"),
                "ad_continuum_3way": ad_label,
                "current_dx_3way": current_dx,
                "current_dx_match_date": current_dx_date,
                "future_worst_dx_3way": future_worst_dx,
                "future_followup_count": future_followup_count,
                "future_followup_last_date": future_last_date,
                "progression_forecasting_3way": progression_label,
            }
        )

    table = pd.DataFrame(output_rows).sort_values(["subject_id", "visit_date"]).reset_index(drop=True)
    report = {
        "visit_rows": int(len(table)),
        "subjects": int(table["subject_id"].nunique()) if not table.empty else 0,
        "ad_continuum_counts": table["ad_continuum_3way"].value_counts(dropna=False).to_dict(),
        "progression_counts": table["progression_forecasting_3way"].value_counts(dropna=False).to_dict(),
        "ad_continuum_class_order": ["CU_nonAD", "Preclinical_AD", "Symptomatic_AD"],
        "progression_class_order": list(PROGRESSION_DEFINITION.class_order),
        "progression_horizon_months": int(cfg.progression_horizon_months),
        "max_cdr_gap_days": int(cfg.max_cdr_gap_days),
        "max_dx_gap_days": int(cfg.max_dx_gap_days),
    }
    return table, report
