from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


ADContinuumLabel = Literal["CU_nonAD", "Preclinical_AD", "Symptomatic_AD", "UNLABELED"]
ClinicalStage = Literal["CU", "MCI", "Dementia", "UNLABELED"]
AmyloidFlag = Literal["positive", "negative", "unknown"]


@dataclass(frozen=True)
class ADContinuumDefinition:
    task_name: str = "ad_continuum"
    label_column: str = "ad_continuum_3way"
    class_order: tuple[str, str, str] = ("CU_nonAD", "Preclinical_AD", "Symptomatic_AD")
    summary: str = (
        "Three-way AD-continuum label derived from current-visit CDR stage together with amyloid positivity."
    )


DEFINITION = ADContinuumDefinition()


def normalize_clinical_stage(value: object) -> str:
    text = str(value or "").strip()
    if text in {"CU", "MCI", "Dementia"}:
        return text
    return "UNLABELED"


def normalize_amyloid_flag(value: object) -> str:
    text = str(value or "").strip().lower()
    if text in {"positive", "negative"}:
        return text
    return "unknown"


def assign_ad_continuum_label(clinical_stage: object, amyloid_flag: object) -> ADContinuumLabel:
    stage = normalize_clinical_stage(clinical_stage)
    amyloid = normalize_amyloid_flag(amyloid_flag)

    if stage == "CU" and amyloid == "negative":
        return "CU_nonAD"
    if stage == "CU" and amyloid == "positive":
        return "Preclinical_AD"
    if stage in {"MCI", "Dementia"} and amyloid == "positive":
        return "Symptomatic_AD"
    return "UNLABELED"
