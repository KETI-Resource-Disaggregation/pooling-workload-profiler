"""Exp_12-A: Trace Analyzer 정적 프로파일링 추정 엔진 코어 (Track 4-1).

K8s Pod annotation(keti.re.kr/*)만으로 워크로드 자원 요구를 계산식 기반으로
추정하고, controller가 소비할 초기값(SM/Quota/토큰예산)을 산출한다.

- 모든 출력은 "초기 추정값(source=static_estimate)"이며 Exp_12-B 실측 보정 대상.
- 근거 식이 없는 항목은 미추정(unestimated)으로 표기한다.
- controller 실적용(POST /register)은 하지 않는다 — 출력 형식과 매핑만 정의.
"""
from .schema import parse_annotations, WorkloadSpec, ANNOTATION_PREFIX
from .estimator import estimate
from .output_schema import validate_output, OUTPUT_SCHEMA_VERSION

__all__ = [
    "parse_annotations", "WorkloadSpec", "ANNOTATION_PREFIX",
    "estimate", "validate_output", "OUTPUT_SCHEMA_VERSION",
]
