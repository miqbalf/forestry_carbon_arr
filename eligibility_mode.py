"""Eligibility run-mode selection: FCD-only vs ML training overlay."""

from typing import Any, Dict, List

try:
    from .exceptions import ForestryCarbonError
except ImportError:
    from exceptions import ForestryCarbonError

FCD_REQUIRED_KEYS: List[str] = ['AOI_path', 'OID_field_name', 'date_start_end']
ML_REQUIRED_KEYS: List[str] = [
    'AOI_path',
    'input_training',
    'OID_field_name',
    'date_start_end',
]


def has_training_input(config: Dict[str, Any]) -> bool:
    path = config.get('input_training')
    return isinstance(path, str) and bool(path.strip())


def resolve_eligibility_mode(is_only_fcd: bool, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Decide whether GEE eligibility uses FCD zones only or ML + training overlay.

    Args:
        is_only_fcd: If True, skip training/ML and use AssignClassZone.assigning_fcd_class.
        config: Eligibility config dict.

    Returns:
        dict with keys:
            mode: 'fcd' or 'ml'
            use_training: whether training points will be loaded and classified
            required_keys: config keys required for that mode
    """
    if is_only_fcd:
        return {
            'mode': 'fcd',
            'use_training': False,
            'required_keys': list(FCD_REQUIRED_KEYS),
        }

    if not has_training_input(config):
        raise ForestryCarbonError(
            "is_only_fcd=False requires config['input_training'] "
            "(path to training points shapefile). "
            "Pass is_only_fcd=True to run FCD-based eligibility without training."
        )

    return {
        'mode': 'ml',
        'use_training': True,
        'required_keys': list(ML_REQUIRED_KEYS),
    }
