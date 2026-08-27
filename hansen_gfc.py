"""Hansen Global Forest Change asset resolution.

Hansen GFC is republished yearly (e.g. 2023_v1_11, 2025_v1_13). The Earth Engine
asset ID is therefore an input, not a constant baked into HansenHistorical.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

DEFAULT_HANSEN_GFC_ASSET = "UMD/hansen/global_forest_change_2025_v1_13"
DEFAULT_HANSEN_LOSSYEAR_MAX = 25


def _clean_asset(value: Any) -> Optional[str]:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _from_config(config: Optional[Dict[str, Any]]) -> Optional[str]:
    if not isinstance(config, dict):
        return None
    flat = _clean_asset(config.get("hansen_gfc_asset") or config.get("gfc_asset"))
    if flat:
        return flat
    nested = config.get("hansen")
    if isinstance(nested, dict):
        return _clean_asset(nested.get("gfc_asset") or nested.get("hansen_gfc_asset"))
    return None


def resolve_hansen_gfc_asset(
    argument: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    instance_config: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Resolve the Hansen GFC Earth Engine Image asset ID.

    Precedence: explicit argument > run config > ForestryCarbonARR init config > default.
    """
    for value in (_clean_asset(argument), _from_config(config), _from_config(instance_config)):
        if value:
            return value
    return DEFAULT_HANSEN_GFC_ASSET


def hansen_lossyear_max(gfc_asset: str) -> int:
    """lossyear band encoding: 1=2001 ... N=20NN. Parse N from the asset year."""
    match = re.search(r"global_forest_change_(\d{4})", gfc_asset or "")
    if match:
        return int(match.group(1)[2:4])
    return DEFAULT_HANSEN_LOSSYEAR_MAX
