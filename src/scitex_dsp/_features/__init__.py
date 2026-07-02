"""Feature-extraction primitives — canonical named scalar-feature vectors.

Ships:

- the PAC-summary set (``pac_features``) plus its self-documenting registry
  (``PAC_FEATURE_REGISTRY`` / ``feature_registry``); and
- the multi-backend facade (``extract_all``) that emits a single flat named
  vector over many engines (pac + catch22), with ``extract_all_registry`` and
  ``AVAILABLE_BACKENDS`` for provenance/introspection.
"""

from ._extract_all import (
    AVAILABLE_BACKENDS,
    extract_all,
    extract_all_registry,
)
from ._pac_features import pac_features
from ._registry import PAC_FEATURE_REGISTRY, feature_registry

__all__ = [
    "pac_features",
    "PAC_FEATURE_REGISTRY",
    "feature_registry",
    "extract_all",
    "extract_all_registry",
    "AVAILABLE_BACKENDS",
]
