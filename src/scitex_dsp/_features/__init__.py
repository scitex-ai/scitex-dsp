"""Feature-extraction primitives — canonical named scalar-feature vectors.

Currently ships the PAC-summary set (``pac_features``) plus its
self-documenting registry (``PAC_FEATURE_REGISTRY`` / ``feature_registry``).
"""

from ._pac_features import pac_features
from ._registry import PAC_FEATURE_REGISTRY, feature_registry

__all__ = [
    "pac_features",
    "PAC_FEATURE_REGISTRY",
    "feature_registry",
]
