#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-11-05 01:03:47 (ywatanabe)"
# File: ./scitex_repo/src/scitex/dsp/_ensure_3d.py

import numpy as np
import torch
from scitex_decorators import signal_fn


@signal_fn
def ensure_3d(x):
    # Coerce list/tuple input — signal_fn doesn't always reach lists deeply
    if isinstance(x, (list, tuple)):
        x = torch.as_tensor(np.asarray(x))
    if x.ndim == 1:  # assumes (seq_len,)
        x = x.unsqueeze(0).unsqueeze(0)
    elif x.ndim == 2:  # assumes (batch_size, seq_len)
        x = x.unsqueeze(1)
    return x


# EOF
