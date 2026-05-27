#!./env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-06-30 12:11:01 (ywatanabe)"
# /mnt/ssd/ripple-wm-code/scripts/externals/scitex/src/scitex/dsp/_time.py


import numpy as np

<<<<<<< Updated upstream

def time(start_sec, end_sec, fs):
    """Linearly spaced time samples between [start_sec, end_sec) at sample rate fs.

    Replaces the prior ``scitex.gen.float_linspace`` call so this module
    doesn't pull the umbrella ``scitex`` distribution at import time. The
    behaviour matches np.linspace(endpoint=False) which is also what the
    legacy float_linspace returned.
    """
    n = int((end_sec - start_sec) * fs)
    return np.linspace(start_sec, end_sec, n, endpoint=False)
=======
# import scitex  # lazy-loaded in __main__ block


def time(start_sec, end_sec, fs):
    # return np.linspace(start_sec, end_sec, (end_sec - start_sec) * fs)
    return __import__("scitex_gen").float_linspace(start_sec, end_sec, (end_sec - start_sec) * fs)
>>>>>>> Stashed changes


def main():
    out = time(10, 15, 256)
    print(out)


if __name__ == "__main__":
    import scitex
    import sys

    import matplotlib.pyplot as plt

    # Lazy-imported here so `import scitex_dsp` doesn't require the umbrella
    # at runtime — only the script entry-point uses session lifecycle.
    import scitex  # noqa: E402

    CONFIG, sys.stdout, sys.stderr, plt, CC = scitex.session.start(
        sys, plt, verbose=False
    )
    main()
    scitex.session.close(CONFIG, verbose=False, notify=False)

# EOF
