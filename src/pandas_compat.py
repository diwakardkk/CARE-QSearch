from __future__ import annotations


def configure_pandas() -> None:
    import pandas as pd

    try:
        pd.options.future.infer_string = False
    except Exception:
        pass
    try:
        pd.options.mode.string_storage = "python"
    except Exception:
        pass
