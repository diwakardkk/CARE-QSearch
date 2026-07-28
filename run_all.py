from __future__ import annotations

from src.pandas_compat import configure_pandas

configure_pandas()

from src.run_experiments import main


if __name__ == "__main__":
    main()
