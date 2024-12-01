from __future__ import annotations

import contextlib


@contextlib.contextmanager
def optional_import_error_message(pip_package_name: str, /):
    try:
        yield
    except ImportError as e:
        raise ImportError(
            f"The `{pip_package_name}` package is not installed. Please install it by running "
            f"`pip install {pip_package_name}`."
        ) from e
