"""Smoke test: the package imports and reports a version."""


def test_package_imports():
    import irap_screening

    assert irap_screening.__version__
