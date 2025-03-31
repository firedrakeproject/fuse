import os
import pytest

def pytest_addoption(parser):
    parser.addoption(
        "--run-cleared",
        action="store_true",
        default=False,
        help="Run tests that require a cleared cache",
    )

def _skip_test_dependency(dependency):
    """
    Returns whether to skip tests with a certain dependency.

    Usually, this will return True if the dependency is not available.
    However, on CI we never want to skip tests because we should
    test all functionality there, so if the environment variable
    FIREDRAKE_CI=1 then this will always return False.
    """
    skip = True

    if os.getenv("FUSE_CI") == "1":
        return not skip

    if dependency == "basix":
        try:
            import basix # noqa: F401
            del basix
            return not skip
        except ImportError:
            return skip

dependency_skip_markers_and_reasons = (
    ("basix", "skipbasix", "Basix not installed"),)

def pytest_configure(config):
    """Register an additional marker."""
    config.addinivalue_line(
        "markers",
        "skipbasix: mark as skipped unless Basix is installed"
    )

def pytest_collection_modifyitems(session, config, items):
    for item in items:
        for dep, marker, reason in dependency_skip_markers_and_reasons:
            if _skip_test_dependency(dep) and item.get_closest_marker(marker) is not None:
                item.add_marker(pytest.mark.skip(reason))
