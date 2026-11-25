import importlib.util
import os
import pytest

# Test modules that do not import Firedrake, directly or transitively
SMOKE_MODULES = {
    "test_2d_examples_docs",
    "test_3d_examples_docs",
    "test_dofs",
    "test_perms",
    "test_plotting",
    "test_polynomial_space",
    "test_interpolation_space",
    "test_group_utils",
    "test_kernel_actions",
    "test_generators",
}


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
    FUSE_CI=1 then this will always return False.
    """
    if os.getenv("FUSE_CI") == "1":
        return False
    return importlib.util.find_spec(dependency) is None


dependency_skip_markers_and_reasons = (
    ("basix", "skipbasix", "Basix not installed"),)


def pytest_configure(config):
    """Register the dependency markers."""
    for dependency, marker, _ in dependency_skip_markers_and_reasons:
        config.addinivalue_line("markers", f"{marker}: mark as skipped unless {dependency} is installed")


def pytest_collection_modifyitems(session, config, items):
    for item in items:
        if item.path.stem in SMOKE_MODULES:
            item.add_marker(pytest.mark.smoke)
        for dep, marker, reason in dependency_skip_markers_and_reasons:
            if _skip_test_dependency(dep) and item.get_closest_marker(marker) is not None:
                item.add_marker(pytest.mark.skip(reason))
