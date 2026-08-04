import pytest

# Test modules that do not import Firedrake, directly or transitively
SMOKE_MODULES = {
    "test_2d_examples_docs",
    "test_3d_examples_docs",
    "test_dofs",
    "test_perms",
    "test_plotting",
    "test_polynomial_space",
    "test_sobolev_space",
}


def pytest_addoption(parser):
    parser.addoption(
        "--run-cleared",
        action="store_true",
        default=False,
        help="Run tests that require a cleared cache",
    )


def pytest_collection_modifyitems(config, items):
    for item in items:
        if item.path.stem in SMOKE_MODULES:
            item.add_marker(pytest.mark.smoke)
