import pytest
import os

@pytest.fixture(scope="session", autouse=True)
def set_env():
    os.environ["FIREDRAKE_USE_FUSE"] = "True"

def pytest_addoption(parser):
    parser.addoption(
        "--run-cleared",
        action="store_true",
        default=False,
        help="Run tests that require a cleared cache",
    )
