def pytest_addoption(parser):
    parser.addoption(
        "--run-cleared",
        action="store_true",
        default=False,
        help="Run tests that require a cleared cache",
    )
