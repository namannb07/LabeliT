def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "jetson: requires Jetson hardware with TensorRT (skipped by default)",
    )
