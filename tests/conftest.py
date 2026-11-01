"""Hooks for the whole suite."""


def pytest_collection_modifyitems(items):
    """Tests marked ``long`` first. The suite runs in four processes (pytest-xdist, work
    stealing): a long test started late would keep one process busy after the others finish."""
    # a stable sort: the rest keep their order
    items.sort(key=lambda item: item.get_closest_marker("long") is None)
