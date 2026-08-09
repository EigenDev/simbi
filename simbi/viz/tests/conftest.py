# =============================================================================
# conftest.py
#
# pyplot keeps every figure it creates alive until something closes it, so a
# suite that builds one per test accumulates them until matplotlib warns at
# twenty and the process holds the memory of all of them. the gates here drive
# the real rendering path, which means they create figures by design.
#
# closing them is done here, once, rather than in each test: a test that forgets
# leaks silently, and the leak surfaces as a warning attributed to whichever test
# happened to cross the threshold rather than to the one that leaked. an axes
# object keeps its state after its figure is closed, so assertions that run after
# the teardown of a previous test are unaffected.
# =============================================================================
import matplotlib
import pytest

matplotlib.use("Agg")


@pytest.fixture(autouse=True)
def _close_figures():
    """close every pyplot figure a test leaves behind."""
    yield
    import matplotlib.pyplot as plt

    plt.close("all")
