import importlib
import sys
import warnings

import pytest


@pytest.mark.integration
@pytest.mark.parametrize(
    "module_name", [
        "core.event",
        "execution.broker",
        "data_pipeline.data_manager",
        "risk_management.stop_loss",
    ],
)
def test_legacy_shims_emit_deprecation(module_name):
    sys.modules.pop(module_name, None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        module = importlib.import_module(module_name)

    assert module is not None
    assert any(issubclass(item.category, DeprecationWarning) for item in caught)
