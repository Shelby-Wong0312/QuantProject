"""Deprecated shim; import :class:`LSTMPricePredictor` from :mod:`quantproject.models.ml_models.lstm_predictor`."""

import warnings

from .lstm_predictor import LSTMPricePredictor

warnings.warn(
    "`quantproject.models.ml_models.lstm_price_predictor` is deprecated. Import from `quantproject.models.ml_models.lstm_predictor` instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["LSTMPricePredictor"]
