from quantproject.models.ml_models import LSTMPricePredictor

__all__ = ["StrategyLSTMPredictor"]

class StrategyLSTMPredictor(LSTMPricePredictor):
    """Strategy-level wrapper around the canonical LSTMPricePredictor."""
    pass
