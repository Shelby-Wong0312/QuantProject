from rl3.broker.simulator import BrokerParams, BrokerSim


def test_broker_sim_costs_with_cap():
    params = BrokerParams(
        commission_bps=2.0,
        slippage_alpha=0.10,
        slippage_beta=0.30,
        participation_cap=0.05,
    )
    broker = BrokerSim(params)
    cost = broker.execute(px=100.0, dweight=0.10, nav=1_000_000.0)

    notional = 0.10 * 1_000_000.0
    cap_notional = 0.05 * 1_000_000.0
    trade_notional = min(notional, cap_notional)
    impact_rate = (params.slippage_alpha + params.slippage_beta * abs(0.10)) / 1e4
    expected_impact = trade_notional * impact_rate
    expected_commission = trade_notional * (params.commission_bps / 1e4)

    assert abs(cost - (expected_commission + expected_impact)) < 1e-12


def test_broker_sim_zero_inputs():
    broker = BrokerSim(BrokerParams())
    assert broker.execute(px=0.0, dweight=0.5, nav=1_000_000.0) == 0.0
    assert broker.execute(px=100.0, dweight=0.0, nav=1_000_000.0) == 0.0
    assert broker.execute(px=100.0, dweight=0.2, nav=0.0) == 0.0
