from a2o.scheduling.speculative_value import PDistribution, expected_value


def test_distribution_no_future_and_window():
    p = PDistribution(2)
    assert p.predict() == {}
    for x in [16, 24, 32]:
        p.observe(x)
    assert p.predict() == {24: .5, 32: .5}


def test_expected_value_and_readiness():
    costs = {24: (100, 200, True), 32: (200, 50, True)}
    assert not expected_value({24: .9, 32: .1}, costs, 30, 200)['prepare']
    assert expected_value({24: .1, 32: .9}, costs, 30, 200, weight=.5)['prepare']
    assert not expected_value({32: 1}, costs, 30, 0)['prepare']
    assert expected_value({48: 1}, costs, 30, 200)['reason'] == 'coverage_missing'
    assert not expected_value({32: 1}, {32: (200, 50, False)}, 30, 200)['prepare']


def test_incremental_value_is_bounded_by_prepare():
    costs = {24: (100, 200, True), 32: (1000, 1, True)}
    decision = expected_value({24: .25, 32: .75}, costs, 30, 200, weight=.5)
    assert decision['expected_gain_ms'] == 22.5
    assert decision['expected_net_ms'] == 7.5
    assert not expected_value({32: 1}, costs, 30, 200, weight=1)['prepare']
    assert expected_value({24: .99, 32: .01}, costs, 30, 200, weight=0)['prepare']
