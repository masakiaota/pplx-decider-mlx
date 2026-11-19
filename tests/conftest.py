"""Tests that need a model run only when DECIDER_MLX_TEST_MODELS is set to a comma-separated list of model names.
Their weights come from the Hugging Face cache like any other use (decider-mlx fetch first, or set HF_HUB_CACHE to a
cache that holds them; HF_HUB_OFFLINE=1 keeps the tests off the network). Without it they are skipped."""
import os

import pytest


def models_under_test():
    return [m for m in os.environ.get('DECIDER_MLX_TEST_MODELS', '').split(',') if m]


def pytest_generate_tests(metafunc):
    if 'model' in metafunc.fixturenames:
        names = models_under_test()
        metafunc.parametrize('model', names or [pytest.param(None, marks=pytest.mark.skip(reason='set DECIDER_MLX_TEST_MODELS'))],
                             scope='module')


_apps = {}


@pytest.fixture(scope='module')
def app(model):
    import decider_mlx
    if model not in _apps:
        _apps[model] = decider_mlx.load(model)
    return _apps[model]
