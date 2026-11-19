"""Tests that need a model run only when its weights are available: set DECIDER_MLX_TEST_MODELS to a comma-separated
list of model names (the weights must be in the Hugging Face cache or fetchable). Without it they are skipped."""
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
