from __future__ import annotations

import pytest
import ray


@pytest.fixture(autouse=True)
def ray_context():
    if not ray.is_initialized():
        ray.init(local_mode=True, ignore_reinit_error=True, include_dashboard=False)
    yield
    if ray.is_initialized():
        ray.shutdown()
