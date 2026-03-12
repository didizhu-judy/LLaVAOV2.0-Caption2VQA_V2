from __future__ import annotations

import pytest
import ray


_RAY_AVAILABLE = True


@pytest.fixture(autouse=True)
def ray_context():
    global _RAY_AVAILABLE
    if not ray.is_initialized():
        try:
            ray.init(local_mode=True, ignore_reinit_error=True, include_dashboard=False)
            _RAY_AVAILABLE = True
        except PermissionError:
            _RAY_AVAILABLE = False
    yield
    if _RAY_AVAILABLE and ray.is_initialized():
        ray.shutdown()


@pytest.fixture
def ray_available() -> bool:
    return _RAY_AVAILABLE
