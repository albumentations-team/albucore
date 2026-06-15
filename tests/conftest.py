from __future__ import annotations

from hypothesis import HealthCheck, settings


settings.register_profile(
    "ci-fast",
    max_examples=20,
    deadline=None,
    suppress_health_check=(HealthCheck.too_slow,),
)
settings.register_profile(
    "nightly",
    max_examples=100,
    deadline=None,
    suppress_health_check=(HealthCheck.too_slow,),
)
settings.register_profile(
    "release",
    max_examples=200,
    deadline=None,
    suppress_health_check=(HealthCheck.too_slow,),
)
