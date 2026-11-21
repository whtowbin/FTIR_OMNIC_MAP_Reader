"""Hypothesis settings: HYPOTHESIS_PROFILE=ci runs 200 examples per property, the default "dev" runs 50."""

import os

from hypothesis import settings

settings.register_profile("ci", deadline=None, max_examples=200)  # deadline=None: file I/O time varies on CI runners
settings.register_profile("dev", deadline=None, max_examples=50)
settings.load_profile(os.getenv("HYPOTHESIS_PROFILE", "dev"))
