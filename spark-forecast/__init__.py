# -*- coding: utf-8 -*-
"""Top level package for spark_forecast."""

from spark_forecast.logger import configure_logging
from spark_forecast.settings import init_settings

SETTINGS = init_settings()

logger = configure_logging("spark_forecast", SETTINGS, kidnap_loggers=True)
