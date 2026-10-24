# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest

from rpent.utils import logging as run_logging


@pytest.fixture(autouse=True)
def isolate_run_logging(monkeypatch):
    """Restore process-wide logging after planner, CLI, or robot tests."""
    logger = run_logging.get_logger()
    handlers = list(logger.handlers)
    level, propagate = logger.level, logger.propagate
    monkeypatch.setattr(run_logging, "_log_initialized", run_logging._log_initialized)
    monkeypatch.setattr(run_logging, "_output_dir", run_logging._output_dir)
    try:
        yield
    finally:
        for handler in logger.handlers:
            if handler not in handlers:
                handler.close()
        logger.handlers[:] = handlers
        logger.setLevel(level)
        logger.propagate = propagate
