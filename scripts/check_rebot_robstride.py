#!/usr/bin/env python3
"""Passive seven-motor connectivity check for a reBot DevArm RobStride arm.

This script only reads RobStride mechPos/mechVel parameters. It never clears
faults, changes modes, enables motors, or sends a target.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from robots.rebot_robstride.config import load_config
from robots.rebot_robstride.driver import RebotRobstrideDriver
from robots.rebot_robstride.env_server import validate_socketcan


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    validate_socketcan(config.channel, config.bitrate)
    driver = RebotRobstrideDriver(config)
    try:
        state = driver.connect()
        print(json.dumps(state, indent=2, sort_keys=True))
    finally:
        driver.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
