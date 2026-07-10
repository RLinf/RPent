"""Optional dashboard layer for live-monitoring a RPent run.

Opt in via ``python cli/main.py --dashboard``; never imported on the normal
CLI path.
"""
from rpent.dashboard.server import DashboardServer
from rpent.dashboard.state import State

__all__ = ["DashboardServer", "State"]
