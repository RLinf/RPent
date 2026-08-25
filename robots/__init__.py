"""Robot implementations loaded by name.

Each subpackage (e.g. :mod:`robots.libero`) bundles the agent-side robot
package (``get_robot_spec`` / ``get_toolkit`` factories, toolkit, prompts,
guides) together with its server-side scripts (``env_server.py`` /
``vla_server.py``). The robot registry in :mod:`rpent.robots.base` resolves a
robot by importing ``robots.<name>``.
"""
