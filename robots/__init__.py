"""Environment/robot implementations loaded by name.

Each subpackage (e.g. :mod:`robots.libero`) bundles the agent-side env
package (``get_env_spec`` / ``get_toolkit`` factories, toolkit, prompts,
guides) together with its driver-side scripts (``env_server.py`` /
``vla_server.py``). The env registry in :mod:`rpent.envs.base` resolves an
env by importing ``robots.<name>``.
"""
