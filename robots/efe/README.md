# EFE-compatible evaluation

[English](README.md) | [简体中文](README.zh-CN.md)

Status: experimental, **EFE-compatible / non-official**. Harbor `0.23.0` is the
supported host version and requires Python 3.12.

RPent runs EFE (Embodied First Exam) tasks as a Harbor installed agent. Harbor
starts an agent container (`main`) and a simulator container (`sim`); RPent's
`CodexPlanner` operates the robot through the task's `rcb_client` service, and
EFE's verifier scores the episode.

- Supported: 1150 `standard` and `standard-realtime` tasks.
- Not supported yet: 959 `privileged` tasks, whose deliverable is a replayable
  `trajectory.npz` or a policy program/model instead of step-by-step control.

Setup, run command, tool sets, and result locations:
<https://rpent.readthedocs.io/en/latest/rst_source/simulators/efe.html>

Code: `harbor_agent.py` (Harbor adapter), `runner.py`, and `cli.py`; robot tools
live in `robots/efe/`.
