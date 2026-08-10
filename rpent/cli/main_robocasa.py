"""RoboCasa CLI entrypoint — spawns env server, creates client, runs driver."""
import os, sys, socket, subprocess

from robots.robocasa.interactive_driver import RoboCasaDriver
from robots.robocasa import _add_cli_args, _init_runtime

def main():
    import argparse
    ap = argparse.ArgumentParser()
    _add_cli_args(ap, False)
    ap.add_argument("--workdir", required=True,
                    help="run directory (commands + evidence)")
    args = ap.parse_args()

    if not args.workdir:
        ap.error("--workdir is required and must be a non-empty path")

    from pathlib import Path
    from rpent.utils.logging import init_output_dir
    from rpent.dashboard.events import NullDashboardEventSink

    output_dir = Path(args.workdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    init_output_dir(output_dir)

    daemons, primitives_kwargs = _init_runtime(args, str(output_dir), NullDashboardEventSink())

    # Run the driver with the remote env + VLA
    from robots.robocasa.primitives import RoboCasaPrimitives
    from robots.robocasa import tools as robocasa_tools
    from rpent.utils.logging import get_output_dir

    out_dir = get_output_dir()

    primitives = RoboCasaPrimitives(**primitives_kwargs)
    primitives.reset()
    primitives.start_recording()
    robocasa_tools.dump_state(primitives, str(out_dir) if out_dir else "/tmp", step_idx=0, log=None)

    driver = RoboCasaDriver(primitives, workdir=str(output_dir))
    driver.run()


if __name__ == "__main__":
    main()
