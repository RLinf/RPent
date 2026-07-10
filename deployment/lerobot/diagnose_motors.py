"""Motor-health diagnostic for the SO101 follower arm.

Reads each servo's temperature, voltage, current, load, hardware-error status,
and torque state WITHOUT moving the arm. Use it when the arm stops reaching
commanded positions (drifts / presses the table / grasps miss even though the
scene localization and IK are correct) to check whether a joint is overheated,
under-volted, faulted, or straining.

Optionally (``--tracking-test``) it nudges each arm joint a few degrees around
its CURRENT pose and measures how far the achieved angle is from the command,
to pinpoint a joint that no longer tracks (slipped horn, weak motor, or
calibration drift). That part MOVES the arm, so put the arm in a safe, raised
pose first.

Run with the env server STOPPED (it needs exclusive access to the motor bus)::

    conda activate lerobot
    python deployment/lerobot/diagnose_motors.py                 # health only, no motion
    python deployment/lerobot/diagnose_motors.py --tracking-test # also nudges each joint
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Make ``rpent`` importable if ever needed; also keeps paths consistent
# with the driver when run from the repo root.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
_ALL_MOTORS = _ARM_JOINTS + ("gripper",)


def _read(bus, name: str, motor: str):
    """Read one raw register value, returning a short error string on failure."""
    try:
        return bus.read(name, motor, normalize=False)
    except Exception as e:  # noqa: BLE001 - diagnostic must never crash on one bad read
        return f"err:{type(e).__name__}"


def _fmt(v, width: int) -> str:
    return f"{v:>{width}}" if not isinstance(v, float) else f"{v:>{width}.1f}"


def _set_p_coefficient(robot, p: int) -> None:
    """Set the position gain on the arm joints (to test tracking stiffness)."""
    try:
        with robot.bus.torque_disabled():
            for m in _ARM_JOINTS:
                robot.bus.write("P_Coefficient", m, int(p))
        print(f"[set P_Coefficient={int(p)} on arm joints for this test]\n")
    except Exception as e:  # noqa: BLE001
        print(f"[could not set P_Coefficient: {e}]\n")


def _health(robot) -> None:
    bus = robot.bus
    try:
        obs = robot.get_observation()
    except Exception:
        obs = {}
    print("\n=== motor health (no motion) ===")
    print(
        f"{'motor':16}{'pos_deg':>9}{'temp_C':>8}{'volt_V':>8}"
        f"{'load':>8}{'current':>9}{'status':>8}{'torque':>8}{'Pgain':>7}"
    )
    for m in _ALL_MOTORS:
        temp = _read(bus, "Present_Temperature", m)   # deg C
        volt = _read(bus, "Present_Voltage", m)       # 0.1 V units
        load = _read(bus, "Present_Load", m)
        curr = _read(bus, "Present_Current", m)
        stat = _read(bus, "Status", m)                # 0 = OK; nonzero = HW error flags
        torq = _read(bus, "Torque_Enable", m)
        pgain = _read(bus, "P_Coefficient", m)
        pos = obs.get(f"{m}.pos")
        volt_v = round(volt / 10.0, 1) if isinstance(volt, (int, float)) else volt
        pos_s = f"{pos:9.1f}" if isinstance(pos, (int, float)) else f"{str(pos):>9}"
        print(
            f"{m:16}{pos_s}{_fmt(temp, 8)}{_fmt(volt_v, 8)}"
            f"{_fmt(load, 8)}{_fmt(curr, 9)}{_fmt(stat, 8)}{_fmt(torq, 8)}{_fmt(pgain, 7)}"
        )
    print(
        "\nInterpretation:\n"
        "  temp_C  > ~55  -> overheating; Feetech servos derate torque and under-reach.\n"
        "  volt_V         -> should match your supply and be steady; sag => weak torque.\n"
        "  status  != 0   -> a hardware-error flag latched (overload/overheat/voltage).\n"
        "  load/current   -> high while merely holding a light pose => a straining joint.\n"
    )


def _tracking_test(robot, nudge_deg: float) -> None:
    base = robot.get_observation()
    q0 = {m: float(base[f"{m}.pos"]) for m in _ARM_JOINTS}
    g0 = float(base.get("gripper.pos", 50.0))

    def send(qd: dict) -> None:
        act = {f"{m}.pos": float(qd[m]) for m in _ARM_JOINTS}
        act["gripper.pos"] = g0
        robot.send_action(act)

    print("=== per-joint tracking test (MOVES the arm ±%.0f deg around the current pose) ===" % nudge_deg)
    worst = 0.0
    for j in _ARM_JOINTS:
        for sgn in (+1.0, -1.0):
            qd = dict(q0)
            qd[j] = q0[j] + sgn * nudge_deg
            send(qd)
            time.sleep(0.9)
            ach = float(robot.get_observation()[f"{j}.pos"])
            err = ach - qd[j]
            worst = max(worst, abs(err))
            flag = "   <-- POOR TRACKING" if abs(err) > 3.0 else ""
            print(f"  {j:16} cmd={qd[j]:7.1f}  achieved={ach:7.1f}  err={err:+6.1f} deg{flag}")
        send(q0)
        time.sleep(0.9)
    print(
        f"\nworst tracking error: {worst:.1f} deg\n"
        "  > ~3 deg on a free-space nudge = that joint is not tracking "
        "(slipped horn / weak or faulted motor / calibration drift). Recalibrate "
        "the follower or inspect that motor before running the agent again.\n"
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", default="/dev/ttyACM1")
    p.add_argument("--calibration-id", default="my_awesome_follower_arm")
    p.add_argument(
        "--tracking-test", action="store_true",
        help="Also nudge each arm joint a few degrees and report the "
             "command-vs-readback error. MOVES THE ARM — place it in a safe, "
             "raised pose first.",
    )
    p.add_argument("--nudge-deg", type=float, default=6.0)
    p.add_argument(
        "--p-coefficient", type=int, default=None,
        help="Set the arm servos' P_Coefficient (position gain) before testing "
             "(LeRobot uses 16; factory default 32). Sweep e.g. 32 then 48 to "
             "see if tracking tightens.",
    )
    args = p.parse_args()

    from lerobot.robots.so_follower import SO101Follower
    from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig

    robot = SO101Follower(
        SO101FollowerConfig(
            port=args.port,
            id=args.calibration_id,
            use_degrees=True,
            disable_torque_on_disconnect=False,
            cameras={},
        )
    )
    robot.connect(calibrate=False)
    try:
        if args.p_coefficient is not None:
            _set_p_coefficient(robot, args.p_coefficient)
        _health(robot)
        if args.tracking_test:
            _tracking_test(robot, args.nudge_deg)
    finally:
        robot.disconnect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
