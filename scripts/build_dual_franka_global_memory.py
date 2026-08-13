"""Build prompt-ready dual-Franka memory drafts from structured episode logs.

This script is intentionally local and deterministic. It does not call any
external model. It reads the structured audit JSON, full_log.jsonl, and
localization_diagnostic JSON files that are already produced by real-robot
episodes, then writes English and Chinese reviewed-memory drafts.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EpisodeAudit:
    episode_dir: Path
    status: str
    strategy_notes: str
    coordination: str
    tool_sequence: str
    final_state: str
    durable_lessons: tuple[str, ...]
    next_run_change: str


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            rows.append(item)
    return rows


def _clean(text: Any) -> str:
    return str(text or "").replace("\n", " ").strip()


def _load_audits(root: Path) -> list[EpisodeAudit]:
    by_episode: dict[Path, tuple[Path, dict[str, Any]]] = {}
    for path in sorted(root.glob("episode_*/dual_franka_*.json")):
        data = _read_json(path)
        if not data or data.get("schema") != "physical_agent.dual_franka.audit.v1":
            continue
        current = by_episode.get(path.parent)
        # Prefer the task-named audit over the generic runner audit when both
        # exist in the same episode directory.
        if current is not None and (
            "clean_desk_dual_franka_agent_vla" in current[0].name
            or "clean_desk_dual_franka_stage4" in current[0].name
        ):
            continue
        by_episode[path.parent] = (path, data)

    audits: list[EpisodeAudit] = []
    for path, data in sorted(by_episode.values(), key=lambda item: item[0]):
        lessons = data.get("durable_lessons")
        if not isinstance(lessons, list):
            lessons = []
        audits.append(
            EpisodeAudit(
                episode_dir=path.parent,
                status=_clean(data.get("status") or "unknown"),
                strategy_notes=_clean(data.get("strategy_notes")),
                coordination=_clean(data.get("arm_coordination_summary")),
                tool_sequence=_clean(data.get("tool_sequence_summary")),
                final_state=_clean(data.get("final_state_summary")),
                durable_lessons=tuple(_clean(item) for item in lessons if _clean(item)),
                next_run_change=_clean(data.get("next_run_change")),
            )
        )
    return audits


def _iter_snapshots(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("episode_*/full_log.jsonl")):
        for item in _read_jsonl(path):
            if item.get("event") == "snapshot":
                item["_episode_dir"] = path.parent.name
                rows.append(item)
    return rows


def _vla_events(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for row in rows:
        command = row.get("command")
        result = row.get("result")
        if not isinstance(command, dict) or command.get("action") != "run_vla_skill":
            continue
        if not isinstance(result, dict):
            continue
        vla = result.get("vla")
        if not isinstance(vla, dict):
            continue
        stop_rule = vla.get("stop_rule")
        if not isinstance(stop_rule, dict):
            stop_rule = {}
        events.append(
            {
                "episode": row.get("_episode_dir"),
                "step": row.get("step_idx"),
                "skill": vla.get("skill_name") or command.get("skill_name"),
                "chunks": vla.get("chunks_executed"),
                "steps": vla.get("steps_executed"),
                "stop_phase": stop_rule.get("phase"),
                "stop_condition": stop_rule.get("condition"),
            }
        )
    return events


def _joint_warnings(rows: list[dict[str, Any]], *, drift_threshold: float = 0.8) -> list[str]:
    warnings: list[str] = []
    for row in rows:
        state = row.get("state")
        if not isinstance(state, dict):
            continue
        health = state.get("joint_health")
        if not isinstance(health, dict):
            continue
        for arm in ("left", "right"):
            arm_health = health.get(arm)
            if not isinstance(arm_health, dict):
                continue
            status = arm_health.get("status")
            drift = arm_health.get("drift_l2")
            if status == "ok" and not (
                isinstance(drift, (int, float)) and drift >= drift_threshold
            ):
                continue
            warnings.append(
                "{episode} step {step} {arm}: status={status}, drift_l2={drift}, "
                "condition_number={cond}, sigma_min={sigma}, reasons={reasons}".format(
                    episode=row.get("_episode_dir"),
                    step=row.get("step_idx"),
                    arm=arm,
                    status=status,
                    drift=drift,
                    cond=arm_health.get("condition_number"),
                    sigma=arm_health.get("sigma_min"),
                    reasons=arm_health.get("reasons") or [],
                )
            )
    return warnings


def _localization_stats(root: Path) -> tuple[Counter[str], list[str]]:
    counts: Counter[str] = Counter()
    examples: list[tuple[tuple[int, str], str]] = []
    for path in sorted(root.glob("episode_*/localization_diagnostic/*.json")):
        data = _read_json(path)
        if not data:
            continue
        proj = data.get("projection")
        if not isinstance(proj, dict):
            continue
        camera = str(proj.get("camera") or "unknown")
        ok = bool(proj.get("ok"))
        selection_valid = proj.get("selection_valid")
        if selection_valid is None and ok:
            validity = "legacy_ok"
        elif ok and bool(selection_valid):
            validity = "valid"
        else:
            validity = "rejected"
        key = f"{camera}:{validity}"
        counts[key] += 1
        score = (
            0 if camera == "d455" else 1,
            0 if proj.get("target_name") else 1,
            0 if validity != "legacy_ok" else 1,
            str(path),
        )
        examples.append(
            (
                score,
                "{episode}/{name}: camera={camera}, target={target}, pixel={pixel}, "
                "ok={ok}, valid={valid}, depth_m={depth}, point_xyz={point}, "
                "annotated={annotated}".format(
                    episode=path.parent.parent.name,
                    name=path.name,
                    camera=camera,
                    target=proj.get("target_name"),
                    pixel=proj.get("pixel"),
                    ok=proj.get("ok"),
                    valid=proj.get("selection_valid"),
                    depth=proj.get("depth_m"),
                    point=proj.get("point_xyz"),
                    annotated=Path(str(data.get("annotated_image") or "")).name,
                ),
            )
        )
    examples.sort(key=lambda item: item[0])
    return counts, [item for _, item in examples[:16]]


def _bullet(items: list[str] | tuple[str, ...]) -> list[str]:
    if not items:
        return ["- (none)"]
    return [f"- {item}" for item in items]


def _status_table(audits: list[EpisodeAudit]) -> list[str]:
    lines = ["| Episode | Status | Key outcome |", "| --- | --- | --- |"]
    for audit in audits:
        outcome = audit.final_state or audit.strategy_notes
        if len(outcome) > 220:
            outcome = outcome[:217].rstrip() + "..."
        lines.append(f"| `{audit.episode_dir.name}` | `{audit.status}` | {outcome} |")
    return lines


def _collect_lessons(audits: list[EpisodeAudit]) -> list[str]:
    lessons: list[str] = []
    seen: set[str] = set()
    for audit in audits:
        for item in audit.durable_lessons:
            if item not in seen:
                seen.add(item)
                lessons.append(f"`{audit.episode_dir.name}`: {item}")
        if audit.next_run_change and audit.next_run_change not in seen:
            seen.add(audit.next_run_change)
            lessons.append(f"`{audit.episode_dir.name}` next run: {audit.next_run_change}")
    return lessons


def _render_en(
    *,
    root: Path,
    audits: list[EpisodeAudit],
    vla: list[dict[str, Any]],
    joint_warnings: list[str],
    loc_counts: Counter[str],
    loc_examples: list[str],
) -> str:
    status_counts = Counter(audit.status for audit in audits)
    vla_skill_counts = Counter(str(item.get("skill")) for item in vla)
    vla_stop_counts = Counter(
        f"{item.get('stop_phase')}/{item.get('stop_condition')}" for item in vla
    )
    lessons = _collect_lessons(audits)
    lines = [
        "# Dual-Franka Clean-Desk Global Memory Draft",
        "",
        f"- source_root: `{root}`",
        f"- generated_at: {datetime.now().isoformat(timespec='seconds')}",
        f"- episodes_with_audit: {len(audits)}",
        f"- audit_status_counts: {dict(status_counts)}",
        f"- vla_skill_counts: {dict(vla_skill_counts)}",
        f"- vla_stop_rule_counts: {dict(vla_stop_counts)}",
        f"- localization_counts: {dict(loc_counts)}",
        "",
        "## Non-Negotiable Rules",
        "",
        "- Keep the category gate: bowls first, then plates, then cup, then chopsticks, then spoon. Do not advance until the current category is visually absent from the source workspace or visibly inside the basket.",
        "- Use the shared `right_base` world frame for all metric reasoning and rule-based motions. Left-arm positions must also be interpreted after transformation into `right_base`.",
        "- Do not issue robot motion without a synchronized state snapshot containing RGB artifacts and depth for the depth cameras. A camera-stall or missing `latest_state.json` is a stop condition, not an invitation to move blind.",
        "- Rule-based motions are for free-space staging, inspection clearance, gripper opening/closing, and reset recovery. Contact-rich grasp, handoff, and object release should normally go through VLA.",
        "- Do not let the agent declare task success. Operator verdict controls episode completion.",
        "",
        "## VLA Boundary",
        "",
        "- Treat the VLA checkpoint as a full-process policy. Object-specific VLA tool names are operational handles, not evidence that the underlying checkpoint has separate clean subskills.",
        "- The VLA can overrun into the next behavior if allowed to execute for too long. Use rule-based stop criteria and visual inspection after each VLA call.",
        "- For grasp-like VLA calls, stop the skill when the active gripper closes and the object has lifted about 0.15 m, then inspect before handoff/place.",
        "- For handoff, stop when the right gripper opens after a stable left receive motion. For placement, stop when the left gripper opens. This stop means the skill phase ended, not that the object was successfully placed.",
        "- If a VLA call leaves a gripper closed at minimum width but the object did not move with a small lift, treat it as an ineffective grasp and do not proceed to place.",
        "",
        "## Perception And Localization",
        "",
        "- Use D455 or base RealSense depth for metric points. The wrist Lumos fisheye RGB views are useful for contact/context inspection but currently do not provide stable depth in this deployment path.",
        "- The selected pixel must be inside visible material of the named object, not on image-space air/background above it. Always inspect the annotated diagnostic image returned by the projection tool.",
        "- Reject projections that fail tabletop/workspace validity checks or whose annotated point visibly lands on a different object.",
        "- Thin chopsticks are difficult for depth: D455/base depth can be dominated by the tabletop near narrow material. Rejected chopstick projections must not be used as metric targets.",
        "- Base RealSense pixels near plate edges can return background depth. Prefer D455 interior/rim-band pixels for plate localization when available.",
        "",
        "### Localization Evidence",
        "",
        "*localization_counts are computed from `localization_diagnostic/*.json`.*",
        "",
        "*examples:*",
    ]
    lines.extend(_bullet(loc_examples))
    lines.extend(
        [
            "",
            "## Reset And Joint Health",
            "",
            "- Reset is a recovery primitive for joint accumulation, failed handoff/place states, and unsafe/ambiguous scene states. Reset both arms together and do not rely on reset to change gripper state.",
            "- After reset, re-stage both arms from the new nominal pose before invoking VLA again. Do not assume the TCP returned to the previous object pose.",
            "- Monitor `joint_health`: high drift from nominal, high condition number, small `sigma_min`, or explicit warning status are evidence to reset before further VLA. Recent logs show drift values above 1.0 can appear after VLA handoff/place sequences.",
            "",
            "### Joint-Health Evidence",
            "",
        ]
    )
    lines.extend(_bullet(joint_warnings[-20:]))
    lines.extend(
        [
            "",
            "## Object Recipes",
            "",
            "- Bowls: localize bowl material with D455/base depth, stage the right arm high above the bowl in free space, then call bowl VLA. If adjacent dish/plate is selected instead, stop and improve separation or change staging rather than repeating the same call.",
            "- Plates: avoid center-only approaches. Localize a valid rim/interior rim-band point, stage higher and laterally aligned with a graspable rim, call plate VLA once, then inspect lift. If it closes without lift, open and re-stage to a different rim.",
            "- Cup: cup grasp can succeed after right-arm staging. Delay cup manipulation until bowls/plates are cleared enough that cup skill does not select nearby objects. After cup is held, use a deliberate handoff/pre-place station rather than invoking handoff from a crowded basket/table-edge pose.",
            "- Chopsticks/spoon: do not trust rejected depth for thin utensils. If the VLA drifts toward the spoon while processing chopsticks, pause; do not break the category gate.",
            "- Handoff/place: pre-stage the left gripper open and visible at a deliberate handoff station. After handoff, verify the left gripper is closed and holding the object, then move the left-held object above/near the basket before place VLA. Do not call place directly from the table-center handoff pose.",
            "",
            "## Forbidden Actions",
            "",
            "- Do not repeatedly call the same VLA grasp after the first attempt visibly drifts, sweeps, or selects the wrong object.",
            "- Do not use runtime VLA instruction text as the main control surface unless the server configuration confirms that runtime instruction is actually overriding the config instruction.",
            "- Do not use stale absolute coordinates across resets or scene changes.",
            "- Do not continue motion when a human/operator hand occludes D455/base views or enters the workspace.",
            "- Do not use background/air pixels above objects for depth projection.",
            "",
            "## Episode Evidence Table",
            "",
        ]
    )
    lines.extend(_status_table(audits))
    lines.extend(
        [
            "",
            "## Raw Durable Lessons",
            "",
        ]
    )
    lines.extend(_bullet(lessons))
    return "\n".join(lines) + "\n"


def _render_zh(
    *,
    root: Path,
    audits: list[EpisodeAudit],
    vla: list[dict[str, Any]],
    joint_warnings: list[str],
    loc_counts: Counter[str],
    loc_examples: list[str],
) -> str:
    status_counts = Counter(audit.status for audit in audits)
    vla_skill_counts = Counter(str(item.get("skill")) for item in vla)
    vla_stop_counts = Counter(
        f"{item.get('stop_phase')}/{item.get('stop_condition')}" for item in vla
    )
    lessons = _collect_lessons(audits)
    lines = [
        "# 双臂 Franka 清理桌面全局记忆草稿",
        "",
        f"- 来源目录: `{root}`",
        f"- 生成时间: {datetime.now().isoformat(timespec='seconds')}",
        f"- 含 audit 的 episode 数量: {len(audits)}",
        f"- audit 状态统计: {dict(status_counts)}",
        f"- VLA 技能调用统计: {dict(vla_skill_counts)}",
        f"- VLA 停止规则统计: {dict(vla_stop_counts)}",
        f"- 反投影诊断统计: {dict(loc_counts)}",
        "",
        "## 不可协商规则",
        "",
        "- 严格保持类别顺序：先碗，再盘子，再杯子，再筷子，最后勺子。只有当前类别在源工作区看不到，或已经明确在篮筐里，才允许进入下一类。",
        "- 所有规则化坐标推理统一使用 `right_base` 世界坐标系。左臂位置也要转换到 `right_base` 后再描述和使用。",
        "- 没有同步状态快照时禁止运动。快照必须包含 RGB 图像，以及深度相机对应的深度。相机卡死、`latest_state.json` 缺失或为空，都应该视为停止条件。",
        "- 规则化原语负责自由空间预定位、观察避挡、夹爪开合和 reset 恢复。抓取、换手、放置这类 contact-rich 操作一般交给 VLA。",
        "- 任务是否结束由 operator 决定，不能让 agent 自己判断成功并结束。",
        "",
        "## VLA 介入边界",
        "",
        "- 当前 VLA checkpoint 本质上是全流程策略。对象级 VLA 工具名只是操作入口，不代表底层真的训练出了互相独立的干净子技能。",
        "- 如果一次 VLA 执行太长，它可能越过当前阶段，直接进入下一段行为。每次 VLA 后都要用规则化停止条件和视觉检查截断。",
        "- 抓取类 VLA：当主动夹爪闭合，并且物体上升约 0.15 m 后，应停止当前技能并检查，而不是继续让 VLA 自由运行。",
        "- 换手类 VLA：以右夹爪打开作为阶段结束信号。放置类 VLA：以左夹爪打开作为阶段结束信号。注意这只代表阶段结束，不代表成功。",
        "- 如果 VLA 后夹爪闭到很小，但物体没有随小幅抬升一起移动，应判定为无效抓取，不能直接进入放置。",
        "",
        "## 感知和定位",
        "",
        "- 需要 metric 坐标时优先使用 D455 或 base RealSense 深度。腕部 Lumos 鱼眼 RGB 主要用于接触状态和局部观察，目前这条部署链路里不能稳定提供深度。",
        "- agent 选点必须落在目标物体可见材质内部，不能点在物体上方的空气/背景。每次反投影都要看返回的标点诊断图。",
        "- 如果反投影失败、越出桌面/工作区约束，或者标点图显示点到了别的物体上，必须拒绝这个坐标。",
        "- 筷子这类细长物体的深度不稳定，深度容易被桌面主导。被拒绝的筷子反投影不能拿来做规则化移动。",
        "- 盘子边缘附近的 base RealSense 像素可能返回背景深度。盘子优先用 D455 的内部/边缘带有效像素。",
        "",
        "### 反投影证据",
        "",
        "下面统计来自 `localization_diagnostic/*.json`。",
        "",
    ]
    lines.extend(_bullet(loc_examples))
    lines.extend(
        [
            "",
            "## Reset 和关节健康",
            "",
            "- reset 是处理关节累积、换手/放置失败状态、以及不安全/不明确场景的恢复原语。按当前策略，左右臂一起 reset，不依赖 reset 改变夹爪状态。",
            "- reset 后必须重新从 nominal 位姿预定位，不能假设 TCP 还在原来的物体附近。",
            "- 监控 `joint_health`：drift 明显增大、condition number 变高、`sigma_min` 变小，或 status 出现 warning，都应作为继续 VLA 前 reset 的证据。最近日志中 VLA 换手/放置后 drift 超过 1.0 的情况需要特别注意。",
            "",
            "### 关节健康证据",
            "",
        ]
    )
    lines.extend(_bullet(joint_warnings[-20:]))
    lines.extend(
        [
            "",
            "## 物体操作配方",
            "",
            "- 碗：用 D455/base 深度定位碗的可见材质，先把右臂在自由空间高位移到碗上方，再调用 bowl VLA。如果 VLA 选中了旁边的盘子/碟子，不要重复同样调用，应先改善物体分离或改变预定位。",
            "- 盘子：不要只对准盘子中心。应该定位有效的盘沿/内侧盘沿点，高位、侧向对齐可抓盘沿后调用 plate VLA。若闭合但没有抬起，打开夹爪并换另一个盘沿点。",
            "- 杯子：右臂预定位后杯子抓取可以成功。但应等碗/盘子清掉，杯子不再被邻近物干扰后再做。杯子抓住后，应先进入明确换手/预放置站位，避免直接在篮筐边缘或桌面中心调用 handoff。",
            "- 筷子/勺子：不要相信被拒绝的细物体深度。如果处理筷子时 VLA 开始漂向勺子，应该暂停，不能破坏类别顺序。",
            "- 换手/放置：左夹爪要预先打开，并移动到可见、安全的换手站位。换手后先确认左夹爪闭合且确实持物，再用规则化左臂自由空间移动到篮筐上方/附近，最后再调用 place VLA。不要在桌面中心换手位直接调用 place。",
            "",
            "## 禁止动作",
            "",
            "- 第一次 VLA 已经明显漂移、扫物体、或选错物体时，不要连续重复同一个 VLA 抓取。",
            "- 除非 server 配置确认 runtime instruction 会覆盖 config instruction，否则不要把运行时 VLA prompt 当作主要控制手段。",
            "- reset 或场景变化后，不要沿用旧的绝对坐标。",
            "- D455/base 视角被人手或人体遮挡时，停止运动并等待 operator 处理。",
            "- 不要用物体上方背景/空气点做深度反投影。",
            "",
            "## Episode 证据表",
            "",
        ]
    )
    lines.extend(_status_table(audits))
    lines.extend(
        [
            "",
            "## 原始 durable lessons / next-run changes",
            "",
        ]
    )
    lines.extend(_bullet(lessons))
    return "\n".join(lines) + "\n"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build dual-Franka global memory drafts from structured logs.",
    )
    parser.add_argument(
        "--root",
        default=(
            "logs/dual_franka_real/clean_desk_dual_franka_agent_vla"
        ),
        help="Episode root, relative to repo root unless absolute.",
    )
    parser.add_argument(
        "--output-dir",
        default="logs/dual_franka_memory_comparison/enhanced_structured",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    repo = Path(__file__).resolve().parents[1]
    root = Path(args.root).expanduser()
    if not root.is_absolute():
        root = repo / root
    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = repo / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    audits = _load_audits(root)
    snapshots = _iter_snapshots(root)
    vla = _vla_events(snapshots)
    joint_warnings = _joint_warnings(snapshots)
    loc_counts, loc_examples = _localization_stats(root)

    en = _render_en(
        root=root,
        audits=audits,
        vla=vla,
        joint_warnings=joint_warnings,
        loc_counts=loc_counts,
        loc_examples=loc_examples,
    )
    zh = _render_zh(
        root=root,
        audits=audits,
        vla=vla,
        joint_warnings=joint_warnings,
        loc_counts=loc_counts,
        loc_examples=loc_examples,
    )
    (output_dir / "global_memory_draft.md").write_text(en, encoding="utf-8")
    (output_dir / "global_memory_draft.zh.md").write_text(zh, encoding="utf-8")
    manifest = {
        "source_root": str(root),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "episodes": [str(audit.episode_dir) for audit in audits],
        "outputs": [
            str(output_dir / "global_memory_draft.md"),
            str(output_dir / "global_memory_draft.zh.md"),
        ],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(output_dir / "global_memory_draft.md")
    print(output_dir / "global_memory_draft.zh.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
