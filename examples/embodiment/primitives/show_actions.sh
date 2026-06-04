#!/bin/bash
# Parse and display the action sequence from a workdir's log files.
#
# Usage:
#   bash show_actions.sh                              # default workdir
#   bash show_actions.sh /tmp/hybrid_repl_<tag>       # custom workdir

set -euo pipefail

WORKDIR="${1:-/tmp/hybrid_repl_object_swap_t2_s0}"

if [ ! -d "$WORKDIR" ]; then
    echo "ERROR: workdir not found: $WORKDIR" >&2
    exit 1
fi

LOGS=$(ls "$WORKDIR"/log_*.json 2>/dev/null | sort -V)
if [ -z "$LOGS" ]; then
    echo "No log files found in $WORKDIR" >&2
    exit 1
fi

printf "%-5s %-13s %-45s %s\n" "Step" "Action" "Params" "Description"
printf "%-5s %-13s %-45s %s\n" "----" "-------------" "--------------------------------------------" "-----------"

for f in $LOGS; do
    # Read action in bash so we can track it across iterations
    action=$(python3 -c "import json;d=json.load(open('$f'));print(d.get('command',{}).get('action',''))" 2>/dev/null || true)

    python3 -c "
import json, sys
d = json.load(open('$f'))
c = d['command']
action = c.get('action', '?')
label = '$f'.rsplit('log_', 1)[1].replace('.json', '')

# Build a compact params string + human description
params = ''
desc = ''

if action == 'move_to':
    xyz = c['xyz']
    x, y, z = xyz[0], xyz[1], xyz[2]
    grip = c.get('gripper', 0)
    params = f'to [{x:.3f}, {y:.3f}, {z:.3f}]'
    if grip > 0:
        desc = 'carry object'
        if z < 0.15:
            desc = 'descend to place'
        elif z > 0.22:
            desc = 'lift object'
    elif z < 0.15:
        desc = 'approach object (pre-grasp)'
    elif z > 0.22:
        desc = 'retreat / move to safe height'
    else:
        desc = 'reposition'
    # Add note if present
    note = c.get('note', '')
    if note:
        desc += f' — {note}'

elif action == 'pi0_pick':
    obj = c.get('track_obj', '?')
    chunks = c.get('max_chunks', '?')
    prompt = c.get('prompt', '?')
    params = f'\"{prompt}\"'
    desc = f'Pi0 grasps {obj} ({chunks} chunks)'

elif action == 'release':
    params = ''
    desc = 'open gripper → drop object'

elif action == 'set_gripper':
    g = c.get('gripper', 0)
    params = f'gripper={g}'
    if g > 0:
        desc = 'close gripper (secure grasp)'
    else:
        desc = 'open gripper'

elif action == 'rotate_wrist':
    roll = c.get('roll', '?')
    pitch = c.get('pitch', '?')
    params = f'roll={roll} pitch={pitch}'
    desc = 'rotate wrist'

elif action == 'rotate_pitch':
    angle = c.get('angle', '?')
    params = f'angle={angle}'
    desc = 'rotate pitch (tilt end-effector)'

elif action == 'move_pose':
    params = json.dumps(c)
    desc = 'move to pose'

elif action == 'exit':
    params = ''
    desc = 'task complete — save video & quit'

elif action in ('start_recording', 'save_video', 'snapshot', 'pi0_doubled'):
    pass  # skip infrastructure commands

else:
    params = json.dumps(c)
    desc = 'unknown action'

if action in ('start_recording', 'save_video', 'snapshot', 'pi0_doubled'):
    pass  # skip silently
else:
    print(f'{label:<5} {action:<13} {params:<45} {desc}')
"
done
