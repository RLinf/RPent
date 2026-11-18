## RoboTwin startup

Before the first robot mutation, read robots/robotwin/guides/GUIDE_RPENT.md
completely, then inspect view_env_state(step=0) and its head image.

## Begin execution

Follow the required read order, bind the current task's targets and
relations from fresh observation, then execute the first unmet recipe phase.
After each action verify its observable gate, preserve achieved relations, and
use the complete current task_language unchanged for every call to the action model.
