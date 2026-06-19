# Physical Agent

To begin with, 
1. clone RLinf and physical agent side-by-side.
```bash
mkdir workspace && cd workspace
# PhysicalAgent depends on a forked branch of RLinf; we plan to merge the branch back to main after some more iterations
git clone https://github.com/jx-qiu/RLinf -b feature/physicalagent rlinf
git clone https://github.com/jx-qiu/PhysicalAgent physicalagent
```
1. in RLinf, configure a openpi+libero venv.
```bash
cd rlinf
bash requirements/install.sh embodied --env libero --model openpi --use-mirror --venv ../.venv-opi-libero
cd ..
source .venv-opi-libero/bin/activate
```
1. install additional PhysicalAgent dependencies on top of the above venv.
```bash
cd physicalagent
uv sync --active --inexact
bash scripts/install_libero_pro_plus.sh
```
1. Try the run: 
```bash
# configure API keys
export ANTHROPIC_BASE_URL=https://xxx
export ANTHROPIC_API_KEY=sk-xxx
export OPENAI_COMPAT_BASE_URL=https://xxx
export OPENAI_COMPAT_API_KEY=sk-xxx

export PI05_CHECKPOINT_PATH=/path/to/rlinf-pi05-libero-130-fullshot-sft # download from https://huggingface.co/datasets/RLinf/rlinf-pi05-libero-130-fullshot-sft and set path here
export LIBERO_TYPE=pro
export CUDA_DEVICE=0

# run a test task (libero_object_swap task 2, seed 0, with perception enabled), using an anthropic "claude-opus-4-7" model and a max token limit of 8192.
# alternatively, you can specify openai-compatible models using --cerebrum openai_compat --model xxx.
python cli/main.py --suite libero_object_swap --task 2 --seed 0 --perception --cerebrum anthropic --model claude-opus-4-7 --max_tokens 8192
```

## Adding new environments to Physical Agent
