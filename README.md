Create virtual environment 
```bash
uv venv --python 3.11 examples/aloha_sim/.venv
source examples/aloha_sim/.venv/bin/activate
uv pip sync examples/aloha_sim/requirements.txt
uv pip install -e packages/openpi-client
```
```bash
uv run scripts/serve_policy.py --env UR3 --default_prompt='press the red button'
```

#In another terminal activate the conda env : 
```bash
conda env create -f environment.yml
conda activate openpi_env
```
```bash
cd examples/ur_env
python ur3_infer.py
```
