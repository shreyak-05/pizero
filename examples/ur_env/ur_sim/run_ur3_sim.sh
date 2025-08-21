#!/bin/bash

# Path to Isaac Sim directory and script
ISAAC_SIM="/home/shreya/isaacsim/isaac-sim.sh"

# Directory with your simulation code
CODE_DIR="/home/shreya/Desktop/PI/openpi/examples/ur_env/ur_sim"

# Check if Isaac Sim exists
if [ ! -f "$ISAAC_SIM" ]; then
    echo "Isaac Sim not found at $ISAAC_SIM"
    exit 1
fi

# # Start the policy server in the background
# cd ~/Desktop/PI/openpi
# echo "Starting policy server..."
# python scripts/serve_policy.py --env ur3_real --port 8000 --default_prompt="pick" &
# POLICY_SERVER_PID=$!

# # Wait for the policy server to start
# echo "Waiting for policy server to start..."
# sleep 5

# Run simulation through Isaac Sim's Python interpreter
echo "Starting UR3 simulation in Isaac Sim..."
cd "$CODE_DIR"
"$ISAAC_SIM" --python main.py --task pick_and_place --host localhost --port 8000

# Kill the policy server when done
echo "Stopping policy server..."
kill $POLICY_SERVER_PID