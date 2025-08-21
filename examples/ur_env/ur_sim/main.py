import dataclasses
import logging
import pathlib
import os
import sys
import numpy as np

# Configure environment variables
os.environ["JAX_TRACEBACK_FILTERING"] = "off"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".50"

# Import OpenPI client libraries
from openpi_client import action_chunk_broker
from openpi_client import websocket_client_policy
from openpi_client.runtime import runtime
from openpi_client.runtime.agents import policy_agent
import tyro

# Import the UR3e environment (will create this next)
import env


@dataclasses.dataclass
class Args:
    """Arguments for the UR3e Isaac Sim runtime."""
    
    # Output directory for videos and logs
    out_dir: pathlib.Path = pathlib.Path("data/ur3_sim/videos")
    
    # Task parameters
    task: str = "pick_and_place"  # Options: "pick_and_place", "stacking", "pushing"
    seed: int = 0
    
    # Policy parameters
    action_horizon: int = 16  # Must match the policy's action_horizon
    prompt: str = "pick"      # Simple prompt to avoid tokenization issues
    
    # Policy server connection
    host: str = "localhost"
    port: int = 8000
    
    # Simulation parameters  
    headless: bool = False    # Set to True for no GUI
    physics_dt: float = 1/60.0
    rendering_dt: float = 1/60.0
    
    # Camera parameters
    use_base_camera: bool = True
    use_wrist_camera: bool = True


def main(args: Args) -> None:
    """Main function to run the UR3e simulation with OpenPI policy."""
    
    # Create the environment
    env = env.UR3eIsaacEnv(
        task=args.task,
        seed=args.seed,
        headless=args.headless,
        physics_dt=args.physics_dt,
        rendering_dt=args.rendering_dt,
        use_base_camera=args.use_base_camera,
        use_wrist_camera=args.use_wrist_camera
    )
    
    # Create policy agent with action chunk broker
    agent = policy_agent.PolicyAgent(
        policy=action_chunk_broker.ActionChunkBroker(
            policy=websocket_client_policy.WebsocketClientPolicy(
                host=args.host,
                port=args.port,
            ),
            action_horizon=args.action_horizon,
        )
    )
    
    # Create video saver (if needed)
    video_saver = env.VideoSaver(args.out_dir, fps=30)
    
    # Create runtime
    sim_runtime = runtime.Runtime(
        environment=env,
        agent=agent,
        subscribers=[video_saver],
        max_hz=60  # Control frequency
    )
    
    # Run the simulation
    logging.info(f"Starting UR3e simulation with task: {args.task}")
    logging.info(f"Connecting to policy server at {args.host}:{args.port}")
    try:
        sim_runtime.run()
    except KeyboardInterrupt:
        logging.info("Simulation stopped by user")
    finally:
        env.close()  # Clean up Isaac Sim resources


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        force=True
    )
    
    # Parse arguments and run
    tyro.cli(main)