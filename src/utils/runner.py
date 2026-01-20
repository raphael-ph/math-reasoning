# --- Runner Implementarion ---
# This is a functional abstraction to implement all Running jobs using VastAI SDK. Eventually I will
# promote this to a more structured module. For now, this should work fine to implement stuff faster.

import os
import time
import subprocess

# vast ai
from vastai_sdk import VastAI

# internal modules
from ..utils.logger import get_logger
from ..preprocessing.tokenizer import Tokenizer
from settings import VAST_AI_API_KEY, SSH_KEY_PATH

_logger = get_logger("runner", level="DEBUG")

# initialize VastAI
if VAST_AI_API_KEY:
    vast_client = VastAI(api_key=VAST_AI_API_KEY)
    _logger.info("VastAI initialized successfully.")
else:
    _logger.error("VastAI API Key not found on environment")
    raise RuntimeError("VastAI API Key not found on environment")

# get the ssh key
raw_ssh_path = SSH_KEY_PATH 
if raw_ssh_path:
    full_ssh_path = os.path.abspath(os.path.expanduser(raw_ssh_path))
else:
    _logger.error("SSH_KEY_PATH is None. Check your settings and environment variables.")
    raise FileNotFoundError("SSH Key Path not configured.")

if not os.path.exists(full_ssh_path):
    _logger.error(f"SSH Key not found at: {full_ssh_path}")
    raise FileNotFoundError(f"No SSH key at {full_ssh_path}")

def train_formalizer_remote():
    """
    Orchestrates the remote training job:
    1. Rents GPU -> 2. Syncs Code/Data -> 3. Runs Job -> 4. Downloads Model -> 5. Destroys Instance
    """
    instance_id = None
    try:
        # --- RENTING ---
        # searches for the cheapest rentable GPU (dph = dollars per hour)
        offer = vast_client.search_offers(query='gpu_name=RTX_5090 cpu_ram>128 disk_space>100 reliability>0.98', order="dph_total")[0]
        _logger.info(f"Found offer: {offer['gpu_name']} | ID: {offer['id']} ($ {offer['dph_total']:.2f}/hr) located at {offer['geolocation']}. Renting...")
        instance = vast_client.launch_instance(id=offer['id'], 
                                               image="vastai/vllm:v0.10.2-cuda-12.9-pytorch-2.8.0-py312", 
                                               disk=20, # GB
                                               gpu_name=offer['gpu_name'].replace(" ", "_"),
                                               num_gpus='1',
                                               label="formalizer_training",
                                               )
        instance_id = instance['new_contract']
        
        # --- WAIT FOR READY & GET IP/PORT ---
        _logger.info(f"Waiting for instance {instance_id} to initialize...")
        ssh_host, ssh_port = None, None
        while True:
            inst = vast_client.show_instance(id=instance_id)
            # We need the actual status AND the network info to be populated
            if inst.get('actual_status') == 'running' and inst.get('ssh_host'):
                ssh_host = inst['ssh_host']
                ssh_port = inst['ssh_port']
                break
            time.sleep(5)
        
        _logger.info(f"Connected to {ssh_host}:{ssh_port}")
        time.sleep(5) # Let the container's SSH service settle

        # --- SYNC FILES (Direct SCP) ---
        _logger.info("Syncing files via Rsync (filtering junk)...")
        
        # Define what to ignore
        excludes = [
            "--exclude=__pycache__",
            "--exclude=*.pyc",
            "--exclude=.git",
            "--exclude=.venv",
            "--exclude=venv",
            "--exclude=*.log",
            "--exclude=.env",
            "--exclude=.pytest_cache",
            "--exclude=bin/",
            "--exclude=lib/",
            "--exclude=include/",
            "--exclude=data/raw/*",      # Don't upload old results if they exist locally
            "--exclude=notebooks/",      # Don't upload old results if they exist locally
        ]

        rsync_cmd = [
            "rsync", "-avz",
            "--progress",
            "-e", f"ssh -p {ssh_port} -i {full_ssh_path} -o StrictHostKeyChecking=no",
            *excludes,
            "./", 
            f"root@{ssh_host}:/workspace/"
        ]
        
        _logger.info("Running rsync...")
        subprocess.run(rsync_cmd, check=True)

        # --- EXECUTE COMMANDS (Direct SSH) ---
        _logger.info("Starting remote execution...")
        remote_script = (
                "cd /workspace && "
                "curl -LsSf https://astral.sh/uv/install.sh | sh && "  # Install uv
                "source $HOME/.local/bin/env && "                      
                "uv venv .venv && "                                    # Create isolated env
                "source .venv/bin/activate && "                        # Activate env
                "uv pip install . && "                                 # Install dependencies
                "export PYTHONPATH=$PYTHONPATH:. && " 
                "python3 remote_formalizer_training_job.py"
        )
        
        ssh_cmd = [
            "ssh",
            "-p", str(ssh_port),
            "-i", full_ssh_path,
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            f"root@{ssh_host}",
            remote_script
        ]
        # Using subprocess.run will stream the output directly to your console
        subprocess.run(ssh_cmd, check=True)
    except Exception as e:
        _logger.error(f"Failed to launch remote training job: {e}")
    finally:
        # Destroy the instance
        if instance_id:
            _logger.info(f"Destroying instance: {instance_id}...")
            vast_client.destroy_instance(id=instance_id)
            _logger.info(f"Instance {instance_id} destroyed!")

if __name__ == "__main__":
    train_formalizer_remote()