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

def train_tokenizer_remote():
    """
    Orchestrates the remote training job:
    1. Rents GPU -> 2. Syncs Code/Data -> 3. Runs Job -> 4. Downloads Model -> 5. Destroys Instance
    """
    instance_id = None
    try:
        # --- RENTING ---
        # searches for the cheapest rentable GPU (dph = dollars per hour)
        offer = vast_client.search_offers(query='gpu_name=RTX_5090 rented=False', order="dph_total")[0]
        _logger.info(f"Found offer: {offer['gpu_name']} | ID: {offer['id']} ($ {offer['dph_total']:.2f}/hr) located at {offer['geolocation']}. Renting...")
        instance = vast_client.launch_instance(id=offer['id'], 
                                               image="pytorch/pytorch", 
                                               disk=20, # GB
                                               gpu_name='RTX_5090',
                                               num_gpus='1',
                                               label="tokenizer_training",
                                               )
        instance_id = instance['new_contract']
        
        # --- WAIT FOR READY ---
        _logger.info(f"Rented {instance_id}. Waiting for 'running' status...")
        while True:
            status = vast_client.show_instance(id=instance_id)
            if status.get('actual_status') == 'running':
                break
            time.sleep(10)
        
        # --- ATTACH SSH KEY ---
        _logger.info(f"Attaching SSH Key to instance {instance_id}...")
        ssh_key = vast_client.show_ssh_keys()[0]["public_key"] # I currently only have one key on VastAI, so I am retrieving it and attaching to the rented machine
        try:
            vast_client.attach_ssh(instance_id=instance_id, ssh_key=ssh_key)
            _logger.info(f"Successfully attached SSH key to instance {instance_id}")
        except Exception as e:
            _logger.error(f"Failed to attach SSH key to instnace {instance_id}: {e}")
            raise

        # --- COPY FILES (Native SDK) ---
        _logger.info("Syncing files using SDK copy...")
        # SDK copy: src can be local folder, dst is 'instance_id:/path'
        vast_client.copy(
            src="./", 
            dst=f"{instance_id}:/workspace/", 
            identity=ssh_key
        )

        # --- EXECUTE COMMANDS (Native SDK) ---
        _logger.info("Starting remote execution...")
        
        remote_script = (
            "curl -LsSf https://astral.sh/uv/install.sh | sh && "
            "source $HOME/.cargo/env && "
            "uv pip install --system -r /workspace/requirements.txt && "
            "python /workspace/remote_entrypoint.py"
        )

        # The execute method runs the command and returns the output
        result = vast_client.execute(id=instance_id, COMMAND=remote_script)
        _logger.info(f"Remote Output: {result}")

        # --- DOWNLOAD RESULTS ---
        vast_client.copy(
            src=f"{instance_id}:/workspace/data/vocab/", 
            dst="./data/vocab/", 
            identity=ssh_key
        )
    except Exception as e:
        _logger.error(f"Failed to launch remote training job: {e}")
    finally:
        # Destroy the instance
        if instance_id:
            _logger.info(f"Destroying instance: {instance_id}...")
            vast_client.destroy_instance(id=instance_id)
            _logger.info(f"Instance {instance_id} destroyed!")

if __name__ == "__main__":
    train_tokenizer_remote()