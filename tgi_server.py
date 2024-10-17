
import docker
import transformers
import torch
import os
import argparse
import time
import subprocess

class TGIServer():
    """This class is a python class wrapper around the TGI docker server. It allows you to start and stop the server with specified parameters and also handles converting saved model formats (the direct preference optimization codebase saves the model in a different format as that expected by TGI)."""
    
    def get_slurm_device_ids(self):
        """Check the SLURM environment variables to get the device IDs"""
        command = f"scontrol show job $SLURM_JOB_ID -d | grep IDX:"
        result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        out = result.stdout
        tgi_device_ids = out.split("IDX:")[1].split(")")[0].strip()
        if "-" in tgi_device_ids:
            commas = tgi_device_ids.split(",")
            out = ""
            for c in commas:
                if "-" in c:
                    start, end = map(int, c.split("-"))
                    out += ",".join(map(str, range(start, end + 1))) + ","
                else:
                    out += c + ","
            out = out[:-1]
            tgi_device_ids = out
        return tgi_device_ids

    def __init__(self, model_name="meta-llama/Meta-Llama-3-8B-Instruct", model_archive=None, prepared_model_archive=None, device_ids="0", port=3002, gpu_memory=1.0, cache_dir="./cache", hf_token=None, max_total_tokens=8192, max_input_tokens=8000):
        """To use with standard HuggingFace model, pass in model_name. To use with fine tuned model, pass in model_archive. device_ids can be a comma-separated list of device IDs or "unknown" to get the device IDs from SLURM environment variables. gpu_memory can be set to less than 1 (ideally 0.9) if you are also running colbert on the same GPU during evaluation."""
        self.device_ids = self.get_slurm_device_ids() if device_ids == "unknown" else device_ids         
        self.model_name = model_name
        self.model_archive = model_archive
        self.prepared_model_archive = prepared_model_archive
        self.port = port
        self.gpu_memory = gpu_memory
        self.cache_dir = cache_dir
        self.hf_token = hf_token if hf_token else os.getenv("HF_TOKEN")
        self.max_total_tokens = max_total_tokens
        self.max_input_tokens = max_input_tokens
        
    def prepare_model(self, overwrite=False):
        """Models are saved from training code using write_state_dict, so we need to load the model and tokenizer and save them using save_pretrained"""
        if self.model_archive is None:
            return
        model_path = self.model_archive if self.model_archive[0] == "/" else f"{self.cache_dir}/{self.model_archive}"
        out_path = os.path.join(model_path, "tgi_policy")
        if os.path.exists(out_path) and not overwrite:
            self.prepared_model_archive = out_path
            return
        print("TGI server is preparing model", self.model_archive)
        policy = transformers.AutoModelForCausalLM.from_pretrained(self.model_name, cache_dir=self.cache_dir, low_cpu_mem_usage=True)
        tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_name, cache_dir=self.cache_dir)
        state_dict = torch.load(os.path.join(model_path, "policy.pt"), map_location='cpu')
        policy.load_state_dict(state_dict['state'])
        policy.save_pretrained(out_path)
        tokenizer.save_pretrained(out_path)
        print("TGI server prepared model, saved model and tokenizer to", out_path)
        self.prepared_model_archive = f"{model_path}/tgi_policy"
        del policy
        del tokenizer
        torch.cuda.empty_cache()
        
    def start(self, verbose=True):
        """Start the docker container that runs TGI server. It launches the docker container and then reads the logs to check if the server is ready to accept connections."""
        
        # Initialize the Docker client
  
        client = docker.from_env()
        if self.prepared_model_archive is None:
            mount_path = self.cache_dir
            path = self.model_name
        else:
            mount_path = self.prepared_model_archive.split("tgi_policy")[0]
            path = "/data/tgi_policy"
        if not os.path.isabs(mount_path): # docker requires mount path to be absolute
            mount_path = os.path.abspath(mount_path)
        print("starting tgi container with path", path, "mount_path", mount_path, "port", self.port, "device_ids", self.device_ids)
        self.container = client.containers.run(
            image="ghcr.io/huggingface/text-generation-inference:2.2.0",
            environment={"HUGGING_FACE_HUB_TOKEN": self.hf_token},
            runtime="nvidia",
            device_requests=[
                docker.types.DeviceRequest(device_ids=[self.device_ids], capabilities=[['gpu']])
            ],
            shm_size="1g",
            ports={"80/tcp": self.port},
            volumes={mount_path: {'bind': '/data', 'mode': 'rw'}},
            command=f"--model-id '{path}' --max-best-of 1 --max-total-tokens {self.max_total_tokens} --max-input-tokens {self.max_input_tokens} --cuda-memory-fraction {self.gpu_memory}",
            detach=True
        )
        connected = False
        
        while not connected:
            for line in self.container.logs(stream=True):
                if verbose:
                    print(line.strip())
                if "Connected" in str(line):
                    connected = True
                    break
        print(f"Container started with ID: {self.container.id}")
        
    def stop(self):
        try:
            self.container.stop()
            time.sleep(1)
            print(f"Container {self.container.id} stopped")
        except ImportError:
            print("Container already stopped")
    
    def __del__(self):        
        self.stop()
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start a TGI server with specified parameters.")
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct", help="Name of the model to use.")
    parser.add_argument("--model_archive", type=str, default=None, help="Path to the model archive, should be to folder containing optimizer.pt, policy.pt, scheduler.pt")
    parser.add_argument("--device_ids", type=str, default="unknown", help="Comma-separated list of device IDs or 'unknown' to get from SLURM.")
    parser.add_argument("--port", type=int, default=3002, help="Port to run the TGI server on.")
    parser.add_argument("--gpu_memory", type=float, default=1.0, help="Fraction of GPU memory to use.")
    parser.add_argument("--cache_dir", type=str, default="./cache", help="Directory to cache models.")
    parser.add_argument("--hf_token", type=str, default=None, help="Hugging Face token for model download.")
    parser.add_argument("--max_total_tokens", type=int, default=8192, help="Maximum total tokens.")
    parser.add_argument("--max_input_tokens", type=int, default=8000, help="Maximum input tokens.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output.")
    args = parser.parse_args()

    server = TGIServer(
        model_name=args.model_name,
        model_archive=args.model_archive,
        device_ids=args.device_ids,
        port=args.port,
        gpu_memory=args.gpu_memory,
        cache_dir=args.cache_dir,
        hf_token=args.hf_token,
        max_total_tokens=args.max_total_tokens,
        max_input_tokens=args.max_input_tokens
    )

    server.prepare_model()
    server.start(args.verbose)
    input("Press Enter to stop the server")
    print("Stopping server")
    server.stop()
