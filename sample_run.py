import os

os.environ["HF_TOKEN"] = "YOUR_HF_TOKEN"

from create_preference_dataset import CreatePreferenceDataset
import utils
from run_evals import RunEvals


### Create a preference dataset, we use just 1% of the hotpotqa dataset for demonstration purposes
cp = CreatePreferenceDataset(
    "meta-llama/Meta-Llama-3-8B-Instruct",
    tgi_server=False,
    dataset_name="hotpotqa",
    num_iterations=100,
    iteration=0,
    local_retriever=True,
    cache_dir="/iris/u/sherylh/.cache",
    together_api_key="YOUR_TOGETHER_KEY",
)
out_file = "/your/path/LeReT_sample_run/"

ensemble_filepaths, prompt_path = cp.generate_fewshot(
    canidates=3, num_threads=32, out_path=out_file
)

for i in range(1, 3):
    data_file = "hotpotqa" if i == 1 else out_file + f"hop{i-1}.json"
    data = cp.make_nth_dataset(
        i,
        data_file,
        ensemble_filepaths,
        prompt_path,
        out_file + f"hop{i}.json",
        out_file + f"hop{i}_preference.json",
        num_threads=32,
    )

preference_files = [out_file + f"hop{i}_preference.json" for i in range(1, 3)]
combined_dataset = cp.combine_datasets(preference_files, out_file + "combined.json")


### Train model with SFT + IPO

setup_command = ". activate LeReT; export WANDB_API_KEY=YOUR_WANDB_KEY; export HF_TOKEN=YOUR_HF_TOKEN; cd /path/to/LeReT/direct-preference-optimization"

exp_name_prefix = "LeReT_sample_run"

run_command = f"python -u train.py model=llama3-8b datasets=[{combined_dataset}] n_epochs=1 loss=sft lr=1e-7 exp_name={exp_name_prefix}_sft trainer=FSDPTrainer sample_during_eval=false eval_every=1_000_000  do_first_eval=false debug=false wandb.project=rl-hotpotqa-finalize batch_size=8 max_prompt_length=2048 max_length=2048"
os.system(f"{setup_command}; {run_command}")

directory = (
    "/scr/username"  # check what is in direct-preference-optimization/config.yaml
)
prefix = f"{exp_name_prefix}_sft"
matching_directory = utils.find_directories_with_prefix(directory, prefix)

run_command = f"python -u train.py model=llama3-8b datasets=[{combined_dataset}] n_epochs=2 loss=ipo lr=1e-7 loss.beta=0.05 exp_name={exp_name_prefix}_ipo trainer=FSDPTrainer sample_during_eval=false eval_every=1_000_000  do_first_eval=false debug=false wandb.project=rl-hotpotqa-finalize eval_batch_size=64 batch_size=4 max_prompt_length=2048 max_length=2048 model.archive={matching_directory}/LATEST/policy.pt"

os.system(f"{setup_command}; {run_command}")

### Evaluate model
prefix = f"{exp_name_prefix}_ipo"
matching_directory = utils.find_directories_with_prefix(directory, prefix)
models = [f"prog:{i}" for i in ensemble_filepaths] + [matching_directory + "/LATEST"]
RunEvals(
    models=models,
    splits="dev",
    tgi_device_ids="unknown",
    model_name="meta-llama/Meta-Llama-3-8B-Instruct",
    gpu_memory=0.9,
    dataset_name="hotpotqa",
    cache_dir="./.cache",
    tgi_verbose=True,
    local_retriever=False,
    tgi_server=True,
    num_threads=16
)
