from customdspy.colbertv2_local import ColBERTv2Local
import customdspy.tgi_chat
import dspy
from customdspy.model import SingleHop, save_model
from local_evaluate import LocalEvaluate
from custom_metrics import AP_Metric
from dspy.teleprompt import BootstrapFewShotWithRandomSearch
from customdspy.hotpot_retrieval_dataset import HotpotRetrievalDataset
from customdspy.hover_retrieval_dataset import HoverRetrievalDataset
import customdspy
from tgi_server import TGIServer
import json
from collections import defaultdict
import random
import os
import utils
import argparse

class CreatePreferenceDataset():
    """This class is used to sample a preference dataset according to the prompt driven diverse sampling algorithm."""
    def generate_fewshot(self, threads=20, canidates=4, file_prefix="", out_path=""):
        """generate_fewshot is called to generate few shot prompts. Saves the dspy state dict as well as extracts the prompt from the last history entry."""
        dspy.settings.configure(rm=self.retriever, lm=self.lm)
        if self.dataset_name == "hotpotqa":
            dataset = HotpotRetrievalDataset(train_seed=1, eval_seed=1, num_iterations=self.num_iterations, iteration=self.iteration, cache_dir=self.cache_dir)
        elif self.dataset_name == "hover":
            dataset = HoverRetrievalDataset(train_seed=1, eval_seed=1, num_iterations=self.num_iterations, iteration=self.iteration, cache_dir=self.cache_dir)
        else:
            raise ValueError("Invalid dataset name, must be hotpotqa or hover")
        trainset = [dspy.Example(x).with_inputs('question') for x in dataset.train]
        metric = AP_Metric
        tp = BootstrapFewShotWithRandomSearch(metric=metric, max_bootstrapped_demos=3, num_threads=threads, num_candidate_programs=4 * canidates)
        basicmh_bs = tp.compile(SingleHop(), trainset=trainset[:50], valset=trainset[50:100])
        all_ensembles = [prog for *_, prog in basicmh_bs.candidate_programs]
        ensemble = []
        for i in all_ensembles:
            if "Example" not in str(i.dump_state()):
                continue
            ensemble.append(i)
        ensemble = ensemble[:canidates] 
        ensemble.insert(0, SingleHop())
        out_paths = []
        out_template = f'{out_path}/{self.model_name.replace("/", "_")}_{self.dataset_name}_it{self.iteration}_of{self.num_iterations}/{file_prefix}bootstrapped'
        
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(out_template), exist_ok=True)
        
        for idx, prog in enumerate(ensemble):
            out_path = f'{out_template}_{idx}.json'
            save_model(prog, out_path)
            out_paths.append(out_path)
        print(f"saved models to {out_paths[0]}")
        
        prompts = {}
        for index, e in enumerate(ensemble):
            try:
                e("{PROMPT}")
            except AssertionError:
                pass
            prompts[f"ensemble{index}"] = self.lm.history[-1]["prompt"]
            prompt_path = f"{out_template}_prompts.json"
            with open(prompt_path, "w+") as f:
                json.dump(prompts, f)
        print(f"saved prompts to {prompt_path}")
        return out_paths, prompt_path


    def __init__(self, model_name, model_archive=None, num_iterations=1, iteration=0, tgi_device_ids="0", tgi_server=True, dataset_name="hotpotqa", tgi_gpu_memory=1, local_retriever=True, together_api_key=None, tgi_verbose=False, cache_dir="./cache"):
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.tgi_server = tgi_server
        self.cache_dir = cache_dir
        if self.tgi_server:
            self.tgiserver = TGIServer(device_ids=tgi_device_ids, model_archive=model_archive, model_name=model_name, gpu_memory=tgi_gpu_memory)
            self.tgiserver.prepare_model()
            self.tgiserver.start(verbose=tgi_verbose)
            self.lm = customdspy.tgi_chat.TGIChat(model=self.model_name, port=3002, url=f"http://localhost")
        else:
            self.lm = dspy.Together(model=utils.get_together_model_name(self.model_name), max_tokens=1024, stop=["<|eot_id|>"], api_key=together_api_key)

        self.num_iterations = num_iterations
        self.iteration = iteration
        self.retriever =  ColBERTv2Local() if local_retriever else dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')


    def __del__(self):
        if self.tgi_server:
            self.tgiserver.stop()
    
    def get_prompt(self, obj, prompt_file, context=True, step_key_n = -1):
        with open(prompt_file, "r") as f:
            prompts = json.load(f)
            prompt = prompts[f"ensemble0"]
        retrieved = []
        for step in range(1, step_key_n + 1):
            retrieved.extend(obj[f"hop{step}"]["retrieved"])
        retrieved = sorted(retrieved, key=lambda x: x['score'], reverse=True)
        if context and len(retrieved) > 0:
            context_str = "Context:\n"
            for i in range(0, len(retrieved)):
                context_str += f"[{i+1}] «{retrieved[i]['long_text']}»\n"
            return prompt.replace("Context: N/A\n\nQuestion: {PROMPT}", f"{context_str}\n\nQuestion: {obj['question']}")
        return prompt.replace("{PROMPT}", obj['question'])
    
    def make_nth_dataset(self, n, data_path, ensemble_filepaths, prompt_file, out_file, preference_out_file, num_threads=64):
        
        ### sample data
        file_template = f"n{n}_sampled_{self.model_name}_{self.iteration}_ensemble"

        evaluate = LocalEvaluate(self.lm, self.retriever, model=self.model_name, dataset_name=self.dataset_name, cache_dir=self.cache_dir)

        data = []
        for i, filepath in enumerate(ensemble_filepaths):
            evaluate.update_fewshot(filepath)
            filepath_end = filepath.split("/")[-1].split(".json")[0]
            _, out, _, _ = evaluate.evaluate(data_path, "train", output_prefix=f"{file_template}_{filepath_end}", num_iterations=self.num_iterations, iteration=self.iteration, num_threads=num_threads, hop_number=n)
            data.append(out)
            
        ### create next hop dataset
        dataset = defaultdict(list)
        
        for index, arr in enumerate(data):
            for j in arr:
                dataset[j["example_id"]].append(
                    {
                        **j,
                        "model": index
                    }
                )   

        next_hop_data = []
        hop_key = f"hop{n}"
        for i in dataset:
            grouped_by_score = defaultdict(list)
            for j in dataset[i]:
                
                if j[hop_key]["score"]["ap"] == 1:
                    continue
                grouped_by_score[j[hop_key]["score"]["ap"]].append(j)
            if len(grouped_by_score) == 0:
                continue
            all_keys = sorted(list(grouped_by_score.keys()))
            weighted_scores = all_keys.copy()
            weighted_scores[-1] += 0.5 # increase prob of highest score
            key = random.choices(all_keys, weights=weighted_scores, k=1)[0]
            next_hop_chosen = random.choice(grouped_by_score[key])
            next_hop_data.append(next_hop_chosen)
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        with open(out_file, "w+") as f:
            json.dump(next_hop_data, f)
        print(f"saved next hop data for step {n} to", out_file, len(next_hop_data))

        ### create preference dataset
        
        dataset = defaultdict(list)
        
        for i, arr in enumerate(data):
            for j in arr:
                p = self.get_prompt(j, prompt_file, step_key_n=n - 1)
                dataset[p].append(
                    {
                        **j,
                        "model": i,
                        "prompt": p
                    }
                )
        
        preference_data = defaultdict(lambda: defaultdict(list))
        dpo_size = 0
        for k, v in dataset.items():
            if len(set([i[hop_key]["score"]["ap"] for i in v])) == 1:
                continue
            if len(v) == 0:
                continue
            prompt = k
            sorted_v = sorted(v, key=lambda x: x[hop_key]["score"]["ap"])
            sorted_v = list(filter(lambda x: "Context: [1]" not in x[hop_key]["search_query"], sorted_v))
            if len(sorted_v) == 0:
                continue
            preference_data[prompt]["responses"] = [sorted_v[i][hop_key]["search_query"] for i in range(len(sorted_v))]
            
            preference_data[prompt]["pairs"] = []
            grouped_by_score = defaultdict(list)
            seen = {}
            for index, i in enumerate(sorted_v):
                if i[hop_key]["search_query"] in seen:
                    continue
                seen[i[hop_key]["search_query"]] = 1
                grouped_by_score[i[hop_key]["score"]["ap"]].append(index)
            scores = sorted(list(grouped_by_score.keys()))
            if len(scores) > 1:
                for i in range(len(scores) - 1):
                    for j in grouped_by_score[scores[i]]:
                        for k in grouped_by_score[scores[i + 1]]:
                            preference_data[prompt]["pairs"].append((k, j))
                            dpo_size += 1
            
            preference_data[prompt]["sft_target"] = sorted_v[-1][hop_key]["search_query"]
        
        with open(preference_out_file, "w+") as f:
            json.dump(preference_data, f)
        print(f"Save {n} step preference dataset to", preference_out_file)

        return data
    
    def combine_datasets(self, list, out_file):
        """take list of preference datasets (from each hop) and create one final dataset"""
        combined_data = {}
        for file in list:
            with open(file, "r") as f:
                data = json.load(f)
                print(file, len(data))
                combined_data.update(data)
        with open(out_file, "w+") as f:
            json.dump(combined_data, f)
        print("saved combined dataset to", out_file)
        return out_file
    
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Create a preference dataset using prompt driven diverse sampling algorithm.")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to use.")
    parser.add_argument("--model_archive", type=str, default=None, help="Path to the model archive.")
    parser.add_argument("--num_iterations", type=int, default=1, help="Number of iterations.")
    parser.add_argument("--iteration", type=int, default=0, help="Current iteration.")
    parser.add_argument("--tgi_device_ids", type=str, default="0", help="Device IDs for TGI server.")
    parser.add_argument("--tgi_server", type=bool, default=True, help="Whether to use TGI server.")
    parser.add_argument("--dataset_name", type=str, default="hotpotqa", choices=["hotpotqa", "hover"], help="Name of the dataset.")
    parser.add_argument("--tgi_gpu_memory", type=int, default=1, help="GPU memory for TGI server.")
    parser.add_argument("--together_api_key", type=str, default=None, help="API key for Together.")
    parser.add_argument("--tgi_verbose", type=bool, default=False, help="Verbose mode for TGI server.")
    parser.add_argument("--cache_dir", type=str, default="./cache", help="Cache directory.")
    parser.add_argument("--out_file", type=str, required=True, help="Output file path.")
    parser.add_argument("--num_hops", type=int, default=None, help="Number of hops")
    parser.add_argument("--hosted_retriever", action='store_true', help="Use hosted retriever instead of local.")
    parser.add_argument("--num_threads", type=int, default=64, help="Number of threads to use.")
    args = parser.parse_args()

    cp = CreatePreferenceDataset(
        model_name=args.model_name,
        model_archive=args.model_archive,
        num_iterations=args.num_iterations,
        iteration=args.iteration,
        tgi_device_ids=args.tgi_device_ids,
        tgi_server=args.tgi_server,
        dataset_name=args.dataset_name,
        tgi_gpu_memory=args.tgi_gpu_memory,
        local_retriever=not args.hosted_retriever,
        together_api_key=args.together_api_key,
        tgi_verbose=args.tgi_verbose,
        cache_dir=args.cache_dir
    )
    out_file = args.out_file
    num_hops = args.num_hops
    if num_hops is None:
        num_hops = 4 if args.dataset_name == "hover" else 2
    ensemble_filepaths, prompt_path = cp.generate_fewshot(canidates=3, threads=args.num_threads, out_path=out_file)

    for i in range(1, num_hops+ 1):
        data_file = args.dataset_name if i == 1 else out_file + f"hop{i-1}.json"
        data = cp.make_nth_dataset(i, data_file, ensemble_filepaths, prompt_path, out_file + f"hop{i}.json", out_file + f"hop{i}_preference.json", num_threads=args.num_threads)
    preference_files = [out_file + f"hop{i}_preference.json" for i in range(1, num_hops + 1)]
    combined_dataset = cp.combine_datasets(preference_files, out_file + "combined.json")
