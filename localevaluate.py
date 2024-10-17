import os
import dspy
from dspy.evaluate import Evaluate
from customdspy.hotpotretrievaldataset import HotpotRetrievalDataset
from customdspy.hoverretrievaldataset import HoverRetrievalDataset
from customdspy.tgichat import TGIChat
from customdspy.colbertv2local import ColBERTv2Local
import tqdm
import json
from collections import defaultdict
from customdspy.model import SingleHop
from custommetrics import EM_Metric, AP_Metric


class LocalEvaluate():
    """
    This class is used to evaluate a model. You need to prove a language model and retriever. Dataset_name is the name used for the output, dataset_filepath is the actual dataset. Evaluate runs one hop. For the first hop, you can pass in hotpotqa or hover and it will use that dataset, for subsequent hops you should pass in the output file from the previous hop. You should also pass in the appropriate hop_number so it can add the correct hop to the file (otherwise it will overwrite)."""
    def __init__(self, lm, retriever, model="meta-llama/Meta-Llama-3-8B-Instruct", fewshot_path=None, dataset_name="hotpotqa", cache_dir="./cache"):
        self.model = model
        self.dataset_name = dataset_name
        self.cache_dir = cache_dir
        self.lm = lm
        self.retriever = retriever
        dspy.settings.configure(rm=self.retriever, lm=self.lm)
        self.prog = SingleHop()
        if fewshot_path is not None:
            self.update_fewshot(fewshot_path)
    
    def update_lm(self, lm):
        self.lm = lm
        
    def update_fewshot(self, fewshot_path):
        print("loaded fewshot state from", fewshot_path)
        with open(fewshot_path, "r") as f:
            state = eval(f.read())
        self.prog.load_state(state)

    def evaluate(self, dataset_filepath, split, num_threads=32, output_prefix=None, num_iterations=1, iteration=0, hop_number=1, save_every=50_000, overwrite=False):
        dataset_output_name = f"{self.dataset_name}_it{iteration}_of{num_iterations}"
        if output_prefix is None:
            output_filepath = f"{self.cache_dir}/{dataset_output_name}_{split}.json"
        elif output_prefix[0] == "/":
            output_filepath = f"{output_prefix}_{split}.json"
        else:
            output_filepath = f"{self.cache_dir}/{output_prefix}_{dataset_output_name}_{split}.json"
        if not os.path.exists(os.path.dirname(output_filepath)):
            os.makedirs(os.path.dirname(output_filepath))
        print("OUT_FILE", output_filepath)
        score_filepath = output_filepath.replace(".json", "_scores.json")

        if dataset_filepath in ["hotpotqa", "hover"]:
            dataset_args = {
                "train_seed": 1,
                "eval_seed": 1,
                "num_iterations": num_iterations,
                "iteration": iteration,
                "cache_dir": self.cache_dir
            }
            dataset = HotpotRetrievalDataset(**dataset_args) if dataset_filepath == "hotpotqa" else HoverRetrievalDataset(**dataset_args)
            sets = {
                "train": dataset.train,
                "dev": dataset.dev,
                "test": dataset.test,
            }
                
            examples = [dspy.Example({**i, "example_id": i["id"]}) for i in sets[split]]
            selected_dataset = [i.with_inputs('question') for i in examples]
        else:
            with open(dataset_filepath, "r") as f:
                data = json.load(f)
            dataset = [dspy.Example({**i, "context": i["full_retrieved"]}) for i in data]
            selected_dataset = [x.with_inputs('question', 'context') for x in dataset]

        
        dspy.settings.configure(rm=self.retriever, lm=self.lm)
        already_done = 0
        out = []
        if not overwrite:
            try:
                out = json.load(open(output_filepath))
                already_done = len(out)
                print("already_done", already_done, "len out", len(out))
            except Exception:
                already_done = 0
        
        for j in tqdm.tqdm(range(already_done, len(selected_dataset), save_every), desc=f"grouped by {save_every}"):   
            print(f"for outfile {output_filepath}: on {j} out of {len(selected_dataset)}")
            evaluate_hotpot = Evaluate(devset=selected_dataset[j:j+save_every], metric=EM_Metric, num_threads=num_threads, display_progress=True, display_table=0, return_outputs=True)
            sub_scores, sub_examples = evaluate_hotpot(self.prog)
            
            for k in sub_examples:
                if 'search_query' not in k[1]:
                    continue
                hop_dict = {
                    f"hop{hop_number}": {
                        "search_query": k[1]["search_query"],
                        "score": {
                            "em": k[2],
                            "ap": AP_Metric(k[0], k[1])
                        },
                        "retrieved": k[1]["passages_this_hop"],
                    },
                    "full_retrieved": k[1]["passages"],
                }
                if hop_number == 1:
                    out.append({
                        "example_id": k[0].example_id,
                        "question": k[0].question,
                        "answer": k[0].answer,
                        **hop_dict
                    })
                else:
                    out.append({
                        **k[0],
                        **hop_dict
                    })
            print("wrote", len(out), "to", output_filepath)
            json.dump(out, open(output_filepath, "w+"))
            
        score_dict = defaultdict(int)
        for k in out:
            hop = "hop" + str(hop_number)
            for k, v in k[hop]["score"].items():
                score_dict[k] += v
            score_dict["count"] += 1
        for k, v in score_dict.items():
            if k == "count":
                continue
            score_dict[k] = v / score_dict["count"]
        with open(score_filepath, "a+") as f:
            json.dump(score_dict, f)
        print(score_dict)
        return output_filepath, out, score_filepath, score_dict

