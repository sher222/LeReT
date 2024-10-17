import pandas as pd
from dspy.datasets.dataset import Dataset
import datasets
import dspy
import random

class HoverRetrievalDataset(Dataset):
    def _process_data(self, split, num_iterations=1, iteration=0):
        dataset = self.dataset[split]
        ret = []
        for i in dataset:
            data = {**i}
            data["complete_answer"] = "SUPPORTED" if data["label"] == 1 else "NOT_SUPPORTED"
            answer_set = set()
            for k in data["supporting_facts"]:
                answer_set.add(k["key"])
            data["retrieval_answer"] = list(answer_set)
            data["generator_answer"] = data["complete_answer"]
            data["answer"] = data["retrieval_answer"]
            data["question"] = f"Is the following statement supported: {data['claim']}"
            ret.append(dspy.Example(
                data
            ))
        for i in range(len(ret)):
            ret[i] = ret[i]
        per_iteration = len(ret) / num_iterations
        start = int((iteration) * per_iteration)
        end = int((iteration + 1) * per_iteration)
        
        return ret[start:end]

    def __init__(self, num_iterations=1, traintest_seed=1, iteration=0, cache_dir="./cache", *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.dataset = datasets.load_dataset('hover-nlp/hover', 'fullwiki', cache_dir=cache_dir, trust_remote_code=True)
        random.seed(traintest_seed)
        self._train = self._process_data("train", num_iterations, iteration)
        self._dev = self._process_data("validation")
        self._test = self._process_data("test")

if __name__ == "__main__":
    dataset = HoverRetrievalDataset()
    breakpoint()
