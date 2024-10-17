import pandas as pd
from dspy.datasets.dataset import Dataset
import datasets
import dspy
import random

class HotpotRetrievalDataset(Dataset):
    def _process_data(self, split, num_iterations=1, iteration=0):
        dataset = self.dataset[split]
        ret = []
        for i in dataset:
            data = {**i}
            data["complete_answer"] = data["answer"]
            data["retrieval_answer"] = set()
            title_to_index = {b: a for a, b in enumerate(data["context"]["title"])}
            for j, title in enumerate(data["supporting_facts"]["title"]):
                if title not in title_to_index:
                    continue
                index = title_to_index[title]
                sentence_index = data["supporting_facts"]["sent_id"][j]
                if sentence_index < len(data["context"]["sentences"][index]):
                    data["retrieval_answer"].add(data["context"]["sentences"][index][sentence_index])
            data["answer"] = list(set(data['supporting_facts']['title']))
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
        self.dataset = datasets.load_dataset('hotpotqa/hotpot_qa', 'fullwiki', cache_dir=cache_dir, trust_remote_code=True)
        random.seed(traintest_seed)
        self._train = self._process_data("train", num_iterations, iteration)
        self._dev = self._process_data("validation")
        self._test = self._process_data("test")
        
if __name__ == "__main__":
    dataset = HotpotRetrievalDataset()
    breakpoint()
