from typing import Any, List, Optional, Union


from dsp.modules.cache_utils import CacheMemory, NotebookCacheMemory
from dsp.utils import dotdict
from colbert import Searcher

# TODO: Ideally, this takes the name of the index and looks up its port.


class ColBERTv2Local:
    """Wrapper for the ColBERTv2 Retrieval."""

    def __init__(
        self,
        index_root: str = "/iris/u/sherylh/rl-rag/dspy/experiments/ColBERT/default/indexes/",
        index_name: str = "wiki17.nbits.local",
    ):
        self.searcher = Searcher(index=index_name, index_root=index_root)


    # taken from https://github.com/stanford-futuredata/ColBERT/blob/main/server.py
    def api_search_query(self, query, k):
        if k == None: k = 3
        k = min(int(k), 100)
        pids, ranks, scores = self.searcher.search(query, k=k)
        pids, ranks, scores = pids[:k], ranks[:k], scores[:k]
        topk = []
        for pid, rank, score in zip(pids, ranks, scores):
            text = self.searcher.collection[pid]            
            d = {'long_text': text, 'pid': pid, 'rank': rank, 'score': score}
            topk.append(d)
        topk = list(sorted(topk, key=lambda p: (-1 * p['score'], p['pid'])))
        return topk

    def __call__(
        self, query: str, k: int = 10, simplify: bool = False,
    ) -> Union[list[str], list[dotdict]]:
        topk = self.api_search_query(query, k)
        if simplify:
            return [psg["long_text"] for psg in topk]

        return [dotdict(psg) for psg in topk]
