import random
from typing import Dict, List, Optional, Union
from collections.abc import Iterable

import dsp
from dspy.predict.parameter import Parameter
from dspy.primitives.prediction import Prediction


def retrieve(query: str, k: int, **kwargs) -> list[str]:
    """Retrieves passages from the RM for the query and returns the top k passages."""
    if not dsp.settings.rm:
        raise AssertionError("No RM is loaded.")
    passages = dsp.settings.rm(query, k=k, **kwargs)
    if not isinstance(passages, Iterable):
        # it's not an iterable yet; make it one.
        # TODO: we should unify the type signatures of dspy.Retriever
        passages = [passages]
    return passages


def retrieveEnsemble(
    queries: list[str], k: int, by_prob: bool = True, **kwargs
) -> list[str]:
    """Retrieves passages from the RM for each query in queries and returns the top k passages
    based on the probability or score.
    """
    queries = [q for q in queries if q]

    if len(queries) == 1:
        return retrieve(queries[0], k, **kwargs)

    passages = {}
    for q in queries:
        for psg in dsp.settings.rm(q, k=k * 3, **kwargs):
            if by_prob:
                passages[psg.long_text] = passages.get(psg.long_text, 0.0) + psg.prob
            else:
                passages[psg.long_text] = passages.get(psg.long_text, 0.0) + psg.score

    passages = [(score, text) for text, score in passages.items()]
    passages = sorted(passages, reverse=True)[:k]

    return passages


def single_query_passage(passages):
    passages_dict = {key: [] for key in list(passages[0].keys())}
    for docs in passages:
        for key, value in docs.items():
            passages_dict[key].append(value)
    if "long_text" in passages_dict:
        passages_dict["passages"] = passages_dict.pop("long_text")
    return Prediction(**passages_dict)


class RetrieveWithScore(Parameter):
    name = "Search"
    input_variable = "query"
    desc = "takes a search query and returns one or more potentially relevant passages from a corpus"

    def __init__(self, k=3):
        self.stage = random.randbytes(8).hex()
        self.k = k

    def reset(self):
        pass

    def dump_state(self):
        state_keys = ["k"]
        return {k: getattr(self, k) for k in state_keys}

    def load_state(self, state):
        for name, value in state.items():
            setattr(self, name, value)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(
        self,
        query_or_queries: Union[str, List[str]],
        k: Optional[int] = None,
        by_prob: bool = True,
        with_metadata: bool = False,
        **kwargs,
    ) -> Union[List[str], Prediction, List[Prediction]]:
        # queries = [query_or_queries] if isinstance(query_or_queries, str) else query_or_queries
        # queries = [query.strip().split('\n')[0].strip() for query in queries]

        # # print(queries)
        # # TODO: Consider removing any quote-like markers that surround the query too.
        # k = k if k is not None else self.k
        # passages = dsp.retrieveEnsemble(queries, k=k,**kwargs)
        # return Prediction(passages=passages)
        queries = (
            [query_or_queries]
            if isinstance(query_or_queries, str)
            else query_or_queries
        )
        queries = [query.strip().split("\n")[0].strip() for query in queries]

        # print(queries)
        # TODO: Consider removing any quote-like markers that surround the query too.
        k = k if k is not None else self.k
        if not with_metadata:
            passages = retrieveEnsemble(queries, k=k, by_prob=by_prob, **kwargs)
            return Prediction(passages=passages)
        else:
            passages = dsp.retrieveEnsemblewithMetadata(
                queries,
                k=k,
                by_prob=by_prob,
                **kwargs,
            )
            if isinstance(passages[0], List):
                pred_returns = []
                for query_passages in passages:
                    passages_dict = {
                        key: []
                        for key in list(query_passages[0].keys())
                        if key != "tracking_idx"
                    }
                    for psg in query_passages:
                        for key, value in psg.items():
                            if key == "tracking_idx":
                                continue
                            passages_dict[key].append(value)
                    if "long_text" in passages_dict:
                        passages_dict["passages"] = passages_dict.pop("long_text")
                    pred_returns.append(Prediction(**passages_dict))
                return pred_returns
            elif isinstance(passages[0], Dict):
                # passages dict will contain {"long_text":long_text_list,"metadatas";metadatas_list...}
                return single_query_passage(passages=passages)
