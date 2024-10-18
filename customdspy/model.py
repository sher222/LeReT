import dspy
from customdspy.retriever_with_score import RetrieveWithScore


class SingleHop(dspy.Module):
    def process(self, passages):
        s = sorted(passages, key=lambda x: x["score"], reverse=True)
        already_seen = {}
        ret = []
        for i in s:
            if i["long_text"] not in already_seen:
                ret.append(i)
                already_seen[i["long_text"]] = 1
        return ret

    def __init__(self, passages_per_hop=3):
        super().__init__()

        self.retrieve = RetrieveWithScore(k=passages_per_hop)
        self.generate_query = [
            dspy.ChainOfThought("context, question -> search_query") for _ in range(1)
        ]

    def forward(self, question, context=[]):
        if context is None:
            context = []
        for hop in range(1):
            text_context = [i["long_text"] for i in context]
            search_query = self.generate_query[hop](
                context=text_context, question=question
            ).search_query
            resp = self.retrieve(search_query)
            passages = [
                {"long_text": i["long_text"], "score": i["score"]}
                for i in resp.passages
            ]

            context = self.process(context + passages)

        return {
            "search_query": search_query,
            "passages": context,
            "passages_this_hop": passages,
        }


def save_model(prog, path):
    state_str = str(prog.dump_state())
    serialized = state_str.replace("Example", "dspy.Example").replace(
        "(input_keys=None)", ""
    )
    with open(path, "w+") as f:
        f.write(serialized)
