from local_evaluate import LocalEvaluate
from tgi_server import TGIServer
import argparse
import customdspy
from customdspy.colbertv2_local import ColBERTv2Local
import dspy
import utils
class RunEvals():
    """Runs evals on a list of model directories. Specifically, models is a list of models. Each model can be "base" for the base model, "prog:/path/to/dspy/state" to evaluate fewshot prompts, or a path to the directory containing the model which then needs to be run locally with tgi_server."""
    
    def __init__(self, models: list, splits: str="dev", tgi_device_ids="0", model_name="meta-llama/Meta-Llama-3-8B-Instruct", output_prefix=None, gpu_memory=1, tgi_verbose=False, port=3002, dataset_name="hotpotqa", tgi_server=True, num_threads=128, together_api_key=None, local_retriever=True, cache_dir="./cache"):
        retriever =  ColBERTv2Local() if local_retriever else dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')

        for models_counter, model_archive in enumerate(models):
            evaluate = LocalEvaluate(lm=None, retriever=retriever, model=model_name, dataset_name=dataset_name, cache_dir=cache_dir)

            print("on model", model_name, "count", models_counter, "out of", len(models))
            tgi_args = {"device_ids": tgi_device_ids, "model_name": model_name, "gpu_memory": gpu_memory, "port": port}
            if model_archive == "base":
                output_name = "base"
            elif "prog" in model_archive:
                prog_name = model_archive.split("prog:")[1].strip()
                evaluate.update_fewshot(prog_name)
                output_name = model_archive
                if "/models/" in output_name:
                    output_name = model_archive.split("/models/")[-1]
                if "." in output_name:
                    output_name = output_name.rsplit(".", 1)[0]
            else:
                output_name = model_name.replace("/", "_") if model_archive is None else model_archive.replace("/", "_")
                if tgi_server:
                    tgi_args["model_archive"] = model_archive
                else:
                    raise Exception("TGI server must be enabled to evaluate custom models.")
            if tgi_server:
                tgiserver = TGIServer(**tgi_args)
                tgiserver.prepare_model()
                tgiserver.start(verbose=tgi_verbose)
                print("tgi server started")
                evaluate.update_lm(customdspy.tgichat.TGIChat(model=model_name, port=port, url=f"http://localhost"))
            else:
                evaluate.update_lm(dspy.Together(model=utils.get_together_model_name(model_name), max_tokens=1024, stop=["<|eot_id|>"], api_key=together_api_key))
                

            if output_prefix is not None:
                output_name = output_prefix + "_" + output_name
            for split in splits.split(","):
                num_hops = 4 if dataset_name == "hover" else 2
                # first hop
                file, _, _, _ = evaluate.evaluate(dataset_name, split, output_prefix=f"{output_name}/hop1", hop_number=1, num_threads=num_threads)
                # subsequent hops
                for hop in range(2, num_hops + 1):
                    file, _, _, _ = evaluate.evaluate(file, split, output_prefix=f"{output_name}/hop{hop}", hop_number=hop, num_threads=num_threads)
            
            if tgi_server:
                tgiserver.stop()
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evaluations for models.")
    parser.add_argument("--models", type=str, required=True, help="List of models to evaluate.")
    parser.add_argument("--splits", type=str, default="dev", help="Dataset splits to evaluate on.")
    parser.add_argument("--tgi_device_ids", type=str, default="unknown", help="Device IDs for TGI server.")
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct", help="Name of the model.")
    parser.add_argument("--output_prefix", type=str, help="Prefix for output files.")
    parser.add_argument("--gpu_memory", type=float, default=1.0, help="GPU memory allocation.")
    parser.add_argument("--tgi_verbose", action='store_true', help="Enable verbose logging for TGI server.")
    parser.add_argument("--port", type=int, default=3002, help="Port for TGI server.")
    parser.add_argument("--dataset_name", type=str, default="hotpotqa", help="Name of the dataset, hotpotqa or hover.")
    parser.add_argument("--tgi_server", action='store_true', help="Use TGI server.")
    parser.add_argument("--num_threads", type=int, default=128, help="Number of threads for evaluation.")
    parser.add_argument("--together_api_key", type=str, help="API key for Together.")
    parser.add_argument("--hosted_retriever", action='store_true', help="Use hosted retriever instead of local.")
    parser.add_argument("--cache_dir", type=str, default="./cache", help="Directory for cache.")
    args = parser.parse_args()

    run_evals = RunEvals(
        models=args.models.split(","),
        splits=args.splits,
        tgi_device_ids=args.tgi_device_ids,
        model_name=args.model_name,
        output_prefix=args.output_prefix,
        gpu_memory=args.gpu_memory,
        tgi_verbose=args.tgi_verbose,
        port=args.port,
        dataset_name=args.dataset_name,
        tgi_server=args.tgi_server,
        num_threads=args.num_threads,
        together_api_key=args.together_api_key,
        local_retriever=not args.hosted_retriever,
        cache_dir=args.cache_dir
    )