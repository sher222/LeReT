# LeReT: Learning to Retrieve by Trying

Code for Grounding by Trying: LLMs with Reinforcement Learning-Enhanced Retrieval


## What is in this repo?

There are three main portions to the codebase - creating the preference dataset as done by createpreferencedataset.py, training which is done by the direct-preference-optimization codebase, and finally evaluation done by localevaluate.py. To do this, there are quite a few custom dspy components needed and a wrapper around tgiserver which can be found in customdspy/.

The codebase is built on top of [DSPy](https://github.com/stanfordnlp/dspy) for setting up pipelines, [TGI](https://huggingface.co/docs/text-generation-inference/en/index) for sampling from trained models, and [Eric Mitchell's DPO codebase](https://github.com/eric-mitchell/direct-preference-optimization) for training. 

## Setup
Create a venv with requirements listed in requirements.txt.

Additionally, set the HF_TOKEN and WANDB_API_KEY environment variables.

## Infastructure
The codebase has the option to sample LLMs using Together or locally with TGI. Obviously, TGI is necessary for sampling from trained models. The codebase also has the option to run ColBERTv2 locally versus query a endpoint. For large datasets, it is recommended to run the retriever locally to avoid overloading the endpoint.
  
## Sampling a preference dataset

To sample a preference dataset, use the CreatePreferenceDataset class defined in `createpreferencedataset.py`. Sample usage
```
 python3 createpreferencedataset.py --model_name meta-llama/Meta-Llama-3-8B-Instruct --togher_api_key YOUR_API_KEY --out_file /MYOUTFILE/LeReT_demo/ --num_threads 64
```
It is generally recommended to use Together and host ColBERT locally for the best speed. If you are recieving topk errors, the remote ColBERT server is likely overloaded. Adjust the number of threads appropriately. Generally, the num_iterations and iteration variables are used to partition the dataset into num_iterations segments and return the iteration-th segment, it does not actually run LeReT for multiple iterations. 
## Training a model
To train a model, use the direct-preference-optimization codebase. We added a dataloader so you can pass in the path to the preference dataset as a dataset. 
```
python -u direct-preference-optimization/train.py model=gemma-9b datasets=[PATH_TO_SAMPLED_DATASET.json] n_epochs=1 loss=sft lr=1e-7 exp_name=gemma9b_sft trainer=FSDPTrainer sample_during_eval=false eval_every=1_000_000  do_first_eval=false debug=false wandb.project=rl-hotpotqa-finalize batch_size=8 max_prompt_length=2048 max_length=2048
python -u direct-preference-optimization/train.py model=gemma-9b datasets=[PATH_TO_SAMPLED_DATASET.json] n_epochs=2 loss=ipo lr=1e-7 loss.beta=0.05 exp_name=gemma9b_ipo trainer=FSDPTrainer sample_during_eval=false eval_every=1_000_000  do_first_eval=false debug=false wandb.project=rl-hotpotqa-finalize batch_size=4 max_prompt_length=2048 max_length=2048 model.archive=/PATH_TO_SFT_OUTPUT/LATEST/policy.pt
```
## Evaluating
To evaluate your trained model and baselines, use `runevals.py`. Pass in a list of paths to model weights and strings of the form `prog:path_to_fewshot_dspy_saved_state`. 
```
python tgiserver.py --models /PATH_TO_TRAINED_MODEL/LATEST,prog:/PATH_TO_FEWSHOT_DSPY_STATE.json --splits dev --dataset_name hotpotqa --model_name meta-llama/Meta-Llama-3-8B-Instruct --tgi_server
```
## Full run
A full sample run with all three steps combined can be found in `sample_run.py`.