import os

HF_TO_TOGETHER_MODEL_NAME = {
    "meta-llama/Meta-Llama-3-8B-Instruct": "meta-llama/Llama-3-8b-chat-hf"
}


def get_together_model_name(model_name):
    return HF_TO_TOGETHER_MODEL_NAME.get(model_name, model_name)


def find_directories_with_prefix(directory, prefix):
    matching_directories = []
    for entry in os.listdir(directory):
        full_path = os.path.join(directory, entry)
        if os.path.isdir(full_path) and entry.startswith(prefix):
            matching_directories.append(full_path)
    return sorted(matching_directories)[-1]
