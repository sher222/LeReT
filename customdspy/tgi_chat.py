import os
import random
import re
import shutil
import subprocess
from typing import Literal
## probably just want to open pr for tgi chat in dspy
# from dsp.modules.adapter import TurboAdapter, DavinciAdapter, LlamaAdapter
import backoff
import requests

from dsp.modules.cache_utils import CacheMemory, NotebookCacheMemory
from dsp.modules.hf import HFModel, openai_to_hf
from dsp.utils.settings import settings

ERRORS = Exception


def backoff_hdlr(details):
    """Handler from https://pypi.org/project/backoff/"""
    print(
        "Backing off {wait:0.1f} seconds after {tries} tries "
        "calling function {target} with kwargs "
        "{kwargs}".format(**details),
    )


class TGIChat(HFModel):
    def __init__(self, model, port, url="http://future-hgx-1", http_request_kwargs=None, max_tokens=512, **kwargs):
        super().__init__(model=model, is_client=True)
        self.model = model
        self.url = url
        self.ports = port if isinstance(port, list) else [port]
        self.http_request_kwargs = http_request_kwargs or {}

        self.headers = {"Content-Type": "application/json"}
        self.max_tokens = max_tokens
        self.kwargs = {
            "model": model,
            "port": port,
            "url": url,
            "temperature": 0.01,
            "max_tokens": 75,
            "top_p": 0.97,
            "n": 1,
            "stop": ["\n", "\n\n"],
            **kwargs,
        }

        # print(self.kwargs)

    def _generate(self, prompt, **kwargs):
        kwargs = {**self.kwargs, **kwargs}
        if "gemma" in self.model:
            messages = [{
                "role": "user",
                "content": "You are a helpful assistant. You must continue the user text directly without *any* additional interjections." + prompt
            }]
        else:
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant. You must continue the user text directly without *any* additional interjections."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        payload = {
            "stream": False,
            "max_tokens": self.max_tokens,
            "messages": messages,
            "parameters": {
                "do_sample": kwargs["n"] > 1,
                "best_of": kwargs["n"],
                "details": kwargs["n"] > 1,
                # "max_new_tokens": kwargs.get('max_tokens', kwargs.get('max_new_tokens', 75)),
                # "stop": ["\n", "\n\n"],
                **kwargs,
            },
        }

        payload["parameters"] = openai_to_hf(**payload["parameters"])

        payload["parameters"]["temperature"] = max(
            0.1,
            payload["parameters"]["temperature"],
        )

        # print(payload['parameters'])

        # response = requests.post(self.url + "/generate", json=payload, headers=self.headers)
        response = send_hftgi_request_v01_wrapped(
            f"{self.url}:{random.Random().choice(self.ports)}" + "/v1/chat/completions",
            url=self.url,
            ports=tuple(self.ports),
            json=payload,
            headers=self.headers,
            **self.http_request_kwargs,
        )

        try:
            json_response = response.json()
            # completions = json_response["generated_text"]
            
            completions = [i['message']['content'] for i in json_response["choices"]]


            response = {"prompt": prompt, "choices": [{"text": c} for c in completions]}
            return response
        except Exception as e:
            print("Failed to parse JSON response:", response.text)
            raise Exception("Received invalid JSON response from server")


def send_hftgi_request_v01(arg, url, ports, **kwargs):
    # print("ARG", arg, "kwargs", kwargs)
    return requests.post(arg, **kwargs)


# @functools.lru_cache(maxsize=None if cache_turn_on else 0)
def send_hftgi_request_v01_wrapped(arg, url, ports, **kwargs):
    return send_hftgi_request_v01(arg, url, ports, **kwargs)


def send_hftgi_request_v00(arg, **kwargs):
    return requests.post(arg, **kwargs)