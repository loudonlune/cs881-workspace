#!/usr/bin/env python3

import argparse
import os
import jinja2

from tool.llm.base import LLMBackend, login_to_huggingface
from tool.llm.together_ai import TogetherLLMBackend, together_prompt
from tool.llm.local import LocalLLMBackend
from tool.task2 import CheckThatTask2

from typing import Optional

DEFAULT_HUGGINGFACE_MODEL: str = 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B'
FREE_MODEL: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo"

def are_you_sure(clear_eval_table: bool) -> bool:
    if clear_eval_table:
        usn: str = input("Are you sure you would like to clear the evaluation table? [y/N]?")

        if usn.strip().lower() == 'y':
            return True
    return False

def noop(_: argparse.Namespace) -> int:
    print("No operation was given.")
    return 0


def local_chat_cmd(args: argparse.Namespace) -> int:
    print("Loading model...")

    login_to_huggingface()

    local_model = LocalLLMBackend(args.model_id)
    local_model.initialize()

    print("Prompting the model...")

    response = local_model.query(args.query)

    print("Result:")
    print(response)
    return 0


def together_chat_cmd(args: argparse.Namespace) -> int:
    print("Result:", together_prompt(args.model, args.prompt))
    return 0


def checkthat_task2_cmd(args: argparse.Namespace) -> int:
    if not are_you_sure(args):
        print("User was not sure. Terminating.")
        return 0

    llm: LLMBackend
    model_id: Optional[str] = args.model_id

    if args.backend == "together-ai":
        llm = TogetherLLMBackend(model=model_id or FREE_MODEL)
        llm.initialize()
    elif args.backend ==  "local":
        llm = LocalLLMBackend(model_id or DEFAULT_HUGGINGFACE_MODEL)
        llm.initialize(use_flash=args.use_flash_attn, use_4bit_quant=not args.no_4bit_quant)
    else:
        raise NotImplementedError()

    # Load profile from disk.
    with open(os.path.join(os.curdir, "profiles", f"{args.profile}.jinja2")) as templ_file:
        templ_str = templ_file.read()

    # Construct the task 2 class.
    ctt2 = CheckThatTask2(args.profile, llm, jinja2.Template(templ_str))

    # Delete eval table if flag is set.
    if args.clear_eval_table:
        ctt2.delete_eval_table_file()

    ctt2.initialize_eval_table()

    # Bail out early if the user has told us to stop after initializing the evaluation table.
    if args.init_only:
        print("init-only mode: Not doing anything further.")
        return 0

    if not args.no_query:
        ctt2.train()
        ctt2.fill_eval_table()
    else:
        print("no-query mode: Not training or filling eval table for LLM.")

    # Then, use eval table to determine the METEOR score average across each row.
    meteor_score = ctt2.calculate_meteor_score_avg()

    ctt2.save_eval_table()

    # Print the evaluation result.
    print(f'Meteor score for profile "{ctt2.profile_name}": {meteor_score}')

    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Run AI things.")
    parser.set_defaults(cmd=noop)
    subp = parser.add_subparsers()

    together_chat_req = subp.add_parser('together-chat')

    together_chat_req.add_argument('prompt', type=str, help='Prompt to submit to Together.AI')
    together_chat_req.add_argument('-m', '--model', type=str, default="meta-llama/Llama-3.3-70B-Instruct-Turbo", help='Model ID to use.')
    together_chat_req.set_defaults(cmd=together_chat_cmd)

    checkthat_task2 = subp.add_parser('checkthat-task2', description='Harness to run the CheckThat! Task 2 and run tests on data.')
    checkthat_task2.add_argument('backend', type=str, choices=['together-ai', 'local', 'trained'], help='LLM backend to use.')
    checkthat_task2.add_argument('-m', '--model-id', type=str, default=None, help='Override for the default model. Huggingface ID if local/trained, together-ai for the together-ai backend.')
    checkthat_task2.add_argument('profile', type=str, choices=[
        os.path.splitext(x.name)[0]
        for x in os.scandir(os.path.join(os.curdir, 'profiles'))
        if os.path.splitext(x.name)[1] == '.jinja2'
    ], help='Profile to use.')
    checkthat_task2.add_argument('-c', '--clear-eval-table', action='store_true', help='Deletes the eval table when provided.')
    checkthat_task2.add_argument('-i', '--init-only', action='store_true', help='Only run the initialization of the eval table.')
    checkthat_task2.add_argument('-f', '--use-flash-attn', action='store_true', help='Use flash attention implementation.')
    checkthat_task2.add_argument('-n4', '--no-4bit-quant', action='store_true', help='When set, disables 4 bit quantization.')
    checkthat_task2.add_argument('-nq', '--no-query', action='store_true', help='Do not query the LLM.')
    checkthat_task2.set_defaults(cmd=checkthat_task2_cmd)

    local_chat = subp.add_parser('chat')
    local_chat.add_argument('-m', '--model-id', type=str, default=DEFAULT_HUGGINGFACE_MODEL, help='Model ID to load from hugging face.')
    local_chat.add_argument('query', type=str, help='The prompt to make to the LLM')
    local_chat.set_defaults(cmd=local_chat_cmd)

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    
    return args.cmd(args)

if __name__ == "__main__":
    exit(main())
