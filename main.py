#!/usr/bin/env python3

import argparse

from ast import parse
import os
from together import Together

def noop(_: argparse.Namespace) -> int:
    print("No operation was given.")
    return 0


def get_together_client() -> Together | None:
    if api_key := os.environ.get('TOGETHER_API_KEY'):
        return Together(api_key=api_key)
    else:
        print("Failed to get the API key, which is required for the operation. Set TOGETHER_API_KEY in the environment.")
        return None


def together_prompt(model: str, prompt: str) -> str:
    if client := get_together_client():
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )

        return response.choices[0].message.content
    else:
        print("Error: Failed to get the client.")
        return ""


def together_chat_cmd(args: argparse.Namespace) -> int:
    print("Result:", together_prompt(args.model, args.prompt))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Run AI things.")
    parser.set_defaults(cmd=noop)
    subp = parser.add_subparsers()

    together_chat_req = subp.add_parser('together-chat')

    together_chat_req.add_argument('prompt', type=str, help='Prompt to submit to Together.AI')
    together_chat_req.add_argument('-m', '--model', type=str, default="meta-llama/Llama-3.3-70B-Instruct-Turbo", help='Model ID to use.')
    together_chat_req.set_defaults(cmd=together_chat_cmd)

    return parser.parse_args()

def main() -> int:
    args = parse_args()
    return args.cmd(args)

if __name__ == "__main__":
    exit(main())
