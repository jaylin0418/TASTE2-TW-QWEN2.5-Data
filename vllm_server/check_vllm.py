#!/usr/bin/env python3
"""Check if vLLM server is running and responsive."""
import sys
import argparse
from openai import OpenAI

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", default=8000, type=int)
    args = parser.parse_args()

    url = f"http://{args.host}:{args.port}/v1"
    client = OpenAI(base_url=url, api_key="token")

    try:
        models = client.models.list()
        print(f"vLLM server OK at {url}")
        for m in models.data:
            print(f"  Model: {m.id}")

        resp = client.chat.completions.create(
            model=models.data[0].id,
            messages=[{"role": "user", "content": "你好，請用一句話介紹台北一〇一。"}],
            max_tokens=64,
        )
        print(f"Test response: {resp.choices[0].message.content}")
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
