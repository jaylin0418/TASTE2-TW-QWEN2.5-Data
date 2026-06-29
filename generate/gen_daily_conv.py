#!/usr/bin/env python3
"""
Type 2: Daily Conversation generation.
Same pipeline as gen_user_agent.py but with daily_conv config and prompts.

Usage:
    python generate/gen_daily_conv.py --config conf/daily_conv.yaml \
        --topic 美食 --num-scenarios 200 --output output/daily_conv/
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Daily conv uses identical pipeline logic as user_agent;
# only the config (prompts, type) differs.
from generate.gen_user_agent import main

if __name__ == "__main__":
    main()
