import json
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

MODEL = "meta-llama/llama-3.1-8b-instruct"

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)

PROMPT = """You are solving a NYT Connections puzzle. Split these 16 words into
four groups of four, where each group shares a specific connection. Connections
can involve meaning, wordplay, spelling, or trivia. Every word must be used exactly once.

Words: {words}

Respond with only JSON in this format:
{{"groups": [["WORD", "WORD", "WORD", "WORD"], ["WORD", "WORD", "WORD", "WORD"],
["WORD", "WORD", "WORD", "WORD"], ["WORD", "WORD", "WORD", "WORD"]]}}"""


def parse_llm_groups(text, words):
    """Return four groups of positions, or None if the answer is invalid."""
    try:
        groups = json.loads(text)["groups"]
        positions = {w.upper(): i for i, w in enumerate(words)}
        result = [frozenset(positions[w.strip().upper()] for w in g) for g in groups]
    except (json.JSONDecodeError, KeyError, TypeError, AttributeError):
        return None

    valid = len(result) == 4 and all(len(g) == 4 for g in result)
    return result if valid and frozenset().union(*result) == frozenset(range(16)) else None


def llm_grouping(words, seed=42, max_tries=3):
    """Ask Llama for a solution. Returns (groups or None, number of tries)."""
    prompt = PROMPT.format(words=", ".join(words))

    for attempt in range(max_tries):
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            seed=seed + attempt,
            response_format={"type": "json_object"},
            extra_body={"provider": {"require_parameters": True}},
        )
        groups = parse_llm_groups(response.choices[0].message.content, words)
        if groups is not None:
            return groups, attempt + 1

    return None, max_tries