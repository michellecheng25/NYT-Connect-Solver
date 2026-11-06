import json
import re
from pathlib import Path

from src.llm import MODEL, llm_grouping
from src.data import date_split, load_puzzles
from src.evaluate import count_correct_groups

path = Path("baseline_outputs/llm_baseline.json")


def build_entry(puzzle, positions, tries, model):
    """Describe one puzzle's result using words instead of only board positions."""
    words = puzzle["words"]
    predicted = [frozenset(g) for g in positions] if positions else []
    group_names = dict(zip(puzzle["groups"], puzzle["group_names"]))
    groups_correct = count_correct_groups(predicted, puzzle["groups"])

    return {
        "id": puzzle["id"],
        "date": puzzle["date"],
        "model": model,
        "tries": tries,
        "valid_answer": bool(predicted),
        "groups_correct": groups_correct,
        "solved": groups_correct == 4,
        "words": words,
        "predicted_groups": [
            {
                "words": [words[i] for i in sorted(group)],
                "positions": sorted(group),
                "correct": group in group_names,
                "matched_group": group_names.get(group),
            }
            for group in predicted
        ],
        "correct_groups": [
            {
                "name": name,
                "level": level,
                "words": [words[i] for i in sorted(group)],
                "guessed": group in predicted,
            }
            for level, name, group in sorted(
                zip(puzzle["levels"], puzzle["group_names"], puzzle["groups"]),
                key=lambda item: item[0],
            )
        ],
    }


def build_summary(entries):
    found = [entry["groups_correct"] for entry in entries]
    count = len(entries)
    return {
        "model": MODEL,
        "puzzles_evaluated": count,
        "average_groups_correct": round(sum(found) / count, 3) if count else None,
        "puzzles_solved": sum(entry["solved"] for entry in entries),
        "invalid_answers_after_retries": sum(not entry["valid_answer"] for entry in entries),
        "puzzles_by_groups_correct": {str(n): found.count(n) for n in range(5)},
    }


def write_results(test, saved):
    entries = [saved[str(p["id"])] for p in test if str(p["id"]) in saved]
    results = {
        "summary": build_summary(entries),
        "puzzles": {str(entry["id"]): entry for entry in entries},
    }
    text = json.dumps(results, indent=2, ensure_ascii=False)
    # Keep each list of words on one line so the file is easier to read.
    text = re.sub(
        r"\[\s*([^\[\]{}]*?)\s*\]",
        lambda m: "[" + re.sub(r"\s*\n\s*", " ", m.group(1)) + "]",
        text,
    )
    path.write_text(text + "\n", encoding="utf-8")
    return results["summary"]


path.parent.mkdir(exist_ok=True)
data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
saved = data.get("puzzles", {})

_, _, test = date_split(load_puzzles())

# Call the API only for puzzles without a saved answer.
for n, puzzle in enumerate(test, start=1):
    key = str(puzzle["id"])
    if key not in saved:
        groups, tries = llm_grouping(puzzle["words"])
        positions = [sorted(g) for g in groups] if groups else None
        saved[key] = build_entry(puzzle, positions, tries, MODEL)
        write_results(test, saved)
    print(f"{n}/{len(test)} done")

summary = write_results(test, saved)

print(f"Avg. groups: {summary['average_groups_correct']:.3f} / 4")
print(f"Fully solved: {summary['puzzles_solved']}/{summary['puzzles_evaluated']}")
print(f"Invalid after retries: {summary['invalid_answers_after_retries']}/{summary['puzzles_evaluated']}")
