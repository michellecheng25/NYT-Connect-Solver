### NYT Connections

## Project Statemetn and Project Goal

NYT Connections gives 16 words that must be split into four groups of four. Some words look like they fit more than one group. Our main research question is: How do training, search methods, and game feedback affect the accuracy of an AI solving the NYT Connections puzzles? We test these hypotheses to find out what makes a solver effective:

1. A model trained on past puzzles scores candidate groups better than plain word similarity, with the biggest gain on groups where the words aren’t close in meaning, such as wordplay, spelling, and trivia or cultural-knowledge groups. We compare several pretrained embeddings.
2. Splitting all 16 words into four groups at once (every word is used exactly once) finds more correct groups than picking one group at a time.
3. In simulated games, a solver that keeps a probability for each possible full solution and updates it after the game’s feedback (correct, one away, wrong) solves more puzzles than one that ignores feedback. Guessing the most confident group first also wastes fewer of the 4 allowed mistakes than other guessing orders.

## Set Up

## Set Up

Run these commands from the project folder, where `main.py` is located.

Create a virtual environment:

```bash
python3 -m venv .venv
```

Activate it on Mac:

```bash
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install the required packages:

```bash
python -m pip install -r requirements.txt
```

## Demo

To run the terminal version:

```bash
python main.py
```

This prints the first puzzle, the solver's predictions, the correct answers, and results across 10 puzzles.

To run the puzzle interface:

```bash
python -m streamlit run src/frontend/app.py
```

Open the Local URL shown in the terminal, usually http://localhost:8501.

# Demo
