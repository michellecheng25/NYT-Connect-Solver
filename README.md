### NYT Connections

## Project Statemetn and Project Goal
NYT Connections gives 16 words that must be split into four groups of four. Some words look like they fit more than one group. Our main research question is: How do training, search methods, and game feedback affect the accuracy of an AI solving the NYT Connections puzzles? We test these hypotheses to find out what makes a solver effective:
1. A model trained on past puzzles scores candidate groups better than plain word similarity, with the biggest gain on groups where the words aren’t close in meaning, such as wordplay, spelling, and trivia or cultural-knowledge groups. We compare several pretrained embeddings.
2. Splitting all 16 words into four groups at once (every word is used exactly once) finds more correct groups than picking one group at a time.
3. In simulated games, a solver that keeps a probability for each possible full solution and updates it after the game’s feedback (correct, one away, wrong) solves more puzzles than one that ignores feedback. Guessing the most confident group first also wastes fewer of the 4 allowed mistakes than other guessing orders.

## Set Up
[TODO]


# Demo
