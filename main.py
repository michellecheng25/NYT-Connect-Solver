from src.data import load_puzzles;
from src.embeddings import WordVectors;
from src.scorer import plain_similarity;

import numpy as np



def main():
    wv = WordVectors.load_glove()
    words = ['SNOW', 'LEVEL', 'SHIFT', 'KAYAK', 'HEAT', 'TAB', 'BUCKS', 'RETURN', 'JAZZ', 'HAIL', 'OPTION', 'RAIN', 'SLEET', 'RACECAR', 'MOM', 'NETS']
    scores = plain_similarity(words, wv)
    print(scores[:10])
    print(len(scores))


main()