import json
import sys
from collections import Counter

def count_words_and_sentences(sentences):
    sentences_flat = [word for sent in sentences for word in sent]
    total_word_count = len(sentences_flat)
    unique_word_count = len(set(sentences_flat))
    sentence_count = len(sentences)
    return total_word_count, unique_word_count, sentence_count

def main(dataset_path):
    # Load dataset
    with open(dataset_path, 'r') as file:
        sentences = json.load(file)

    # Count words and sentences
    total_word_count, unique_word_count, sentence_count = count_words_and_sentences(sentences)
    
    print(f"Total number of words: {total_word_count}")
    print(f"Number of unique words: {unique_word_count}")
    print(f"Number of sentences: {sentence_count}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python count_words.py <path_to_dataset>")
        sys.exit(1)

    dataset_path = sys.argv[1]
    main(dataset_path)
