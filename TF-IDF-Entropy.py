"""
TF-IDF and Entropy Calculator with N-gram Analysis

This module computes TF-IDF scores and entropy for randomly selected unigrams, bigrams,
and trigrams from chapters of a public domain book. Each chapter is treated as a document
in the corpus. This demonstrates understanding of word importance and n-gram analysis.

Key Skills: TF-IDF, Entropy, N-grams, text preprocessing, Python, statistical NLP
"""

import math
import re
import random
import string

# Constants
FILE_PATH = "pg1727.txt"
CHAPTER_PATTERN = r'(BOOK\s+(I|II|III|IV|V|VI|VII|VIII|IX|X|XI|XII|XIII|XIV|XV|XVI|XVII|XVIII|XIX|XX|XXI|XXII|XXIII|XXIV))\n\n'


def compute_tf(word, document):
    """
    Calculate term frequency (TF) for a word in a document.
    
    Args:
        word (str): The word to calculate TF for
        document (str): The document text
        
    Returns:
        float: Term frequency value
    """
    words = document.split()
    total_words = len(words)
    if total_words == 0:
        return 0
    word_count = words.count(word)
    return word_count / total_words


def compute_idf(word, documents):
    """
    Calculate inverse document frequency (IDF) for a word across documents.
    
    Args:
        word (str): The word to calculate IDF for
        documents (list): List of document texts
        
    Returns:
        float: Inverse document frequency value
    """
    num_docs_with_word = sum(1 for doc in documents if word in doc.split())
    if num_docs_with_word == 0:
        return 0
    return math.log(len(documents) / num_docs_with_word)


def compute_tfidf(word, document, documents):
    """
    Calculate TF-IDF score for a word in a document.
    
    Args:
        word (str): The word to calculate TF-IDF for
        document (str): The document text
        documents (list): List of all documents in the corpus
        
    Returns:
        float: TF-IDF score
    """
    tf = compute_tf(word, document)
    idf = compute_idf(word, documents)
    return tf * idf


def compute_entropy(word, document, documents):
    """
    Calculate entropy for a word in a document.
    
    Args:
        word (str): The word to calculate entropy for
        document (str): The document text
        documents (list): List of all documents in the corpus
        
    Returns:
        float: Entropy value
    """
    tf_idf = compute_tfidf(word, document, documents)
    if tf_idf > 0:
        return -tf_idf * math.log(tf_idf)
    return 0


def generate_ngrams(text, n):
    """
    Generate n-grams from text.
    
    Args:
        text (str): Input text
        n (int): Size of n-grams
        
    Returns:
        list: List of n-grams
    """
    words = text.split()
    return [" ".join(words[i:i+n]) for i in range(len(words)-n+1)]


def main():
    """
    Main function to process text and compute TF-IDF and entropy scores.
    """
    # Read the input file
    with open(FILE_PATH, "r", encoding="utf-8") as file:
        text = file.read()

    # Split text into chapters using a pattern
    documents = re.split(CHAPTER_PATTERN, text)

    # Combine chapter headings with their content
    chapters = [documents[i] + "\n\n" + documents[i + 2] 
               for i in range(0, len(documents) - 2, 3)]

    # Prepare a list of words, bigrams, and trigrams
    all_text = " ".join(chapters).lower()
    all_words = list(set(all_text.split()))
    all_words = [word.translate(str.maketrans("", "", string.punctuation)) 
                for word in all_words if word]

    bigrams = list(set(generate_ngrams(all_text, 2)))
    trigrams = list(set(generate_ngrams(all_text, 3)))

    # Randomly select 10 unigrams, bigrams, and trigrams
    num_words_to_sample = min(10, len(all_words))
    num_bigrams_to_sample = min(10, len(bigrams))
    num_trigrams_to_sample = min(10, len(trigrams))

    random_unigrams = random.sample(all_words, num_words_to_sample)
    random_bigrams = random.sample(bigrams, num_bigrams_to_sample)
    random_trigrams = random.sample(trigrams, num_trigrams_to_sample)

    # Compute TF-IDF and entropy for each word in each chapter
    results = {"unigram": [], "bigram": [], "trigram": []}

    for word in random_unigrams:
        chapter_scores = [(chapter, compute_tfidf(word, chapter, chapters), 
                         compute_entropy(word, chapter, chapters)) 
                        for chapter in chapters]
        results["unigram"].append((word, chapter_scores))

    for word in random_bigrams:
        chapter_scores = [(chapter, compute_tfidf(word, chapter, chapters), 
                         compute_entropy(word, chapter, chapters)) 
                        for chapter in chapters]
        results["bigram"].append((word, chapter_scores))

    for word in random_trigrams:
        chapter_scores = [(chapter, compute_tfidf(word, chapter, chapters), 
                         compute_entropy(word, chapter, chapters)) 
                        for chapter in chapters]
        results["trigram"].append((word, chapter_scores))

    # Display the results
    print("TF-IDF and Entropy for each chapter:")
    for ngram_type, words in results.items():
        print(f"\nType: {ngram_type.capitalize()}")
        for word, chapters_data in words:
            print(f"\nWord: {word}")
            for idx, (chapter, tfidf, entropy) in enumerate(chapters_data):
                print(f"Chapter {idx+1} - TF-IDF: {tfidf:.4f}, "
                      f"Entropy: {entropy:.4f}")


if __name__ == "__main__":
    main()