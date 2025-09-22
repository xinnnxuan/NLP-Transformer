"""
TF-IDF (Term Frequency-Inverse Document Frequency) Calculator

This module computes TF-IDF scores for randomly selected words from chapters of a public domain book.
Each chapter is treated as a document in the corpus. This demonstrates understanding of word
importance in the context of document frequency.

Key Skills: TF-IDF, text preprocessing, Google Colab, Python, statistical NLP
"""

import math
import re
import random
import string

# Constants
FILE_PATH = "pg1727.txt"  # Replace with your file name
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
    if total_words == 0:  # Handle empty documents
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
        return 0  # Avoid division by zero
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


def main():
    """
    Main function to process text and compute TF-IDF scores.
    """
    # Read the input file
    with open(FILE_PATH, "r", encoding="utf-8") as file:
        text = file.read()

    # Split the text into chapters based on the pattern
    documents = re.split(CHAPTER_PATTERN, text)

    # Combine chapter headings with their respective content
    chapters = [
        documents[i] + "\n\n" + documents[i + 2]
        for i in range(0, len(documents) - 2, 3)
    ]

    # Combine all chapters and extract a unique list of words
    all_text = " ".join(chapters).lower()
    all_words = list(set(all_text.split()))

    # Remove punctuation from all_words
    all_words = [word.translate(str.maketrans("", "", string.punctuation)) 
                for word in all_words if word]

    # Handle case where there are fewer than 10 words
    num_words_to_sample = min(10, len(all_words))
    if num_words_to_sample == 0:
        print("No words available to sample after cleaning. "
              "Please check the input file.")
        return

    random_words = random.sample(all_words, num_words_to_sample)

    # Compute TF-IDF for each word in each chapter
    results = []
    for word in random_words:
        tfidf_scores = [compute_tfidf(word, chapter, chapters) 
                       for chapter in chapters]
        results.append((word, tfidf_scores))

    # Display the results
    print(f"Random Words: {random_words}\n")
    for word, scores in results:
        print(f"Word: '{word}'")
        for i, score in enumerate(scores):
            print(f"  TF-IDF in Chapter {i + 1}: {score:.5f}")
        print()


if __name__ == "__main__":
    main()
