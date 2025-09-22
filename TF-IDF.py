import math
import re
import random
import string

# Function to calculate term frequency (TF)
def compute_tf(word, document):
    words = document.split()
    total_words = len(words)
    if total_words == 0:  # Handle empty documents
        return 0
    word_count = words.count(word)
    return word_count / total_words

# Function to calculate inverse document frequency (IDF)
def compute_idf(word, documents):
    num_docs_with_word = sum(1 for doc in documents if word in doc.split())
    if num_docs_with_word == 0:
        return 0  # Avoid division by zero
    return math.log(len(documents) / num_docs_with_word)

# Function to calculate TF-IDF
def compute_tfidf(word, document, documents):
    tf = compute_tf(word, document)
    idf = compute_idf(word, documents)
    return tf * idf

# Read the input file
file_path = "pg1727.txt"  # Replace with your file name
with open(file_path, "r", encoding="utf-8") as file:
    text = file.read()

# Regex to match chapter headings (\n\n: followed by two newline characters)
chapter_pattern = r'(BOOK\s+(I|II|III|IV|V|VI|VII|VIII|IX|X|XI|XII|XIII|XIV|XV|XVI|XVII|XVIII|XIX|XX|XXI|XXII|XXIII|XXIV))\n\n'

# Split the text into chapters based on the pattern
documents = re.split(chapter_pattern, text)

# Combine chapter headings with their respective content
chapters = [
    documents[i] + "\n\n" + documents[i + 2]
    for i in range(0, len(documents) - 2, 3)
]

# Combine all chapters and extract a unique list of words
all_text = " ".join(chapters).lower()
all_words = list(set(all_text.split()))

# Remove punctuation from all_words
all_words = [word.translate(str.maketrans("", "", string.punctuation)) for word in all_words if word]

# Handle case where there are fewer than 10 words
num_words_to_sample = min(10, len(all_words))  # Dynamically adjust the number of words sampled
if num_words_to_sample == 0:
    print("No words available to sample after cleaning. Please check the input file.")
else:
    random_words = random.sample(all_words, num_words_to_sample)

    # Compute TF-IDF for each word in each chapter
    results = []
    for word in random_words:
        tfidf_scores = [compute_tfidf(word, chapter, chapters) for chapter in chapters]
        results.append((word, tfidf_scores))

    # Display the results
    print(f"Random Words: {random_words}\n")
    for word, scores in results:
        print(f"Word: '{word}'")
        for i, score in enumerate(scores):
            print(f"  TF-IDF in Chapter {i + 1}: {score:.5f}")
        print()
