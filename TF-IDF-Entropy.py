import math
import re
import random
import string

# Function to calculate term frequency (TF)
def compute_tf(word, document):
    words = document.split()
    total_words = len(words)
    if total_words == 0:
        return 0
    word_count = words.count(word)
    return word_count / total_words

# Function to calculate inverse document frequency (IDF)
def compute_idf(word, documents):
    num_docs_with_word = sum(1 for doc in documents if word in doc.split())
    if num_docs_with_word == 0:
        return 0
    return math.log(len(documents) / num_docs_with_word)

# Function to calculate TF-IDF
def compute_tfidf(word, document, documents):
    tf = compute_tf(word, document)
    idf = compute_idf(word, documents)
    return tf * idf

# Function to calculate entropy
def compute_entropy(word, document, documents):
    tf_idf = compute_tfidf(word, document, documents)
    if tf_idf > 0:
        return -tf_idf * math.log(tf_idf)
    return 0

# Function to generate n-grams
def generate_ngrams(text, n):
    words = text.split()
    return [" ".join(words[i:i+n]) for i in range(len(words)-n+1)]

# Read the input file
file_path = "pg1727.txt"
with open(file_path, "r", encoding="utf-8") as file:
    text = file.read()

# Split text into chapters using a pattern
chapter_pattern = r'(BOOK\s+(I|II|III|IV|V|VI|VII|VIII|IX|X|XI|XII|XIII|XIV|XV|XVI|XVII|XVIII|XIX|XX|XXI|XXII|XXIII|XXIV))\n\n'
documents = re.split(chapter_pattern, text)

# Combine chapter headings with their content
chapters = [documents[i] + "\n\n" + documents[i + 2] for i in range(0, len(documents) - 2, 3)]

# Prepare a list of words, bigrams, and trigrams
all_text = " ".join(chapters).lower()
all_words = list(set(all_text.split()))
all_words = [word.translate(str.maketrans("", "", string.punctuation)) for word in all_words if word]

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
    chapter_scores = [(chapter, compute_tfidf(word, chapter, chapters), compute_entropy(word, chapter, chapters)) for chapter in chapters]
    results["unigram"].append((word, chapter_scores))

for word in random_bigrams:
    chapter_scores = [(chapter, compute_tfidf(word, chapter, chapters), compute_entropy(word, chapter, chapters)) for chapter in chapters]
    results["bigram"].append((word, chapter_scores))

for word in random_trigrams:
    chapter_scores = [(chapter, compute_tfidf(word, chapter, chapters), compute_entropy(word, chapter, chapters)) for chapter in chapters]
    results["trigram"].append((word, chapter_scores))

# Display the results
print("TF-IDF and Entropy for each chapter:")
for ngram_type, words in results.items():
    print(f"\nType: {ngram_type.capitalize()}")
    for word, chapters_data in words:
        print(f"\nWord: {word}")
        for idx, (chapter, tfidf, entropy) in enumerate(chapters_data):
            print(f"Chapter {idx+1} - TF-IDF: {tfidf:.4f}, Entropy: {entropy:.4f}")