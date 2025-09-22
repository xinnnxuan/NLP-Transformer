import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

def get_contextual_vectors(sentence):
    """
    Generate contextual vectors for each word in a sentence using BERT
    """
    # Load pre-trained model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    model = AutoModel.from_pretrained('bert-base-uncased')

    # Tokenize input
    tokens = tokenizer(sentence, return_tensors='pt', add_special_tokens=True)

    # Get model outputs
    with torch.no_grad():
        outputs = model(**tokens)

    # Get the hidden states
    hidden_states = outputs.last_hidden_state.squeeze(0)

    # Get the original words and their vectors
    words = tokenizer.convert_ids_to_tokens(tokens['input_ids'][0])
    vectors = hidden_states.numpy()

    # Remove special tokens ([CLS] and [SEP])
    words = words[1:-1]
    vectors = vectors[1:-1]

    return words, vectors

def print_similarity_matrix(words, vectors):
    """
    Compute and print the similarity matrix between words
    """
    # Compute cosine similarity
    similarity_matrix = cosine_similarity(vectors)

    # Create a DataFrame for better visualization
    df = pd.DataFrame(similarity_matrix, index=words, columns=words)

    print("\nSimilarity Matrix:")
    print(df.round(3))
    print("\n" + "="*50 + "\n")

def compare_word_in_contexts(sentence1, sentence2, target_word):
    """
    Compare the contextual vectors of the same word in different sentences
    """
    # Get vectors for both sentences
    words1, vectors1 = get_contextual_vectors(sentence1)
    words2, vectors2 = get_contextual_vectors(sentence2)

    # Convert tokens to lowercase for comparison
    words1 = [w.lower() for w in words1]
    words2 = [w.lower() for w in words2]

    # Find the target word in both sentences
    try:
        idx1 = words1.index(target_word.lower())
        idx2 = words2.index(target_word.lower())

        # Get the vectors for the target word in both contexts
        vector1 = vectors1[idx1]
        vector2 = vectors2[idx2]

        # Compute cosine similarity between the two vectors
        similarity = cosine_similarity([vector1], [vector2])[0][0]

        print(f"\nComparing '{target_word}' in different contexts:")
        print(f"Sentence 1: {sentence1}")
        print(f"Sentence 2: {sentence2}")
        print(f"Similarity score: {similarity:.3f}")

    except ValueError:
        print(f"Could not find '{target_word}' in both sentences")

def main():
    # Example sentences
    sentences = [
        "The cat sits on the mat",
        "I love programming and coding",
        "The weather is beautiful today",
        "She reads books in the library",
        "The mountain peaks are covered in snow",
        "Children play in the park",
        "The chef cooks delicious meals",
        "Birds fly high in the sky",
        "Students study hard for exams",
        "The ocean waves crash on the shore"
    ]

    print("Analyzing contextual word similarities for 10 sentences:\n")

    for i, sentence in enumerate(sentences, 1):
        print(f"Sentence {i}: {sentence}")
        words, vectors = get_contextual_vectors(sentence)
        print_similarity_matrix(words, vectors)

    # Add comparison of word in different contexts
    print("\nComparing same word in different contexts:")
    sentence1 = "Please lead the way to the exit."
    sentence2 = "The pipes were made of lead."
    compare_word_in_contexts(sentence1, sentence2, "lead")

    # Another example
    sentence3 = "The bank of the river was muddy."
    sentence4 = "I need to bank this check today."
    compare_word_in_contexts(sentence3, sentence4, "bank")

if __name__ == "__main__":
    main()
