import spacy
import nltk
from nltk.tree import Tree

# Load the small English NLP model
nlp = spacy.load("en_core_web_sm")

def token_to_tree(token):
    """
    Recursively convert a spaCy token to an NLTK Tree format.
    """
    if not list(token.children):
        return token.text
    return Tree(token.text, [token_to_tree(child) for child in token.children])

def generate_pretty_dependency_tree(sentence):
    """
    Generate and display a dependency tree in a human-readable format.
    """
    doc = nlp(sentence)
    root = [token for token in doc if token.head == token][0]  # Find the root token
    tree = token_to_tree(root)

    # Display tree structure
    print(f"\nDependency Tree for: '{sentence}'")
    tree.pretty_print()

# Sample sentences for testing
sentences = [
    "The cat sat on the mat.",
    "She enjoys reading books about history.",
    "AI is transforming the way we work.",
    "He quickly finished his assignment.",
    "The concert was an unforgettable experience.",
    "She found the lost key under the couch.",
    "Technology is advancing at an incredible pace.",
    "They will travel to Japan next summer.",
    "The teacher explained the complex concept clearly.",
    "Creativity and hard work lead to success."
]

# Generate and display dependency trees for each sentence
for sentence in sentences:
    generate_pretty_dependency_tree(sentence)
