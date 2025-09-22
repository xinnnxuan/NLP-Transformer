"""
Make Parse Trees

Developed a Python program that parses sentences into dependency or constituency trees 
using NLP libraries like spacy, NLTK, or diaparser. Visualized tree structures for 10 
sample sentences to illustrate syntactic parsing.


Key Skills: Parsing, dependency trees, sentence structure analysis, spacy, NLTK, tree visualization
"""

try:
    import spacy
    from nltk.tree import Tree
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    print("Warning: spaCy and NLTK not available. Please install: pip install spacy nltk")
    print("Also download the English model: python -m spacy download en_core_web_sm")

# Load the small English NLP model
if SPACY_AVAILABLE:
    nlp = spacy.load("en_core_web_sm")
else:
    nlp = None

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
    if not SPACY_AVAILABLE or nlp is None:
        print(f"\nCannot generate tree for: '{sentence}' - spaCy not available")
        return

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

def main():
    """
    Main function to generate and display dependency trees.
    """
    if not SPACY_AVAILABLE:
        print("Cannot run demonstration without spaCy and NLTK.")
        return

    # Generate and display dependency trees for each sentence
    for sentence in sentences:
        generate_pretty_dependency_tree(sentence)


if __name__ == "__main__":
    main()
