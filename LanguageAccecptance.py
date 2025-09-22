"""
Language Acceptance Task

Created a classifier to identify whether strings belong to a formal language of 
the form aⁿbⁿaⁿ. Generated and tested both valid and near-miss invalid strings. 
Applied formal language theory and string pattern recognition.

Key Skills: Regular language recognition, string classification, generative testing,
            Python logic functions
"""
import random

def generate_valid_string(min_n=1, max_n=5):
    """
    Generate a valid string in language L of format a^n b^n a^n
    """
    n = random.randint(min_n, max_n)
    return 'a' * n + 'b' * n + 'a' * n

def generate_invalid_string(min_n=1, max_n=5):
    """
    Generate an invalid string that's almost in language L but has 1-2 errors
    """
    n = random.randint(min_n, max_n)
    error_type = random.randint(1, 4)

    if error_type == 1:
        # Wrong number of a's in first section
        first_a = n + random.choice([-1, 1])
        return 'a' * first_a + 'b' * n + 'a' * n
    elif error_type == 2:
        # Wrong number of b's
        b_count = n + random.choice([-1, 1])
        return 'a' * n + 'b' * b_count + 'a' * n
    elif error_type == 3:
        # Wrong number of a's in last section
        last_a = n + random.choice([-1, 1])
        return 'a' * n + 'b' * n + 'a' * last_a
    else:
        # Insert a wrong character
        valid = generate_valid_string(n, n)
        pos = random.randint(0, len(valid) - 1)
        wrong_char = random.choice(['c', 'd'])
        return valid[:pos] + wrong_char + valid[pos+1:]

def is_in_language(s):
    """
    Check if string s is in language L (a^n b^n a^n)
    Returns: (bool, str) - (is_valid, explanation)
    """
    # Check if string only contains a's and b's
    if not all(c in 'ab' for c in s):
        return False, "String contains characters other than 'a' and 'b'"

    # Find the first 'b' to separate first a-section
    first_b_idx = s.find('b')
    if first_b_idx == -1:
        return False, "No 'b' found in string"

    # Find the last 'b' to separate last a-section
    last_b_idx = s.rfind('b')

    # Extract sections
    first_a = s[:first_b_idx]
    middle_b = s[first_b_idx:last_b_idx + 1]
    last_a = s[last_b_idx + 1:]

    # Check if middle section contains only b's
    if not all(c == 'b' for c in middle_b):
        return False, "Middle section contains non-'b' characters"

    # Check if first and last sections contain only a's
    if not all(c == 'a' for c in first_a) or not all(c == 'a' for c in last_a):
        return False, "First or last section contains non-'a' characters"

    # Check if all sections have equal length
    if len(first_a) == len(middle_b) == len(last_a):
        return True, f"Valid string with n={len(first_a)}"
    else:
        return False, f"Unequal sections: |a1|={len(first_a)}, |b|={len(middle_b)}, |a2|={len(last_a)}"

def main():
    print("Testing Language L = {a^n b^n a^n | n ≥ 1}")
    print("\nGenerating and classifying 20 strings:")
    print("-" * 60)

    for i in range(20):
        # Generate either valid or invalid string
        if random.random() < 0.5:
            s = generate_valid_string()
            expected = "Valid"
        else:
            s = generate_invalid_string()
            expected = "Invalid"

        # Classify the string
        is_valid, explanation = is_in_language(s)
        result = "Valid" if is_valid else "Invalid"

        # Print results
        print(f"{i+1}. String: {s}")
        print(f"   Expected: {expected}")
        print(f"   Result: {result}")
        print(f"   Explanation: {explanation}")
        print("-" * 60)

if __name__ == "__main__":
    main()
