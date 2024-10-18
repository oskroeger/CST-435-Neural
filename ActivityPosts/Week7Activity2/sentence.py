import itertools
from tqdm import tqdm

# Original jumbled sentence with known commas and blanks
words = [
    "owed", "(4-letter blank)", "she", "at", "a", "debt,", "Alex", "broke", "his", "ball,", 
    "paying", "however", "(6-letter blank)", "with", "the", "boy", "a", "man", "him", "the", "wasn't."
]

# Possible blanks
candidates_4_letter = ["debt", "loan", "time"]
candidates_6_letter = ["credit", "future", "person"]

# Possible red herring (a money-related word)
money_related_words = ["owed", "paying", "credit"]

# Lock "debt," and "ball," in their positions
locked_words = ["debt,", "ball,"]

# Helper function to validate the sentence based on basic rules
def validate_sentence(sentence):
    # Basic sentence validation rules (subject-verb-object structure etc.)
    return "Alex" in sentence and "the boy" in sentence and "wasn't." in sentence

# Function to generate permutations of smaller chunks of the sentence
def targeted_permutation(words, candidates_4_letter, candidates_6_letter):
    # Identify the index positions of the blanks
    index_4 = words.index("(4-letter blank)")
    index_6 = words.index("(6-letter blank)")
    
    # Split the sentence into smaller sections for permutation
    # Focus on a small part around the blanks
    context_4 = words[max(0, index_4 - 2):min(len(words), index_4 + 3)]
    context_6 = words[max(0, index_6 - 2):min(len(words), index_6 + 3)]
    
    # Generate permutations for just these small contexts
    permutations_4 = list(itertools.permutations(context_4))
    permutations_6 = list(itertools.permutations(context_6))
    
    with tqdm(total=len(permutations_4) * len(permutations_6), desc="Processing Local Permutations") as pbar:
        for perm_4 in permutations_4:
            for perm_6 in permutations_6:
                perm_4 = list(perm_4)
                perm_6 = list(perm_6)
                
                # Properly insert the blanks with candidate words
                perm_4[perm_4.index("(4-letter blank)")] = candidates_4_letter[0]  # Start with first candidate for now
                perm_6[perm_6.index("(6-letter blank)")] = candidates_6_letter[0]  # Start with first candidate for now
                
                # Merge the smaller sections back into the full sentence
                # Ensure no part of the sentence repeats
                reconstructed_sentence = " ".join(words[:index_4-2] + perm_4 + words[index_4+3:index_6-2] + perm_6 + words[index_6+3:])
                
                # Clean up and check for redundancy
                reconstructed_sentence = " ".join(list(dict.fromkeys(reconstructed_sentence.split())))
                
                # Validate the sentence structure
                if validate_sentence(reconstructed_sentence):
                    return reconstructed_sentence
                
                pbar.update(1)
    return None

# Call the function to reconstruct the sentence using the targeted permutation approach
final_sentence = targeted_permutation(words, candidates_4_letter, candidates_6_letter)
if final_sentence:
    print("Reconstructed sentence:", final_sentence)
else:
    print("No valid sentence found.")
