# --- Fill in the Middle ---
# This script holds the function to apply the Fill in The Middle (FIM) strategy;
# 
# I took reference from the 2022 paper from OpenAI: Efficient Training of Language Models to Fill in the Middle
# Reference: https://arxiv.org/pdf/2207.14255

import random
from typing import Literal

SPECIAL_CHARACTERS = {
    "prefix": "<|fim_prefix|>",
    "middle": "<|fim_middle|>",
    "suffix": "<|fim_suffix|>",
    "end": "<|endoftext|>",
}

def apply_line_level_fim(document: str, 
                        rate: float = 0.9, 
                        strategy: Literal["psm", "spm"] = "psm") -> str:
    """Implements Fill in the Middle (FIM) strategy on a LINE level.
    
    This is a little twist on the original implementation since most of the data
    for the formalizer in my implementation will be code, line works better since we 
    preserve the structure of the input itself

    Args:
        document (str): Source text
        rate (float): Probability rate of applying FIM
        strategy (str): The FIM strategy that will be applied
    
    Returns:
        str: Reorganize source text based on the defined strategy
        
    """

    # apply Probability Check
    if random.random() > rate:
        return document + SPECIAL_CHARACTERS["end"]

    # split into Lines (Critical for Code)
    lines = document.splitlines(keepends=True)
    num_lines = len(lines)
    
    # safety check: Need enough lines to form P, M, S
    if num_lines < 3:
        return document + SPECIAL_CHARACTERS["end"]
    
    # randomly Select the "Middle" Span
    # ensure we leave at least 1 line for Prefix and 1 line for Suffix
    start_idx = random.randint(1, num_lines - 2)
    
    # Calculate how much space is left for the gap
    max_gap_size = num_lines - start_idx - 1 
    gap_length = random.randint(1, max_gap_size)
    end_idx = start_idx + gap_length
    
    # Create the Chunks slicing the list, not the string
    prefix = "".join(lines[:start_idx])
    middle = "".join(lines[start_idx:end_idx])
    suffix = "".join(lines[end_idx:])

    if strategy == "psm":
        # Standard FIM: Prefix -> Suffix -> Middle
        return (
            SPECIAL_CHARACTERS["prefix"] + prefix +
            SPECIAL_CHARACTERS["suffix"] + suffix +
            SPECIAL_CHARACTERS["middle"] + middle +
            SPECIAL_CHARACTERS["end"]
        )
    
    elif strategy == "spm":
        # Variant: Suffix -> Prefix -> Middle
        return (
            SPECIAL_CHARACTERS["suffix"] + suffix +
            SPECIAL_CHARACTERS["prefix"] + prefix +
            SPECIAL_CHARACTERS["middle"] + middle +
            SPECIAL_CHARACTERS["end"]
        )
    
    return document + SPECIAL_CHARACTERS["end"]
    

if __name__ == "__main__":
    # testing
    code_doc = """def add(a, b):
        result = a + b
        return result

    def multiply(a, b):
        return a * b
    """
    print("--- ORIGINAL DOC ---")
    print(code_doc)
    print("--- FIM DOC ---")
    print(line_level_fim(code_doc))
    