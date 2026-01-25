# --- Fill in the Middle ---
# This script holds the function to apply the Fill in The Middle (FIM) strategy;
# 
# I took reference from the 2022 paper from OpenAI: Efficient Training of Language Models to Fill in the Middle
# Reference: https://arxiv.org/pdf/2207.14255

from pathlib import Path

SPECIAL_CHARACTERS = {
    "prefix": "<|fim_prefix|>",
    "middle": "<|fim_middle|>",
    "suffix": "<|fim_suffix|>",
    "end": "<|endoftext|>",
}

def character_level_psm_fim(document: str) -> str:
    """Implements the Prefix, Suffix, Middle strategy"""

    # count the amount of characters
    document_len = len(document)
    sep_point = document_len//3 # using the third strategy implemented in the reference paper

    # separate the document in thirds
    prefix = document[:sep_point] 
    middle = document[sep_point:2*sep_point]
    suffix = document[2*sep_point:]

    # apply the transformation: document -> (prefix, middle, suffix) -> (prefix, suffix, middle)
    psm_doc = SPECIAL_CHARACTERS["prefix"] + prefix + \
              SPECIAL_CHARACTERS["suffix"] + suffix + \
              SPECIAL_CHARACTERS["middle"] + middle + \
              SPECIAL_CHARACTERS["end"]
    
    return psm_doc

if __name__ == "__main__":
    # testing
    test_doc = "Sun is up. We go out. Day is on."
    print(test_doc)
    print(character_level_psm_fim(test_doc))
    