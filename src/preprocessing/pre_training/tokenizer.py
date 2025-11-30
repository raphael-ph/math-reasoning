# --- Tokenizer class ---
# This implementation has many inspirations and it took me sometime to build this rather simple
# Acknowledgements to:
# 
# 1. Karpathy (as always): https://www.youtube.com/watch?v=zduSFxRajkE
# 2. This blog: https://sebastianraschka.com/blog/2025/bpe-from-scratch.html
# 3. GPT-2 paper: https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf

from typing import Dict, Tuple, List

# internal imports
from ...utils.logger import get_logger

# set up logging
_logger = get_logger(__name__, level="DEBUG")
class Tokenizer:
    def __init__(self, ):
        pass

    def _merge(self, ids: List, bigram: Tuple, idx: int) -> List[int]:
        """Helper function to substitute pairs
        
        This is equivalent to Karpathy's `merge()` function. It finds the 
        byte pairs in ids and changes it to the byte-pair encoded token it to the tokenizer's vocab.

        Args:
            ids (List): list of tokens from the corpus
            bigram (Tuple): These are the byte pairs
            idx (int): The correspondent idx for the pair
        
        Returns:
            List[int]: the new ids, byte-encoded
        """
        bpe = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == bigram[0] and ids[i+1] == bigram[1]:
                bpe.append(idx)
                i += 2
            else:
                bpe.append(ids[i])
                i += 1
        return bpe

    def _get_stats(self, ids: List) -> Dict[Tuple, int]:
        """Helper function to get information of bigrams:
        
        Args:
            ids (list): list of the encoded tokens
        
        Returns:
            Dict[Tuple, int]: dictionary conatining the bigram and it's frequency
            of appearance on the corpus
        """
        counts = {} # dict to handle the counting of pairs
        for bigram in zip(ids, ids[1:]):  # getting together n and n+1 bytes
            # Get value if key does not exist, with a specified default value
            counts[bigram] = counts.get(bigram, 0) + 1
        return counts
