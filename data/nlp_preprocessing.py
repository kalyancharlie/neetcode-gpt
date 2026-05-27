import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import List

class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        # 1. Build vocabulary: collect all unique words, sort them, assign integer IDs starting at 1
        # 2. Encode each sentence by replacing words with their IDs
        # 3. Combine positive + negative into one list of tensors
        # 4. Pad shorter sequences with 0s using nn.utils.rnn.pad_sequence(tensors, batch_first=True)
        
        # 1. Build vocabulary: collect all unique words, sort them
        vocab_set = set()
        
        # Split words from sentence
        for sentence in positive + negative:
            # .split() without arguments handles multiple spaces cleanly
            for word in sentence.split():
                vocab_set.add(word)
                
        sorted_vocab = sorted(vocab_set)
        
        # Assign integer IDs starting at 1 (0 is reserved for padding)
        # Using dictionary comprehension and enumerate makes this very clean
        word_to_ind = {word: i + 1 for i, word in enumerate(sorted_vocab)}

        # Encode each sentence and combine into one list of tensors
        tensors_list = []
        for sentence in positive + negative:
            encoded_sentence = [word_to_ind[word] for word in sentence.split()]
            
            # Convert the standard Python list of numbers into a PyTorch tensor
            tensor = torch.tensor(encoded_sentence, dtype=torch.float)
            tensors_list.append(tensor)

        # 4. Pad shorter sequences with 0s so they all share the exact same length
        # batch_first=True outputs (batch_size, seq_length) e.g., 3 sentences of 5 words = [3, 5]. False flips it to [5, 3].
        padded_dataset = nn.utils.rnn.pad_sequence(tensors_list, batch_first=True)

        return padded_dataset
