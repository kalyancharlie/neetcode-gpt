from typing import List
from collections import defaultdict

class Solution:
    def get_merges(self, corpus: str, num_merges: int) -> List[List[str]]:
        # 1. Split corpus into a list of individual characters
        tokens = list(corpus)
        merges = []

        # 2. For each merge step:
        for _ in range(num_merges):
            # If we have less than 2 tokens, no more adjacent pairs can exist
            if len(tokens) < 2:
                break
                
            # a. Count frequency of all adjacent token pairs
            pair_counts = defaultdict(int)
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                pair_counts[pair] += 1
                
            if not pair_counts:
                break

            # b. Find the most frequent pair (break ties lexicographically)
            best_pair = None
            max_freq = -1

            # Loop through every pair and its frequency in our dictionary
            for pair, freq in pair_counts.items():
                
                # 1. If we find a new highest frequency, update our best pair
                if freq > max_freq:
                    max_freq = freq
                    best_pair = pair
                    
                # 2. If the frequencies are a tie, break the tie alphabetically
                elif freq == max_freq:
                    # We check if the current pair comes before our best_pair alphabetically
                    if pair < best_pair:
                        best_pair = pair

            # c. Merge all non-overlapping occurrences left to right
            new_tokens = []
            i = 0
            while i < len(tokens):
                # Check if we are at a pair that matches our best_pair
                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == best_pair:
                    # Concatenate the two tokens to form the new merged token
                    new_tokens.append(tokens[i] + tokens[i + 1])
                    i += 2  # Increment by 2 to avoid overlapping
                else:
                    # Keep the token as is
                    new_tokens.append(tokens[i])
                    i += 1
            
            # Update tokens list for the next iteration
            tokens = new_tokens
            
            # d. Record the merge as [token_a, token_b]
            merges.append([best_pair[0], best_pair[1]])

        # 3. Return the list of merges performed
        return merges