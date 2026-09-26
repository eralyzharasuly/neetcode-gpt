from typing import List


class Solution:
    def get_merges(self, corpus: str, num_merges: int) -> List[List[str]]:
        tokens = list(corpus)
        merges = []
        for merg in range(num_merges):
            pairs = {}
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                pairs[pair] = pairs.get(pair, 0) + 1
            best_pair = min(pairs, key=lambda p: (-pairs[p], p))

            new_tokens = []
            j = 0
            while j < len(tokens):
                if j < len(tokens) - 1 and (tokens[j], tokens[j + 1]) == best_pair:
                    new_tokens.append(tokens[j] + tokens[j + 1])
                    j += 2
                else:
                    new_tokens.append(tokens[j])
                    j += 1
            tokens = new_tokens
            merges.append(list(best_pair))
        return merges