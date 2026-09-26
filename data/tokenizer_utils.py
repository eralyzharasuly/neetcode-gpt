from typing import List, Dict

class Solution:
    def _greedy_tokenize(self, s: str, vocab: Dict[str, int]) -> List[str]:
        answer = []
        while(s != ""):
            size = len(s)
            f = True
            for i in range(size):
                word = s[:size - i]
                if word in vocab:
                    answer.append(word)
                    s = s[size-i: size]
                    f = False
                    break
            if(f):
                answer.append(s[0])
                s = s[1:]
        return answer

    def tokenize_numbers(self, numbers: List[int], vocab: Dict[str, int]) -> List[List[str]]:
        answer = []
        for i in numbers:
            text = str(i)
            answer.append(self._greedy_tokenize(text, vocab))
        return answer

    def count_tokens(self, text: str, vocab: Dict[str, int]) -> int:
        answer = self._greedy_tokenize(text, vocab)
        return len(answer)

    def fertility_score(self, text: str, vocab: Dict[str, int]) -> float:
        tokens = self._greedy_tokenize(text, vocab)
        words = text.split()
        return round(len(tokens) / len(words), 4)
