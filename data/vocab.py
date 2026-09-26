from typing import Dict, List, Tuple

class Solution:
    def build_vocab(self, text: str) -> Tuple[Dict[str, int], Dict[int, str]]:
        text = sorted(set(text))
        stoi = { s: idx for idx, s in enumerate(text)}
        itos = { idx: s for idx, s in enumerate(text)}
        return stoi, itos

    def encode(self, text: str, stoi: Dict[str, int]) -> List[int]:
        answer = []
        for s in text:
            answer.append(stoi[s])
        return answer

    def decode(self, ids: List[int], itos: Dict[int, str]) -> str:
        answer = ""
        for i in ids:
            answer += itos[i]
        return answer;
