import unicodedata
import hashlib

import regex

def split_paragraphs(text) -> list[str]:
    paragraphs = []
    for p in text.split("\n"):
        if p != "":
            paragraphs.append(p)
    return paragraphs

def unicode_token_count(text) -> int:
    segments = regex.split(r"\b", text, flags=regex.WORD | regex.V1)
    count = 0
    for s in segments:
        if not (s == "" or s.isspace()):
            count += 1
    return count
    
    

def is_eligible_eval_paragraph(paragraph) -> bool:
    if unicode_token_count(paragraph) < 13:
        return False
    
    for ch in paragraph:
        if unicodedata.category(ch).startswith(("L", "N")):
            return True

    return False

def _paragraph_hash(paragraph):
    return hashlib.blake2b(paragraph.encode("utf-8"), digest_size=16).digest()

class ContaminationIndex:
    def __init__(self, hashes):
        self.hashes = hashes

    @classmethod
    def from_texts(cls, eval_texts):
        hashes = set()
        for text in eval_texts:
            for paragraph in split_paragraphs(text):
                if is_eligible_eval_paragraph(paragraph):
                    hashes.add(_paragraph_hash(paragraph))
        return cls(hashes)

    def __len__(self):
        return len(self.hashes)

    def is_contaminated(self, document):
        paragraphs = split_paragraphs(document)
        for p in paragraphs:
            if _paragraph_hash(p) in self.hashes:
                return True
        return False