"""Clean short English texts and count frequent words."""

from collections import Counter
import re


SAMPLE_TEXTS = [
    "  Python is FUN!  ",
    "I love Python.",
    "Python is useful!",
    "I love AI.",
]


def normalize_text(text: str) -> str:
    """Lowercase and remove numbers/punctuation, then collapse whitespace."""
    cleaned = text.lower()
    cleaned = re.sub(r"\d+", "", cleaned)
    cleaned = re.sub(r"[!?.]", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def word_frequencies(texts: list[str]) -> Counter[str]:
    """Return word frequencies across all texts (not per-document counts)."""
    normalized = [normalize_text(text) for text in texts]
    words = [word for text in normalized for word in text.split()]
    return Counter(words)


if __name__ == "__main__":
    print("Preprocessed texts:")
    for sentence in SAMPLE_TEXTS:
        print("-", normalize_text(sentence))
    print("\nTop 3 words:")
    for word, count in word_frequencies(SAMPLE_TEXTS).most_common(3):
        print(f"{word}: {count}")
