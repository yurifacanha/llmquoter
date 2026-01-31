import re
import unicodedata

QUOTE_PATTERN = re.compile(r"##begin_quote##\s*(.*?)\s*##end_quote##", re.DOTALL)


def parse_quotes(text: str) -> list[str]:
    if not text:
        return []
    quotes = QUOTE_PATTERN.findall(text)
    return [q.strip() for q in quotes if q.strip()]


def format_score(text: str) -> float:
    if not text:
        return 0.0
    text = text.strip().strip("\ufeff")
    if not text:
        return 0.0
    text_no_spaces = re.sub(r"\s", "", text)
    if not text_no_spaces:
        return 0.0
    blocks = QUOTE_PATTERN.findall(text)
    if not blocks:
        return 0.0
    reconstructed = "".join(f"##begin_quote##{b}##end_quote##" for b in blocks)
    reconstructed_no_spaces = re.sub(r"\s", "", reconstructed)
    if reconstructed_no_spaces != text_no_spaces:
        nfc_text = unicodedata.normalize("NFC", text_no_spaces)
        nfc_reconstructed = unicodedata.normalize("NFC", reconstructed_no_spaces)
        if nfc_text != nfc_reconstructed:
            return 0.0
    return 1.0
