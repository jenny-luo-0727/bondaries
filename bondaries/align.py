"""Align tokenizer content to input characters without guessing missing text."""


def _decode(tokenizer, ids: list[int]) -> str:
    return tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)


def greedy_owners(text: str, pieces: list[str | None]) -> list[int]:
    """Match each decoded piece at its earliest occurrence after the cursor.

    Empty and unknown pieces (represented by None) own nothing. Unmatched pieces
    leave the cursor unchanged; skipped characters retain owner -1.
    """
    owners = [-1] * len(text)
    cursor = 0
    for index, piece in enumerate(pieces):
        if not piece:
            continue
        start = text.find(piece, cursor)
        if start >= 0:
            end = start + len(piece)
            owners[start:end] = [index] * len(piece)
            cursor = end
    return owners


def align(text: str, tokenizer, add_special_tokens: bool = True) -> dict:
    """Encode text and return ownership, preserving full-list token indices.

    Lossless encodings use checked cumulative decoding, the same rule as the
    browser. Tokenizer offsets are not used: some tokenizers trim a token's
    leading whitespace from its span, or attribute the first character to a
    standalone word-boundary token, and offsets can be wrong after a tokenizer
    drops a character. Lossy encodings use greedy matching.
    """
    options = {"add_special_tokens": add_special_tokens, "return_special_tokens_mask": True}
    encoded = tokenizer(text, **options)
    ids = [int(value) for value in encoded["input_ids"]]
    tokens = list(tokenizer.convert_ids_to_tokens(ids))
    mask = [int(value) for value in encoded.get("special_tokens_mask", [])]
    special_ids = set(getattr(tokenizer, "all_special_ids", []))
    if len(mask) != len(ids):
        mask = [0] * len(ids)
    mask = [int(special or value in special_ids) for value, special in zip(ids, mask)]
    content_ids = [value for value, special in zip(ids, mask) if not special]
    decoded = _decode(tokenizer, content_ids)
    shift = len(decoded) - len(text)
    lossless = decoded == text or (
        shift > 0 and decoded.endswith(text) and decoded[:shift].isspace()
    )
    alignment_ok = True
    error = None
    if lossless:
        cumulative = []
        previous = ""
        decoded_owners = [-1] * len(decoded)
        for index, (value, special) in enumerate(zip(ids, mask)):
            if special:
                continue
            cumulative.append(value)
            current = _decode(tokenizer, cumulative)
            if not decoded.startswith(current) or not current.startswith(previous):
                alignment_ok = False
            if len(current) >= len(previous):
                decoded_owners[len(previous):len(current)] = [index] * (len(current) - len(previous))
            previous = current
        owners = decoded_owners[max(shift, 0):]
        if not alignment_ok:
            error = "Alignment failed: decoded prefixes are not stable."
    else:
        unknown = getattr(tokenizer, "unk_token_id", None)
        pieces = [None if special or value == unknown else _decode(tokenizer, [value])
                  for value, special in zip(ids, mask)]
        owners = greedy_owners(text, pieces)
    return {"ids": ids, "tokens": tokens, "special_tokens_mask": mask,
            "text_owners": owners, "lossless": lossless,
            "alignment_ok": alignment_ok, "error": error}
