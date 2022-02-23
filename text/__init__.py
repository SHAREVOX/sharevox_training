from typing import Dict

phoneme_symbols = [
    "pau",
    "A",
    "E",
    "I",
    "N",
    "O",
    "U",
    "a",
    "b",
    "by",
    "ch",
    "cl",
    "d",
    "dy",
    "e",
    "f",
    "g",
    "gw",
    "gy",
    "h",
    "hy",
    "i",
    "j",
    "k",
    "kw",
    "ky",
    "m",
    "my",
    "n",
    "ny",
    "o",
    "p",
    "py",
    "r",
    "ry",
    "s",
    "sh",
    "t",
    "ts",
    "ty",
    "u",
    "v",
    "w",
    "y",
    "z",
]

phoneme_to_id: Dict[str, int] = {s: i for i, s in enumerate(phoneme_symbols)}
id_to_phoneme: Dict[int, str] = {i: s for i, s in enumerate(phoneme_symbols)}

accent_symbols = [
    "[",  # pitch up
    "]",  # pitch down
    "#",  # accent boundary
    "?",  # end of sentence(question)
    "_",  # not changing accent
]

accent_to_id: Dict[str, int] = {s: i for i, s in enumerate(accent_symbols)}
id_to_accent: Dict[int, str] = {i: s for i, s in enumerate(accent_symbols)}

