import re
from typing import Dict, Pattern, List, Tuple

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

# Mappings from symbol to numeric ID and vice versa:
_symbol_to_id = {s: i for i, s in enumerate(phoneme_symbols)}
_id_to_symbol = {i: s for i, s in enumerate(phoneme_symbols)}


accent_symbols = [
    "[",  # pitch up
    "]",  # pitch down
    "#",  # accent boundary
    "?",  # end of sentence(question)
    "_",  # not changing accent
]

_accent_to_id: Dict[str, int] = {s: i for i, s in enumerate(accent_symbols)}
_id_to_accent: Dict[int, str] = {i: s for i, s in enumerate(accent_symbols)}

# full context label to accent label from ttslearn
def numeric_feature_by_regex(regex: Pattern[str], s: str) -> int:
    match = re.search(regex, s)
    if match is None:
        return -50
    return int(match.group(1))


def extract_phoneme_and_accents(labels: List[str]) -> Tuple[List[str], List[str]]:
    phonemes = []
    accents = []
    N = len(labels)

    for n in range(len(labels)):
        lab_curr = labels[n]

        p3 = re.search(r"\-(.*?)\+", lab_curr).group(1)

        if p3 == "sil":
            assert n == 0 or n == N - 1
            # NOTE: FastSpeech2においては、SOS/EOSが不要なので、導入しない
            # if n == 0:
            #     accents.append("^")
            # elif n == N-1:
            #     e3 = numeric_feature_by_regex(r"!(\d+)_", lab_curr)
            #     if e3 == 0:
            #         accents.append("$")
            #     elif e3 == 1:
            #         accents.append("?")
            continue
        else:
            phonemes.append(p3)
            if p3 == "pau":
                accents.append("#")
                continue

        # アクセント型および位置情報（前方または後方）
        a1 = numeric_feature_by_regex(r"/A:([0-9\-]+)\+", lab_curr)
        a2 = numeric_feature_by_regex(r"\+(\d+)\+", lab_curr)
        a3 = numeric_feature_by_regex(r"\+(\d+)/", lab_curr)
        # アクセント句におけるモーラ数
        f1 = numeric_feature_by_regex(r"/F:(\d+)_", lab_curr)
        lab_next = labels[n + 1]
        a2_next = numeric_feature_by_regex(r"\+(\d+)\+", lab_next)
        # アクセント境界
        if (a3 == 1 and a2_next == 1) or n == N - 2:
            f3 = numeric_feature_by_regex(r"#(\d+)_", lab_curr)
            if f3 == 1:
                accents.append("?")
            else:
                accents.append("#")
        # ピッチの立ち下がり（アクセント核）
        elif a1 == 0 and a2_next == a2 + 1 and a2 != f1:
            accents.append("]")
        # ピッチの立ち上がり
        elif a2 == 1 and a2_next == 2:
            accents.append("[")
        else:
            accents.append("_")

    assert len(phonemes) == len(accents)
    return phonemes, accents
