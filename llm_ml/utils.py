import os
import sys
import re
import warnings
import gc
from typing import Callable

import torch
from transformers import PreTrainedTokenizer
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.classes.preprocessor import TextPreProcessor


def clean_cuda(*args):
    """Deletes CUDA cache."""
    for arg in args:
        del arg
    gc.collect()
    torch.cuda.empty_cache()


def preprocessor(type: str):
    """Creates a text preprocessor.

    Args:
        type: the type of preprocessor to create.
            "twitter" or "reddit".

    Returns:
        The processing function.
    """

    if type.lower() == "twitter":
        return twitter_preprocessor()
    elif type.lower() == "reddit":
        return reddit_preprocessor()
    else:
        return lambda x: x


def twitter_preprocessor(
    normalized_tags: list | None = None, extra_tags: list | None = None
) -> Callable[
    [
        str,
    ],
    str,
]:
    """Creates a Twitter specific text preprocessor.

    Args:
        normalized_tags: `ekphrasis` tags to anonymize,
            e.g. "user" for @userNamE -> user.
        extra_tags: other `ekphrasis` normalizations,
            e.g. "repeated" for Helloooooo -> hello.

    Returns:
        The processing function.
    """

    normalized_tags = normalized_tags or ["url", "email", "phone", "user"]

    extra_tags = extra_tags or [
        "hashtag",
        "elongated",
        "allcaps",
        "repeated",
        "emphasis",
        "censored",
    ]

    def intersect_delimiters(l: list[str], demiliter: str) -> list[str]:
        new_l = []
        for elem in l:
            new_l.extend([elem, demiliter])
        return new_l

    def tag_handler_and_joiner(tokens: list[str]) -> str:
        new_tokens = []
        for token in tokens:
            for tag in normalized_tags:
                if token == f"<{tag}>":
                    token = tag
            for tag in set(extra_tags).difference(["hashtag"]):
                if token in (f"<{tag}>", f"</{tag}>"):
                    token = None
            if token:
                new_tokens.append(token)

        full_str = []
        end_pos = -1

        if "hashtag" in extra_tags:
            start_pos = -1
            while True:
                try:
                    start_pos = new_tokens.index("<hashtag>", start_pos + 1)
                    full_str.extend(
                        intersect_delimiters(
                            new_tokens[end_pos + 1 : start_pos], " "
                        )
                    )
                    end_pos = new_tokens.index("</hashtag>", start_pos + 1)
                    full_str.extend(
                        ["# "]
                        + intersect_delimiters(
                            new_tokens[start_pos + 1 : end_pos], "-"
                        )[:-1]
                        + [" "]
                    )
                except:
                    break

        full_str.extend(intersect_delimiters(new_tokens[end_pos + 1 :], " "))
        return "".join(full_str).strip()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # stop ekphrasis prints
        sys.stdout = open(os.devnull, "w")

        preprocessor = TextPreProcessor(
            normalize=normalized_tags,
            annotate=extra_tags,
            all_caps_tag="wrap",
            fix_text=False,
            segmenter="twitter_2018",
            corrector="twitter_2018",
            unpack_hashtags=True,
            unpack_contractions=True,
            spell_correct_elong=False,
            tokenizer=SocialTokenizer(lowercase=True).tokenize,
        ).pre_process_doc

        # re-enable prints
        sys.stdout = sys.__stdout__

    fun = lambda x: tag_handler_and_joiner(preprocessor(x))
    fun.log = f"ekphrasis: {normalized_tags}, {extra_tags} | tag handler"
    return fun


def reddit_preprocessor(
    normalized_tags: list | None = None, extra_tags: list | None = None
) -> Callable[
    [
        str,
    ],
    str,
]:
    """Creates a Reddit specific text preprocessor.

    Args:
        normalized_tags: `ekphrasis` tags to anonymize,
            e.g. "user" for  /u/userNamE -> user.
        extra_tags: other `ekphrasis` normalizations,
            e.g. "repeated" for Helloooooo -> hello.

    Returns:
        The processing function.
    """

    def prepreprocessor(text):
        text = re.sub("\[NAME\]", "@name", text)
        text = re.sub("\[RELIGION\]", "religion", text)
        text = re.sub("/r/", "", text)
        text = re.sub("/u/[A-Za-z0-9_-]*", "@user", text)
        return text

    preprocessor = twitter_preprocessor(
        normalized_tags=normalized_tags, extra_tags=extra_tags
    )

    return lambda x: preprocessor(prepreprocessor(x))


def string_overlap_idx_in_token_space(
    tokenizer: PreTrainedTokenizer,
    t1: str | torch.Tensor,
    t2: str | torch.Tensor,
) -> int:
    """WARNING: this is not reliable since some characters/strings
    might be broken into multiple tokens, increasing the string length
    after `convert_ids_to_tokens()`. Use `tensor_overlap()` instead.

    Computes index of overlap of two strings in token space
    w.r.t largest string's tokens.

    The overlap may only be visible in the strings because of
    tokenization artifacts (e.g. the smallest string being a
    subword of a bigger string in the full string,
    leading to a different tokenization).
    e.g. looking for "unreasonable" in "this is\nunreasonable"
    will not have an overlap in the token space for Llama 2,
    but will have in string space. Nonetheless, we want the first
    "unreasonable" token in "this is\nunreasonable".

    Returns: the index in the tokenized `t1` where `t2` starts
    if it is a subsequence, otherwise the length of `t1`, and the
    token id."""

    # NOTE: the following doesnt work because of the way the tokenizer
    # tokenizes the input, and the fact that the first token may
    # be a subword token by default.
    # t1 = self.tokenizer(t1, return_tensors="pt")["input_ids"][0]
    # t2 = self.tokenizer(t2, return_tensors="pt")["input_ids"][0]
    # return self._tensor_overlap(t1, t2)

    if isinstance(t1, str):
        s1 = t1
        t1 = tokenizer(t1, return_tensors="pt", add_special_tokens=False)[
            "input_ids"
        ][0]
    else:
        s1 = tokenizer.decode(t1, skip_special_tokens=True)
    if isinstance(t2, str):
        s2 = t2
        t2 = tokenizer(t2, return_tensors="pt", add_special_tokens=False)[
            "input_ids"
        ][0]
    else:
        s2 = tokenizer.decode(t2, skip_special_tokens=True)

    if len(t2) > len(t1):
        t1, t2 = t2, t1
        s1, s2 = s2, s1

    # list of token strs
    t1 = tokenizer.convert_ids_to_tokens(t1)
    t2 = tokenizer.convert_ids_to_tokens(t2)

    # find the overlap in the strings, then look for the
    # index in the tokens
    i = s1.find(s2)

    if i != -1:
        cnt = 0
        idx = -1
        while cnt <= i and idx < len(t1):
            idx += 1
            cnt += len(normalize_control_chars(t1[idx]))
            if idx == 0:
                # there is no initial space
                # (and if there is, it's each own token,
                # so this works, e.g. "_", "_let's", ...)
                if not hasattr(tokenizer, "special_first"):
                    wo_space = tokenizer.tokenize("test")
                    with_space = tokenizer.tokenize(" test")
                    tokenizer.special_first = len(wo_space) != len(with_space)

                if tokenizer.special_first:
                    cnt -= 1
        return idx

    return len(t1)


def normalize_control_chars(text: str) -> str:
    r"""
    Normalize hexadecimal control character representations in text to their actual characters.
    Handles both <0xNN> format and raw \xNN format.

    Args:
        text (str): Input text containing hex control characters

    Returns:
        str: Text with control characters normalized

    Examples:
        >>> normalize_control_chars("Hello<0x0A>World")
        'Hello\nWorld'
        >>> normalize_control_chars("Tab:<0x09>here")
        'Tab:\there'
    """

    # Dictionary of common control characters for human-readable debugging
    COMMON_CONTROLS = {
        '0A': '\n',  # Line Feed
        '0D': '\r',  # Carriage Return
        '09': '\t',  # Tab
        '20': ' ',  # Space
        '0C': '\f',  # Form Feed
        '0B': '\v',  # Vertical Tab
    }

    def hex_to_char(match: re.Match) -> str:
        hex_val = match.group(1)
        # Try common controls first for readability
        if hex_val in COMMON_CONTROLS:
            return COMMON_CONTROLS[hex_val]
        # Convert any other hex value to character
        try:
            return chr(int(hex_val, 16))
        except ValueError:
            return match.group(0)  # Return original if invalid

    # Handle <0xNN> format
    text = re.sub(r'<0x([0-9A-Fa-f]{2})>', hex_to_char, text)

    # Handle raw \xNN format
    text = re.sub(r'\\x([0-9A-Fa-f]{2})', hex_to_char, text)

    return text


def tensor_overlap(t1: torch.Tensor, t2: torch.Tensor) -> int:
    """Computes whether smaller tensor `t2` is a subsequence
    of tensor `t1`. Both are 1D.

    Returns the index in `t1` where `t2` starts if it is a subsequence,
    otherwise the length of `t1`.
    """

    if len(t2) > len(t1):
        t1, t2 = t2, t1

    for i in range(len(t1) - len(t2) + 1):
        if len(t2) == 1:
            if t1[i] == t2[0]:
                return i
        elif (t1[i : i + len(t2)] == t2).all():
            return i

    return len(t1)
