from re import Pattern
from typing import List, Tuple

import re


_PART_PATTERN: Pattern = re.compile(
    r"""
    ^\s*
    \[\s*(?:-?\d+\s*(?:,\s*-?\d+\s*)*)\]      # [coords]
    \[\s*[A-Za-z_][A-Za-z0-9_]*\s*\]          # [label]
    (?:\[\s*[A-Za-z_][A-Za-z0-9_]*\s*\])*     # [meta]...
    \s*$
    """,
    re.VERBOSE,
)


def parse_layout_string(
        string: str
) -> List[Tuple[Tuple[float, ...], str, List[str]]]:
    """
    Parse a serialized layout string into structured layout components.

    The input string is expected to consist of one or more layout segments
    separated by the literal delimiter '[PAIR_SEP]'. Each segment follows
    a bracketed token format:

        [x1,x2,x3,x4][label][meta1][meta2]...

    Where:
        - The first bracket contains a comma-separated list of integers
          representing coordinates or indices.
        - The second bracket contains a primary label.
        - Any subsequent brackets contain optional metadata fields.

    The function extracts and converts each segment into a tuple of:
        (
            List[int],   # parsed coordinates
            str,         # primary label
            List[str]    # optional metadata entries
        )

    Args:
        string (str): Serialized layout string to be parsed.

    Returns:
        List[Tuple[List[int], str, List[str]]]: A list of parsed layout tuples,
        one per layout segment in the input string.

    Raises:
        ValueError:
            - If any segment fails regex validation.
            - If integer coordinate conversion fails.
    """
    parts: List[str] = split_layout_string(string)
    results: List[Tuple[Tuple[float, ...], str, List[str]]] = []
    for part in parts:
        # Check the part for a valid pattern.
        if not _PART_PATTERN.match(part):
            raise ValueError(f"Invalid layout segment: {part!r}")

        # Parse the part.
        tokens = []
        index = 0
        while index < len(part):
            if part[index] == "[":
                length = part.index("]", index)
                tokens.append(part[index+1:length])
                index = length + 1
            else:
                index += 1
        coordinates = tuple(list(float(x) for x in tokens[0].split(",")))
        label = tokens[1]
        meta = tokens[2:]

        results.append((coordinates, label, meta))
    return results


def split_layout_string(string: str) -> List[str]:
    """
    Split a serialized layout string into its individual components.

    The input string may contain multiple layout segments separated by the
    literal delimiter '[PAIR_SEP]'. Each of those segments may further contain
    sub-segments separated by '[RELATION_SEP]'.

    This function performs a two-stage split:
        1. Split the input string on '[PAIR_SEP]'.
        2. Split each resulting part on '[RELATION_SEP]'.

    All resulting segments are stripped of leading and trailing whitespace,
    and empty segments are filtered out.

    Args:
        string (str): The serialized layout string to split.

    Returns:
        List[str]: A flat list of non-empty, trimmed layout segments.
    """
    results: List[str] = []
    parts: List[str] = string.split("[PAIR_SEP]")
    for part in parts:
        results.extend(part.split("[RELATION_SEP]"))
    return list(filter(
        lambda part: len(parts) > 0, [part.strip() for part in results]
    ))
