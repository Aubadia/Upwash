"""Paloma decontamination, following Paloma (arXiv 2312.10523) Appendix C.1 (task 033, step 1).

- Text is matched at the paragraph level: newline-separated spans.
- Evaluation paragraphs shorter than 13 Unicode-segmented tokens are ignored, and so are
  paragraphs made only of punctuation, spaces and emoji.
- A training document is removed whole if any of its paragraphs exactly matches an
  eligible evaluation paragraph.

Agreed choices (2026-11-20; the paper leaves them open):

- Segmentation: Unicode default word boundaries (UAX #29) through the `regex` package's
  WORD mode. A token is any non-whitespace segment, so punctuation marks count.
- "Only punctuation, spaces and emoji": a paragraph with no letter or number character.
- Exact matching: no normalization, and a set of paragraph hashes instead of a Bloom
  filter, so there are no false positives.
- Code sources are not decontaminated (Paloma C.1). That rule lives with the source
  configuration, not here.
"""
import pytest

from upwash.prep.decontam import (ContaminationIndex, is_eligible_eval_paragraph,
                                  split_paragraphs, unicode_token_count)

# 13 and 12 Unicode-segmented tokens.
LONG = "The committee approved the new budget after a long debate on Tuesday ."
SHORT = "The committee approved the new budget after a long debate on Tuesday"
assert len(LONG.split()) == 13 and len(SHORT.split()) == 12


def test_paragraphs_are_newline_separated_spans_without_empty_ones():
    assert split_paragraphs("first line\nsecond line\n\nthird") == ["first line", "second line", "third"]
    assert split_paragraphs("") == []
    assert split_paragraphs("one paragraph") == ["one paragraph"]


@pytest.mark.parametrize(("text", "count"), [
    ("Hello, world.", 4),                     # punctuation marks are tokens
    ("don't stop", 2),                        # UAX #29 keeps the apostrophe inside the word
    ("pi is 3.14 here", 4),                   # and the decimal number together
    ("我們今天去學校", 7),                      # each Han character is a segment; no spaces needed
    ("   ", 0),
])
def test_unicode_token_count_uses_uax29_word_segments(text, count):
    assert unicode_token_count(text) == count


def test_eval_paragraphs_need_at_least_13_tokens():
    assert unicode_token_count(LONG) == 13
    assert unicode_token_count(SHORT) == 12
    assert is_eligible_eval_paragraph(LONG)
    assert not is_eligible_eval_paragraph(SHORT)


@pytest.mark.parametrize("paragraph", [
    "! " * 20,
    "... --- *** ### !!! ??? ,,, ;;; ::: ((( ))) [[[ ]]]",
    "🙂 🎉 👍 " * 6,
    "🙂!🙂!🙂! ... ??? *** ~~~ ### @@@ %%% ^^^ &&& ---",
])
def test_paragraphs_of_only_punctuation_spaces_and_emoji_are_ignored(paragraph):
    assert unicode_token_count(paragraph) >= 13
    assert not is_eligible_eval_paragraph(paragraph)


def test_a_number_alone_makes_a_long_paragraph_eligible():
    assert is_eligible_eval_paragraph("2026 " + "! " * 12)


def test_a_document_is_contaminated_when_any_paragraph_matches_exactly():
    index = ContaminationIndex.from_texts([f"Intro.\n{LONG}\nOutro."])

    assert index.is_contaminated(f"Unrelated opening paragraph.\n{LONG}\nMore text.")
    assert index.is_contaminated(LONG)


def test_partial_or_normalized_matches_do_not_count():
    index = ContaminationIndex.from_texts([LONG])

    assert not index.is_contaminated(f"Prefix {LONG}")        # inside a longer paragraph
    assert not index.is_contaminated(LONG.lower())            # no case normalization
    assert not index.is_contaminated(LONG.replace(" ", "  "))  # no whitespace normalization
    assert not index.is_contaminated(LONG[:-2])               # a strict prefix


def test_short_or_symbol_only_eval_paragraphs_never_contaminate():
    symbols = "! " * 20
    index = ContaminationIndex.from_texts([f"{SHORT}\n{symbols}"])

    assert len(index) == 0
    assert not index.is_contaminated(f"{SHORT}\n{symbols}")


def test_the_index_counts_distinct_eligible_paragraphs_across_texts():
    other = LONG.replace("Tuesday", "Friday")
    index = ContaminationIndex.from_texts([LONG, f"{LONG}\n{other}", SHORT])

    assert len(index) == 2
    assert index.is_contaminated(other)


def test_an_empty_index_contaminates_nothing():
    index = ContaminationIndex.from_texts([])

    assert len(index) == 0
    assert not index.is_contaminated(LONG)


def test_from_texts_accepts_a_one_pass_iterable():
    # Paloma is streamed from Flock; the index must not need to iterate twice.
    index = ContaminationIndex.from_texts(iter([LONG]))

    assert index.is_contaminated(LONG)
