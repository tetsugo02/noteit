"""Turn speaker-assigned words into utterances and a readable transcript."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from pydantic import BaseModel

from .schema import SpeakerAssignedWord

# A pause longer than this starts a new utterance even for the same speaker.
DEFAULT_MAX_GAP_SECONDS = 2.0


class Utterance(BaseModel):
    speaker_label: str | None = None
    start_time_seconds: float
    end_time_seconds: float
    text: str


def _needs_space(previous: str, current: str) -> bool:
    """Space-separate Latin words but join CJK text directly."""
    if not previous or not current:
        return False
    # "world," + "again" -> space; "GPU" + "の" or "," -> no space.
    return (
        previous[-1].isascii()
        and not previous[-1].isspace()
        and current[0].isascii()
        and current[0].isalnum()
    )


def join_words(words: Iterable[str]) -> str:
    text = ""
    for word in words:
        if not word:
            continue
        text += (" " if _needs_space(text, word) else "") + word
    return text


def group_utterances(
    words: Iterable[SpeakerAssignedWord],
    *,
    max_gap_seconds: float = DEFAULT_MAX_GAP_SECONDS,
) -> list[Utterance]:
    utterances: list[Utterance] = []
    current: list[SpeakerAssignedWord] = []

    def flush() -> None:
        if current:
            utterances.append(
                Utterance(
                    speaker_label=current[0].speaker_label,
                    start_time_seconds=current[0].start_time_seconds,
                    end_time_seconds=current[-1].end_time_seconds,
                    text=join_words(word.word for word in current),
                )
            )
            current.clear()

    for word in words:
        if current and (
            word.speaker_label != current[-1].speaker_label
            or word.start_time_seconds - current[-1].end_time_seconds > max_gap_seconds
        ):
            flush()
        current.append(word)
    flush()
    return utterances


def format_timestamp(seconds: float) -> str:
    total = int(seconds)
    return f"{total // 3600:02d}:{total % 3600 // 60:02d}:{total % 60:02d}"


def format_transcript(
    utterances: Iterable[Utterance],
    speaker_names: Mapping[str, str] | None = None,
) -> str:
    """Render ``[HH:MM:SS] speaker: text`` lines, optionally renaming speakers."""
    names = speaker_names or {}
    return "\n".join(
        f"[{format_timestamp(u.start_time_seconds)}] "
        + (
            f"{names.get(u.speaker_label, u.speaker_label)}: "
            if u.speaker_label
            else ""
        )
        + u.text
        for u in utterances
    )
