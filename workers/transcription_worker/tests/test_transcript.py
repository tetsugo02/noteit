from transcription_worker.schema import SpeakerAssignedWord
from transcription_worker.transcript import (
    format_transcript,
    group_utterances,
    join_words,
)


def word(text: str, start: float, speaker: str) -> SpeakerAssignedWord:
    return SpeakerAssignedWord(
        word=text,
        start_time_seconds=start,
        end_time_seconds=start + 0.4,
        probablility=None,
        speaker_label=speaker,
    )


def test_join_words_spaces_latin_but_not_japanese() -> None:
    assert join_words(["hello", "world,", "again"]) == "hello world, again"
    assert join_words(["今日は", "会議", "です。"]) == "今日は会議です。"
    assert join_words(["GPU", "の", "話"]) == "GPUの話"


def test_group_utterances_splits_on_speaker_change_and_long_pause() -> None:
    words = [
        word("こんにちは", 0.0, "SPEAKER_00"),
        word("です", 0.5, "SPEAKER_00"),
        word("はい", 1.0, "SPEAKER_01"),
        word("続き", 5.0, "SPEAKER_01"),  # 3.6s pause -> new utterance
    ]

    utterances = group_utterances(words)

    assert [(u.speaker_label, u.text) for u in utterances] == [
        ("SPEAKER_00", "こんにちはです"),
        ("SPEAKER_01", "はい"),
        ("SPEAKER_01", "続き"),
    ]


def test_format_transcript_renames_speakers() -> None:
    utterances = group_utterances([word("hi", 3725.0, "SPEAKER_00")])

    assert (
        format_transcript(utterances, {"SPEAKER_00": "Tanaka"})
        == "[01:02:05] Tanaka: hi"
    )
