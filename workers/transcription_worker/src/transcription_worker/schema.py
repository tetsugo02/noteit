from pydantic import BaseModel, Field


class SpeakerSegment(BaseModel):
    speaker_label: str = Field(..., description="Label of the speaker")
    start_time: float = Field(..., description="Start time of the segment in seconds")
    end_time: float = Field(..., description="End time of the segment in seconds")

    def __repr__(self) -> str:
        return super().__repr__()


class TranscribedWord(BaseModel):
    word: str
    start_time_seconds: float = Field(ge=0.0)
    end_time_seconds: float = Field(ge=0.0)
    probablility: float | None = Field(ge=0.0, le=1.0)


class SpeakerAssignedWord(BaseModel):
    word: str
    start_time_seconds: float = Field(ge=0.0)
    end_time_seconds: float = Field(ge=0.0)
    probablility: float | None = Field(ge=0.0, le=1.0)
    speaker_label: str = Field(..., description="Label of the speaker")
