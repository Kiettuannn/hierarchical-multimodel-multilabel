from abc import ABC, abstractmethod
from typing import BinaryIO


class IVideoStorage(ABC):
    """Abstract interface defining the contract for video file storage."""

    @abstractmethod
    async def save(self, filename: str, file_stream: BinaryIO) -> str:
        """Save an incoming video stream and return its stored system path."""
        raise NotImplementedError

    @abstractmethod
    async def delete(self, file_path: str) -> None:
        """Remove a temporarily stored video file after processing."""
        raise NotImplementedError
