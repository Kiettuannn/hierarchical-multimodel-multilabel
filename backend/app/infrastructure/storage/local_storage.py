import os
import shutil
import uuid
from pathlib import Path
from typing import BinaryIO

from app.core.config import settings
from app.core.logging import logger
from app.domain.interfaces.storage import IVideoStorage


class LocalVideoStorage(IVideoStorage):
    """Local filesystem adapter for temporary video uploads."""

    def __init__(self, base_dir: Path = settings.UPLOAD_DIR):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    async def save(self, filename: str, file_stream: BinaryIO) -> str:
        """Write the file stream to a uniquely named local file."""
        unique_name = f"{uuid.uuid4().hex}_{filename}"
        destination = self.base_dir / unique_name

        with open(destination, "wb") as buffer:
            shutil.copyfileobj(file_stream, buffer)

        logger.info(f"Saved video to temporary storage: {destination}")
        return str(destination)

    async def delete(self, file_path: str) -> None:
        """Safely delete a temporary video file if it exists."""
        target = Path(file_path)
        if target.exists():
            try:
                os.remove(target)
                logger.info(f"Cleaned up temporary video: {file_path}")
            except OSError as error:
                logger.warning(f"Could not delete temporary video {file_path}: {error}")
