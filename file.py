from pathlib import Path
import hashlib
from enum import Enum
from typing import Optional
from parse_config import ParseConfig

class FileStatus(Enum):
    NOT_PARSED = "Not Parsed"
    PARSED = "Parsed"
    PROCESSED = "Processed"
    ERROR = "Error"

class FileState:
    def __init__(self):
        self._status = FileStatus.NOT_PARSED
        self._error_message: Optional[str] = None

    @property
    def status(self) -> FileStatus:
        return self._status

    @property
    def error_message(self) -> Optional[str]:
        return self._error_message

    def set_status(self, status: str | FileStatus):
        if isinstance(status, str):
            status = FileStatus(status)
        self._status = status

    def set_error(self, message: str):
        self._error_message = message
        self._status = FileStatus.ERROR

class File:
    """
    Stores basic metadata about a file and generates a unique file ID.
    """
    def __init__(self, filepath: Path):
        self.filepath = filepath
        self.filename = filepath.name
        self.filetype = filepath.suffix
        self.filesize = filepath.stat().st_size
        self.last_modified = filepath.stat().st_mtime

        # Generate unique file ID using path + last modified time
        hash_input = f"{str(self.filepath)}:{self.last_modified}".encode("utf-8")
        self.file_id = hashlib.sha1(hash_input).hexdigest()[:8]  # e.g., 'a3f9c12b'

        self.state = FileState()  # Add state management
        self.parse_config = ParseConfig()  # Add parsing configuration and results
