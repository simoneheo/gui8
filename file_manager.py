from typing import Dict, List, Optional, Set
from pathlib import Path
from collections import defaultdict
from file import File, FileStatus
import time


class FileManager:
    """
    Manages File objects with efficient lookup and state tracking.
    """
    
    def __init__(self):
        self._files: Dict[str, File] = {}  # file_id -> File
        self._path_to_id: Dict[str, str] = {}  # str(path) -> file_id
        self._files_by_status: Dict[FileStatus, Set[str]] = defaultdict(set)
        self._creation_order: List[str] = []
        
        # Stats
        self._stats = {
            'total_files': 0,
            'last_added': None,
            'last_accessed': None
        }
    
    def add_file(self, file_obj: File) -> bool:
        """
        Add a File object to the manager.
        
        Args:
            file_obj: File object to add
            
        Returns:
            True if added successfully, False if already exists
        """
        file_id = file_obj.file_id
        path_str = str(file_obj.filepath)
        
        if file_id in self._files:
            return False  # Already exists
        
        # Add to main storage
        self._files[file_id] = file_obj
        self._path_to_id[path_str] = file_id
        self._creation_order.append(file_id)
        
        # Update status tracking
        self._files_by_status[file_obj.state.status].add(file_id)
        
        # Update stats
        self._stats['total_files'] += 1
        self._stats['last_added'] = time.time()
        
        return True
    
    def get_file(self, file_id: str) -> Optional[File]:
        """Get File object by file_id."""
        self._stats['last_accessed'] = time.time()
        return self._files.get(file_id)
    
    def get_file_by_path(self, file_path: Path) -> Optional[File]:
        """Get File object by file path."""
        path_str = str(file_path)
        file_id = self._path_to_id.get(path_str)
        if file_id:
            return self.get_file(file_id)
        return None
    
    def remove_file(self, file_id: str) -> bool:
        """
        Remove a File object from the manager.
        
        Args:
            file_id: ID of file to remove
            
        Returns:
            True if removed, False if not found
        """
        if file_id not in self._files:
            return False
        
        file_obj = self._files[file_id]
        path_str = str(file_obj.filepath)
        
        # Remove from all tracking structures
        del self._files[file_id]
        self._path_to_id.pop(path_str, None)
        self._creation_order.remove(file_id)
        
        # Remove from status tracking
        for status_set in self._files_by_status.values():
            status_set.discard(file_id)
        
        self._stats['total_files'] -= 1
        return True
    
    def update_file_status(self, file_id: str, old_status: FileStatus, new_status: FileStatus):
        """
        Update file status tracking when a file's status changes.
        This should be called by File objects when their status changes.
        """
        self._files_by_status[old_status].discard(file_id)
        self._files_by_status[new_status].add(file_id)
    
    def get_files_by_status(self, status: FileStatus) -> List[File]:
        """Get all files with a specific status."""
        file_ids = self._files_by_status[status]
        return [self._files[fid] for fid in file_ids if fid in self._files]
    
    def get_all_files(self) -> List[File]:
        """Get all File objects."""
        return list(self._files.values())
    
    def get_file_ids(self) -> List[str]:
        """Get all file IDs."""
        return list(self._files.keys())
    
    def get_files_in_order(self) -> List[File]:
        """Get files in the order they were added."""
        return [self._files[fid] for fid in self._creation_order if fid in self._files]
    
    def has_file(self, file_id: str) -> bool:
        """Check if file exists in manager."""
        return file_id in self._files
    
    def has_file_path(self, file_path: Path) -> bool:
        """Check if file path exists in manager."""
        return str(file_path) in self._path_to_id
    
    def get_file_count(self) -> int:
        """Get total number of files."""
        return len(self._files)
    
    def get_file_count_by_status(self) -> Dict[FileStatus, int]:
        """Get count of files by status."""
        return {status: len(file_ids) for status, file_ids in self._files_by_status.items()}
    
    def clear_all(self):
        """Remove all files from manager."""
        self._files.clear()
        self._path_to_id.clear()
        self._files_by_status.clear()
        self._creation_order.clear()
        self._stats['total_files'] = 0
    
    def get_stats(self) -> Dict:
        """Get manager statistics."""
        return {
            **self._stats,
            'files_by_status': self.get_file_count_by_status(),
            'total_tracked': len(self._files)
        }
    
    def cleanup_stale_files(self) -> int:
        """
        Remove files that no longer exist on disk.
        
        Returns:
            Number of files removed
        """
        removed_count = 0
        stale_ids = []
        
        for file_id, file_obj in self._files.items():
            if not file_obj.filepath.exists():
                stale_ids.append(file_id)
        
        for file_id in stale_ids:
            if self.remove_file(file_id):
                removed_count += 1
        
        return removed_count
    
    def get_recent_files(self, limit: int = 10) -> List[File]:
        """Get recently added files."""
        recent_ids = self._creation_order[-limit:] if limit > 0 else self._creation_order
        return [self._files[fid] for fid in reversed(recent_ids) if fid in self._files]
    
    def find_files_by_name_pattern(self, pattern: str) -> List[File]:
        """Find files by filename pattern (case-insensitive)."""
        pattern_lower = pattern.lower()
        matches = []
        
        for file_obj in self._files.values():
            if pattern_lower in file_obj.filename.lower():
                matches.append(file_obj)
        
        return matches
    
    def get_file_info_summary(self) -> Dict:
        """Get summary information about all files."""
        if not self._files:
            return {'total': 0, 'summary': 'No files loaded'}
        
        status_counts = self.get_file_count_by_status()
        total_size = sum(f.filesize for f in self._files.values())
        
        return {
            'total': len(self._files),
            'parsed': status_counts.get(FileStatus.PARSED, 0),
            'errors': status_counts.get(FileStatus.ERROR, 0),
            'not_parsed': status_counts.get(FileStatus.NOT_PARSED, 0),
            'total_size_bytes': total_size,
            'most_recent': self._creation_order[-1] if self._creation_order else None
        }
    
    def __len__(self) -> int:
        """Return number of files managed."""
        return len(self._files)
    
    def __contains__(self, file_id: str) -> bool:
        """Check if file_id is managed."""
        return file_id in self._files
    
    def __iter__(self):
        """Iterate over File objects."""
        return iter(self._files.values()) 