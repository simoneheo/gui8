from typing import Dict, List, Optional, Set, Union
from collections import defaultdict
from channel import Channel, SourceType
import time


class ChannelManager:
    """
    Manages Channel objects with efficient lookup, filtering, and relationship tracking.
    """
    
    def __init__(self):
        self._channels: Dict[str, Channel] = {}  # channel_id -> Channel
        self._channels_by_file: Dict[str, Set[str]] = defaultdict(set)  # file_id -> set of channel_ids
        self._channels_by_type: Dict[SourceType, Set[str]] = defaultdict(set)  # type -> set of channel_ids
        self._channels_by_step: Dict[int, Set[str]] = defaultdict(set)  # step -> set of channel_ids
        self._creation_order: List[str] = []
        
        # Lineage tracking
        self._children: Dict[str, Set[str]] = defaultdict(set)  # parent_id -> set of child_ids
        self._parents: Dict[str, Set[str]] = defaultdict(set)   # child_id -> set of parent_ids
        
        # Stats
        self._stats = {
            'total_channels': 0,
            'last_added': None,
            'last_accessed': None
        }
    
    def add_channel(self, channel: Channel) -> bool:
        """
        Add a Channel object to the manager.
        
        Args:
            channel: Channel object to add
            
        Returns:
            True if added successfully, False if already exists
        """
        channel_id = channel.channel_id
        
        if channel_id in self._channels:
            return False  # Already exists
        
        # Add to main storage
        self._channels[channel_id] = channel
        self._creation_order.append(channel_id)
        
        # Update indices
        if channel.file_id:
            self._channels_by_file[channel.file_id].add(channel_id)
        
        if channel.type:
            self._channels_by_type[channel.type].add(channel_id)
        
        self._channels_by_step[channel.step].add(channel_id)
        
        # Update lineage tracking
        if channel.parent_ids:
            for parent_id in channel.parent_ids:
                self._children[parent_id].add(channel_id)
                self._parents[channel_id].add(parent_id)
        
        # Update stats
        self._stats['total_channels'] += 1
        self._stats['last_added'] = time.time()
        
        return True
    
    def add_channels(self, channels: List[Channel]) -> int:
        """
        Add multiple channels at once.
        
        Returns:
            Number of channels successfully added
        """
        added_count = 0
        for channel in channels:
            if self.add_channel(channel):
                added_count += 1
        return added_count
    
    def get_channel(self, channel_id: str) -> Optional[Channel]:
        """Get Channel object by channel_id."""
        self._stats['last_accessed'] = time.time()
        return self._channels.get(channel_id)
    
    def remove_channel(self, channel_id: str) -> bool:
        """
        Remove a Channel object from the manager.
        
        Args:
            channel_id: ID of channel to remove
            
        Returns:
            True if removed, False if not found
        """
        if channel_id not in self._channels:
            return False
        
        channel = self._channels[channel_id]
        
        # Remove from main storage
        del self._channels[channel_id]
        self._creation_order.remove(channel_id)
        
        # Remove from indices
        if channel.file_id:
            self._channels_by_file[channel.file_id].discard(channel_id)
        
        if channel.type:
            self._channels_by_type[channel.type].discard(channel_id)
        
        self._channels_by_step[channel.step].discard(channel_id)
        
        # Remove from lineage tracking
        if channel.parent_ids:
            for parent_id in channel.parent_ids:
                if parent_id in self._children:
                    self._children[parent_id].discard(channel_id)
                if channel_id in self._parents:
                    self._parents[channel_id].discard(parent_id)
        
        # Remove any children relationships - only if channel has children
        if channel_id in self._children:
            for child_id in self._children[channel_id]:
                if child_id in self._parents:
                    self._parents[child_id].discard(channel_id)
            del self._children[channel_id]
        
        # Remove from parents tracking - only if channel has entry
        if channel_id in self._parents:
            del self._parents[channel_id]
        
        self._stats['total_channels'] -= 1
        return True
    
    def get_channels_by_file(self, file_id: str) -> List[Channel]:
        """Get all channels from a specific file."""
        channel_ids = self._channels_by_file[file_id]
        return [self._channels[cid] for cid in channel_ids if cid in self._channels]
    
    def get_channels_by_type(self, source_type: SourceType) -> List[Channel]:
        """Get all channels of a specific type."""
        channel_ids = self._channels_by_type[source_type]
        return [self._channels[cid] for cid in channel_ids if cid in self._channels]
    
    def get_channels_by_step(self, step: int) -> List[Channel]:
        """Get all channels at a specific processing step."""
        channel_ids = self._channels_by_step[step]
        return [self._channels[cid] for cid in channel_ids if cid in self._channels]
    
    def get_raw_channels(self) -> List[Channel]:
        """Get all raw (step=0) channels."""
        return self.get_channels_by_step(0)
    
    def get_processed_channels(self) -> List[Channel]:
        """Get all processed (step>0) channels."""
        processed = []
        for step in range(1, max(self._channels_by_step.keys()) + 1 if self._channels_by_step else 1):
            processed.extend(self.get_channels_by_step(step))
        return processed
    
    def get_child_channels(self, parent_id: str) -> List[Channel]:
        """Get all direct child channels of a parent."""
        child_ids = self._children[parent_id]
        return [self._channels[cid] for cid in child_ids if cid in self._channels]
    
    def get_parent_channels(self, child_id: str) -> List[Channel]:
        """Get all direct parent channels of a child."""
        parent_ids = self._parents[child_id]
        return [self._channels[pid] for pid in parent_ids if pid in self._channels]
    
    def get_channels_by_lineage(self, channel_id: str) -> Dict:
        """
        Get full lineage tree for a channel (parents and children).
        
        Returns:
            Dict with 'parents', 'children', and 'siblings' keys
        """
        if channel_id not in self._channels:
            return {}
        
        parents = self.get_parent_channels(channel_id)
        children = self.get_child_channels(channel_id)
        
        # Find siblings (channels with same parents)
        siblings = []
        for parent in parents:
            siblings.extend(self.get_child_channels(parent.channel_id))
        
        # Remove self from siblings
        siblings = [c for c in siblings if c.channel_id != channel_id]
        
        return {
            'parents': parents,
            'children': children,
            'siblings': siblings
        }
    
    def get_all_channels(self) -> List[Channel]:
        """Get all Channel objects."""
        return list(self._channels.values())
    
    def get_channel_ids(self) -> List[str]:
        """Get all channel IDs."""
        return list(self._channels.keys())
    
    def get_channels_in_order(self) -> List[Channel]:
        """Get channels in the order they were added."""
        return [self._channels[cid] for cid in self._creation_order if cid in self._channels]
    
    def has_channel(self, channel_id: str) -> bool:
        """Check if channel exists in manager."""
        return channel_id in self._channels
    
    def get_channel_count(self) -> int:
        """Get total number of channels."""
        return len(self._channels)
    
    def get_channel_count_by_file(self) -> Dict[str, int]:
        """Get count of channels by file."""
        return {file_id: len(channel_ids) for file_id, channel_ids in self._channels_by_file.items()}
    
    def get_channel_count_by_type(self) -> Dict[SourceType, int]:
        """Get count of channels by type."""
        return {source_type: len(channel_ids) for source_type, channel_ids in self._channels_by_type.items()}
    
    def get_channel_count_by_step(self) -> Dict[int, int]:
        """Get count of channels by processing step."""
        return {step: len(channel_ids) for step, channel_ids in self._channels_by_step.items()}
    
    def remove_channels_by_file(self, file_id: str) -> int:
        """
        Remove all channels from a specific file.
        
        Returns:
            Number of channels removed
        """
        channel_ids = list(self._channels_by_file[file_id])
        removed_count = 0
        
        for channel_id in channel_ids:
            if self.remove_channel(channel_id):
                removed_count += 1
        
        return removed_count
    
    def find_channels_by_label_pattern(self, pattern: str) -> List[Channel]:
        """Find channels by legend label pattern (case-insensitive)."""
        pattern_lower = pattern.lower()
        matches = []
        
        for channel in self._channels.values():
            if channel.legend_label and pattern_lower in channel.legend_label.lower():
                matches.append(channel)
        
        return matches
    
    def find_channels_by_ylabel(self, ylabel: str) -> List[Channel]:
        """Find channels by y-axis label."""
        matches = []
        for channel in self._channels.values():
            if channel.ylabel == ylabel:
                matches.append(channel)
        return matches
    
    def get_visible_channels(self) -> List[Channel]:
        """Get all channels that are set to be visible (show=True)."""
        return [channel for channel in self._channels.values() if channel.show]
    
    def get_hidden_channels(self) -> List[Channel]:
        """Get all channels that are hidden (show=False)."""
        return [channel for channel in self._channels.values() if not channel.show]
    
    def set_channel_visibility(self, channel_id: str, visible: bool) -> bool:
        """Set visibility of a channel."""
        channel = self.get_channel(channel_id)
        if channel:
            channel.show = visible
            return True
        return False
    
    def set_file_channels_visibility(self, file_id: str, visible: bool) -> int:
        """Set visibility for all channels from a file."""
        channels = self.get_channels_by_file(file_id)
        count = 0
        for channel in channels:
            channel.show = visible
            count += 1
        return count
    
    def clear_all(self):
        """Remove all channels from manager."""
        self._channels.clear()
        self._channels_by_file.clear()
        self._channels_by_type.clear()
        self._channels_by_step.clear()
        self._creation_order.clear()
        self._children.clear()
        self._parents.clear()
        self._stats['total_channels'] = 0
    
    def get_stats(self) -> Dict:
        """Get manager statistics."""
        return {
            **self._stats,
            'channels_by_file': {k: len(v) for k, v in self._channels_by_file.items()},
            'channels_by_type': {k.value if hasattr(k, 'value') else str(k): len(v) 
                               for k, v in self._channels_by_type.items()},
            'channels_by_step': {k: len(v) for k, v in self._channels_by_step.items()},
            'total_tracked': len(self._channels)
        }
    
    def get_recent_channels(self, limit: int = 10) -> List[Channel]:
        """Get recently added channels."""
        recent_ids = self._creation_order[-limit:] if limit > 0 else self._creation_order
        return [self._channels[cid] for cid in reversed(recent_ids) if cid in self._channels]
    
    def get_memory_usage_summary(self) -> Dict:
        """Get memory usage summary across all channels."""
        total_bytes = 0
        xdata_bytes = 0
        ydata_bytes = 0
        
        for channel in self._channels.values():
            usage = channel.get_memory_usage()
            total_bytes += usage.get('total_est_bytes', 0)
            xdata_bytes += usage.get('xdata_bytes', 0)
            ydata_bytes += usage.get('ydata_bytes', 0)
        
        return {
            'total_channels': len(self._channels),
            'total_memory_bytes': total_bytes,
            'xdata_memory_bytes': xdata_bytes,
            'ydata_memory_bytes': ydata_bytes,
            'avg_memory_per_channel': total_bytes / len(self._channels) if self._channels else 0
        }
    
    def __len__(self) -> int:
        """Return number of channels managed."""
        return len(self._channels)
    
    def __contains__(self, channel_id: str) -> bool:
        """Check if channel_id is managed."""
        return channel_id in self._channels
    
    def __iter__(self):
        """Iterate over Channel objects."""
        return iter(self._channels.values()) 