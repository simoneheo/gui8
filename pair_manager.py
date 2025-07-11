from typing import Dict, List, Optional, Set, Union
from collections import defaultdict
from pair import Pair, AlignmentMethod
import time
from datetime import datetime


class PairManager:
    """
    Central manager for Pair objects with efficient lookup and management capabilities.
    """
    
    def __init__(self, max_pairs=100):
        # Core storage
        self._pairs: Dict[str, Pair] = {}
        self._pairs_by_order: List[str] = []
        
        # Capacity management
        self.max_pairs = max_pairs
        self.warning_threshold = int(max_pairs * 0.8)  # 80% threshold
        
        # Lookup indices for efficient queries
        self._pairs_by_file: Dict[str, Set[str]] = defaultdict(set)
        self._pairs_by_channel: Dict[str, Set[str]] = defaultdict(set)
        
        # Stats tracking
        self._stats = {
            'total_pairs': 0,
            'visible_pairs': 0,
            'hidden_pairs': 0,
            'created_at': datetime.now()
        }
        
        # Color palette for unique pair styling
        self._color_palette = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
            '#a6cee3', '#fb9a99', '#fdbf6f', '#cab2d6', '#ff9896',
            '#b15928', '#fdb462', '#b3de69', '#fccde5', '#d9d9d9'
        ]
        
        # Marker types for variety
        self._marker_types = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', '+', 'x']
        
        print(f"[PairManager] Initialized with max capacity: {max_pairs} pairs")
    
    def _get_used_colors(self) -> Set[str]:
        """Get set of colors currently used by existing pairs"""
        return {pair.color for pair in self._pairs.values() if pair.color}
    
    def _get_used_marker_types(self) -> Set[str]:
        """Get set of marker types currently used by existing pairs"""
        return {pair.marker_type for pair in self._pairs.values() if pair.marker_type}
    
    def _assign_unique_styling(self, pair: Pair) -> None:
        """
        Assign unique color and marker type to a pair.
        Ensures no conflicts with existing pairs.
        """
        # Get currently used colors and markers
        used_colors = self._get_used_colors()
        used_markers = self._get_used_marker_types()
        
        # Find next available color
        available_colors = [color for color in self._color_palette if color not in used_colors]
        if available_colors:
            pair.color = available_colors[0]
        else:
            # If all colors are used, cycle back to the beginning
            pair.color = self._color_palette[len(self._pairs) % len(self._color_palette)]
        
        # Find next available marker type
        available_markers = [marker for marker in self._marker_types if marker not in used_markers]
        if available_markers:
            pair.marker_type = available_markers[0]
        else:
            # If all markers are used, cycle back to the beginning
            pair.marker_type = self._marker_types[len(self._pairs) % len(self._marker_types)]
        
        # Set default marker color to match the assigned color
        # Map hex colors to display names for UI consistency
        color_to_display = {
            '#1f77b4': '🔵 Blue', '#ff7f0e': '🟠 Orange', '#2ca02c': '🟢 Green',
            '#d62728': '🔴 Red', '#9467bd': '🟣 Purple', '#8c564b': '🟤 Brown',
            '#e377c2': '🩷 Pink', '#7f7f7f': '⚫ Gray', '#bcbd22': '🟡 Yellow',
            '#17becf': '🔶 Cyan', '#a6cee3': '🔵 Light Blue', '#fb9a99': '🩷 Light Pink',
            '#fdbf6f': '🟡 Light Orange', '#cab2d6': '🟣 Light Purple', '#ff9896': '🩷 Light Red',
            '#b15928': '🟤 Dark Brown', '#fdb462': '🟠 Light Orange', '#b3de69': '🟢 Light Green',
            '#fccde5': '🩷 Very Light Pink', '#d9d9d9': '⚫ Light Gray'
        }
        pair.marker_color = color_to_display.get(pair.color, '🔵 Blue')
        
        print(f"[PairManager] Assigned unique styling to {pair.name}: color={pair.color}, marker={pair.marker_type}")
    
    def _is_duplicate(self, new_pair: Pair, existing_pair: Pair) -> bool:
        """
        Check if two pairs are duplicates based on channel and file IDs.
        
        Args:
            new_pair: New pair to check
            existing_pair: Existing pair to compare against
            
        Returns:
            bool: True if pairs are duplicates, False otherwise
        """
        # Exact match check
        exact_match = (new_pair.ref_channel_id == existing_pair.ref_channel_id and
                      new_pair.test_channel_id == existing_pair.test_channel_id and
                      new_pair.ref_file_id == existing_pair.ref_file_id and
                      new_pair.test_file_id == existing_pair.test_file_id)
        
        if exact_match:
            return True
            
        # Reverse match check (swapped channels)
        reverse_match = (new_pair.ref_channel_id == existing_pair.test_channel_id and
                        new_pair.test_channel_id == existing_pair.ref_channel_id and
                        new_pair.ref_file_id == existing_pair.test_file_id and
                        new_pair.test_file_id == existing_pair.ref_file_id)
        
        return reverse_match
    
    def _check_capacity(self) -> tuple[bool, str]:
        """
        Check if adding a new pair would exceed capacity limits.
        
        Returns:
            tuple: (can_add, message)
        """
        current_count = len(self._pairs)
        
        if current_count >= self.max_pairs:
            return False, f"Maximum pairs ({self.max_pairs}) reached"
        
        if current_count >= self.warning_threshold:
            return True, f"WARNING: {current_count}/{self.max_pairs} pairs used"
        
        return True, ""
    
    def add_pair(self, pair: Pair) -> tuple[bool, str]:
        """
        Add a new pair to the manager with validation, duplicate checking, and capacity management.
        
        Args:
            pair: Pair object to add
            
        Returns:
            tuple: (success, message) - True if added successfully, False with error message otherwise
        """
        # Check capacity
        can_add, capacity_msg = self._check_capacity()
        if not can_add:
            error_msg = f"Cannot add pair {pair.name}: {capacity_msg}"
            print(f"[PairManager] ERROR: {error_msg}")
            return False, error_msg
        
        if capacity_msg:
            print(f"[PairManager] {capacity_msg}")
        
        # Validate pair
        errors = pair.validate()
        if errors:
            error_msg = f"Cannot add pair {pair.name}: {', '.join(errors)}"
            print(f"[PairManager] ERROR: {error_msg}")
            return False, error_msg
        
        # Check for duplicates and block if found
        duplicate_pair = None
        for existing_pair in self._pairs.values():
            if self._is_duplicate(pair, existing_pair):
                duplicate_pair = existing_pair
                break
        
        if duplicate_pair:
            error_msg = f"Duplicate pair detected: '{pair.name}' conflicts with existing pair '{duplicate_pair.name}'. Use the delete icon to remove the existing pair first."
            print(f"[PairManager] BLOCKED: {error_msg}")
            return False, error_msg
        
        # Assign unique styling to the new pair
        self._assign_unique_styling(pair)
        
        # Add to storage
        self._pairs[pair.pair_id] = pair
        self._pairs_by_order.append(pair.pair_id)
        
        # Update indices
        if pair.ref_file_id:
            self._pairs_by_file[pair.ref_file_id].add(pair.pair_id)
        if pair.test_file_id:
            self._pairs_by_file[pair.test_file_id].add(pair.pair_id)
        
        if pair.ref_channel_id:
            self._pairs_by_channel[pair.ref_channel_id].add(pair.pair_id)
        if pair.test_channel_id:
            self._pairs_by_channel[pair.test_channel_id].add(pair.pair_id)
        
        # Update stats
        self._stats['total_pairs'] += 1
        if pair.show:
            self._stats['visible_pairs'] += 1
        else:
            self._stats['hidden_pairs'] += 1
        
        success_msg = f"Successfully added new pair: {pair.name}"
        print(f"[PairManager] {success_msg}")
        return True, success_msg
    
    def is_duplicate_pair(self, ref_channel_id: str, test_channel_id: str, 
                         ref_file_id: str, test_file_id: str) -> tuple[bool, str]:
        """
        Check if a pair with the given channel and file IDs would be a duplicate.
        This is used for early duplicate detection before expensive computations.
        
        Args:
            ref_channel_id: Reference channel ID
            test_channel_id: Test channel ID  
            ref_file_id: Reference file ID
            test_file_id: Test file ID
            
        Returns:
            tuple: (is_duplicate, existing_pair_name) - True if duplicate with name of existing pair
        """
        for existing_pair in self._pairs.values():
            # Exact match check
            exact_match = (ref_channel_id == existing_pair.ref_channel_id and
                          test_channel_id == existing_pair.test_channel_id and
                          ref_file_id == existing_pair.ref_file_id and
                          test_file_id == existing_pair.test_file_id)
            
            if exact_match:
                return True, existing_pair.name
                
            # Reverse match check (swapped channels)
            reverse_match = (ref_channel_id == existing_pair.test_channel_id and
                            test_channel_id == existing_pair.ref_channel_id and
                            ref_file_id == existing_pair.test_file_id and
                            test_file_id == existing_pair.ref_file_id)
            
            if reverse_match:
                return True, existing_pair.name
        
        return False, ""
    
    def remove_pair(self, pair_id: str) -> bool:
        """
        Remove a pair from the manager.
        
        Args:
            pair_id: ID of the pair to remove
            
        Returns:
            bool: True if removed successfully, False if pair not found
        """
        if pair_id not in self._pairs:
            return False
        
        pair = self._pairs[pair_id]
        
        # Remove from storage
        del self._pairs[pair_id]
        self._pairs_by_order.remove(pair_id)
        
        # Remove from indices
        if pair.ref_file_id:
            self._pairs_by_file[pair.ref_file_id].discard(pair_id)
        if pair.test_file_id:
            self._pairs_by_file[pair.test_file_id].discard(pair_id)
        
        if pair.ref_channel_id:
            self._pairs_by_channel[pair.ref_channel_id].discard(pair_id)
        if pair.test_channel_id:
            self._pairs_by_channel[pair.test_channel_id].discard(pair_id)
        
        # Update stats
        self._stats['total_pairs'] -= 1
        if pair.show:
            self._stats['visible_pairs'] -= 1
        else:
            self._stats['hidden_pairs'] -= 1
        
        print(f"[PairManager] Removed pair {pair_id}")
        return True
    
    
    def get_visible_pairs(self) -> List[Pair]:
        """Get all visible pairs"""
        return [pair for pair in self._pairs.values() if pair.show]
    
    def get_hidden_pairs(self) -> List[Pair]:
        """Get all hidden pairs"""
        return [pair for pair in self._pairs.values() if not pair.show]
    
    def get_all_pairs(self) -> List[Pair]:
        """Get all pairs in the order they were added"""
        return [self._pairs[pair_id] for pair_id in self._pairs_by_order if pair_id in self._pairs]
    
    def set_pair_visibility(self, pair_id: str, visible: bool) -> bool:
        """
        Set visibility of a pair and update statistics.
        
        Args:
            pair_id: ID of the pair to update
            visible: Whether the pair should be visible
            
        Returns:
            bool: True if pair was found and updated, False otherwise
        """
        if pair_id not in self._pairs:
            return False
        
        pair = self._pairs[pair_id]
        old_visible = pair.show
        pair.show = visible
        
        # Update statistics
        if old_visible != visible:
            if visible:
                self._stats['visible_pairs'] += 1
                self._stats['hidden_pairs'] -= 1
            else:
                self._stats['visible_pairs'] -= 1
                self._stats['hidden_pairs'] += 1
        
        print(f"[PairManager] Set pair {pair_id} visibility to {visible}")
        return True
    
    def get_capacity_info(self) -> dict:
        """
        Get current capacity information.
        
        Returns:
            dict: Capacity information including current count, max, and status
        """
        current_count = len(self._pairs)
        return {
            'current_count': current_count,
            'max_pairs': self.max_pairs,
            'warning_threshold': self.warning_threshold,
            'available_slots': self.max_pairs - current_count,
            'usage_percentage': (current_count / self.max_pairs) * 100,
            'at_warning_level': current_count >= self.warning_threshold,
            'at_capacity': current_count >= self.max_pairs
        }
    
    def get_stats(self) -> dict:
        """
        Get comprehensive statistics about the pair manager.
        
        Returns:
            dict: Statistics including capacity, counts, and performance info
        """
        capacity_info = self.get_capacity_info()
        return {
            **self._stats,
            **capacity_info,
            'pairs_by_file': {file_id: len(pairs) for file_id, pairs in self._pairs_by_file.items()},
            'pairs_by_channel': {channel_id: len(pairs) for channel_id, pairs in self._pairs_by_channel.items()}
        }
    
    
    