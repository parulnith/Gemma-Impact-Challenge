"""
Cache Manager Module
Handles intelligent caching of extraction results for performance optimization.
"""

import os
import pickle
import time
from typing import Dict, Any, Optional, List
from pathlib import Path


class CacheManager:
    """Manages caching of extraction results with intelligent storage and retrieval"""
    
    def __init__(self, cache_dir: str = "cache"):
        """
        Initialize cache manager
        
        Args:
            cache_dir (str): Directory to store cache files
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Separate cache files for different data types
        self.image_cache_file = self.cache_dir / "image_extractions.pkl"
        self.audio_cache_file = self.cache_dir / "audio_transcriptions.pkl"
        
        # In-memory caches for performance
        self._image_cache = self._load_cache(self.image_cache_file)
        self._audio_cache = self._load_cache(self.audio_cache_file)
    
    def _load_cache(self, cache_file: Path) -> Dict[str, Any]:
        """
        Load cache from disk
        
        Args:
            cache_file (Path): Path to cache file
            
        Returns:
            dict: Loaded cache data or empty dict
        """
        if cache_file.exists():
            try:
                with open(cache_file, "rb") as f:
                    cache_data = pickle.load(f)
                print(f"[CacheManager] Loaded {len(cache_data)} entries from {cache_file.name}")
                return cache_data
            except Exception as e:
                print(f"[CacheManager] Failed to load cache from {cache_file}: {e}")
        
        return {}
    
    def _save_cache(self, cache_data: Dict[str, Any], cache_file: Path) -> bool:
        """
        Save cache to disk
        
        Args:
            cache_data (dict): Cache data to save
            cache_file (Path): Path to cache file
            
        Returns:
            bool: Success status
        """
        try:
            with open(cache_file, "wb") as f:
                pickle.dump(cache_data, f)
            return True
        except Exception as e:
            print(f"[CacheManager] Failed to save cache to {cache_file}: {e}")
            return False
    
    def get_image_extraction(self, image_hash: str) -> Optional[List[str]]:
        """
        Retrieve cached image extraction results
        
        Args:
            image_hash (str): MD5 hash of the image
            
        Returns:
            list or None: Cached field names or None if not found
        """
        if image_hash in self._image_cache:
            cache_entry = self._image_cache[image_hash]
            
            # Handle both old format (direct list) and new format (dict with metadata)
            if isinstance(cache_entry, list):
                return cache_entry
            elif isinstance(cache_entry, dict) and 'fields' in cache_entry:
                return cache_entry['fields']
        
        return None
    
    def cache_image_extraction(self, image_hash: str, fields: List[str], 
                             metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Cache image extraction results
        
        Args:
            image_hash (str): MD5 hash of the image
            fields (list): Extracted field names
            metadata (dict, optional): Additional metadata
            
        Returns:
            bool: Success status
        """
        cache_entry = {
            'fields': fields,
            'timestamp': time.time(),
            'field_count': len(fields)
        }
        
        if metadata:
            cache_entry['metadata'] = metadata
        
        self._image_cache[image_hash] = cache_entry
        return self._save_cache(self._image_cache, self.image_cache_file)
    
    def get_audio_transcription(self, audio_hash: str) -> Optional[str]:
        """
        Retrieve cached audio transcription
        
        Args:
            audio_hash (str): Hash of the audio data
            
        Returns:
            str or None: Cached transcription or None if not found
        """
        if audio_hash in self._audio_cache:
            cache_entry = self._audio_cache[audio_hash]
            
            # Handle both old format (direct string) and new format (dict)
            if isinstance(cache_entry, str):
                return cache_entry
            elif isinstance(cache_entry, dict) and 'transcription' in cache_entry:
                return cache_entry['transcription']
        
        return None
    
    def cache_audio_transcription(self, audio_hash: str, transcription: str,
                                metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Cache audio transcription results
        
        Args:
            audio_hash (str): Hash of the audio data
            transcription (str): Transcribed text
            metadata (dict, optional): Additional metadata
            
        Returns:
            bool: Success status
        """
        cache_entry = {
            'transcription': transcription,
            'timestamp': time.time(),
            'length': len(transcription)
        }
        
        if metadata:
            cache_entry['metadata'] = metadata
        
        self._audio_cache[audio_hash] = cache_entry
        return self._save_cache(self._audio_cache, self.audio_cache_file)
    
    def clear_cache(self, cache_type: str = "all") -> bool:
        """
        Clear cache data
        
        Args:
            cache_type (str): Type of cache to clear ("image", "audio", or "all")
            
        Returns:
            bool: Success status
        """
        success = True
        
        if cache_type in ["image", "all"]:
            self._image_cache.clear()
            if self.image_cache_file.exists():
                try:
                    self.image_cache_file.unlink()
                except Exception as e:
                    print(f"[CacheManager] Failed to delete image cache file: {e}")
                    success = False
        
        if cache_type in ["audio", "all"]:
            self._audio_cache.clear()
            if self.audio_cache_file.exists():
                try:
                    self.audio_cache_file.unlink()
                except Exception as e:
                    print(f"[CacheManager] Failed to delete audio cache file: {e}")
                    success = False
        
        print(f"[CacheManager] Cleared {cache_type} cache")
        return success
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics
        
        Returns:
            dict: Cache statistics
        """
        return {
            'image_cache_entries': len(self._image_cache),
            'audio_cache_entries': len(self._audio_cache),
            'image_cache_file_exists': self.image_cache_file.exists(),
            'audio_cache_file_exists': self.audio_cache_file.exists(),
            'cache_directory': str(self.cache_dir)
        }
    
    def cleanup_old_entries(self, max_age_days: float = 30) -> int:
        """
        Remove cache entries older than specified age
        
        Args:
            max_age_days (float): Maximum age in days
            
        Returns:
            int: Number of entries removed
        """
        current_time = time.time()
        max_age_seconds = max_age_days * 24 * 3600
        removed_count = 0
        
        # Clean image cache
        old_image_keys = []
        for key, entry in self._image_cache.items():
            if isinstance(entry, dict) and 'timestamp' in entry:
                if current_time - entry['timestamp'] > max_age_seconds:
                    old_image_keys.append(key)
        
        for key in old_image_keys:
            del self._image_cache[key]
            removed_count += 1
        
        # Clean audio cache
        old_audio_keys = []
        for key, entry in self._audio_cache.items():
            if isinstance(entry, dict) and 'timestamp' in entry:
                if current_time - entry['timestamp'] > max_age_seconds:
                    old_audio_keys.append(key)
        
        for key in old_audio_keys:
            del self._audio_cache[key]
            removed_count += 1
        
        # Save updated caches
        if old_image_keys:
            self._save_cache(self._image_cache, self.image_cache_file)
        if old_audio_keys:
            self._save_cache(self._audio_cache, self.audio_cache_file)
        
        if removed_count > 0:
            print(f"[CacheManager] Cleaned up {removed_count} old cache entries")
        
        return removed_count