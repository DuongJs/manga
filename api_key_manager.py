import json
import os
from threading import Lock
from datetime import datetime


class APIKeyManager:
    """
    Manager for API keys with round-robin rotation support.
    Stores API keys in a JSON file for persistence.
    """
    
    def __init__(self, config_file="api_keys.json"):
        """
        Initialize API key manager.
        
        Args:
            config_file (str): Path to JSON file storing API keys
        """
        self.config_file = config_file
        self.lock = Lock()
        self.current_index = 0
        self.api_keys = []
        self.load_keys()
    
    def load_keys(self):
        """Load API keys from JSON file."""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    data = json.load(f)
                    self.api_keys = data.get('api_keys', [])
                    self.current_index = data.get('current_index', 0)
                print(f"Loaded {len(self.api_keys)} API key(s) from {self.config_file}")
            except json.JSONDecodeError as e:
                print(f"Error: {self.config_file} contains invalid JSON: {e}")
                print(f"Please check the file format. See api_keys.json.example for the correct format.")
                self.api_keys = []
                self.current_index = 0
            except Exception as e:
                print(f"Error loading API keys from {self.config_file}: {e}")
                self.api_keys = []
                self.current_index = 0
        else:
            print(f"No API key file found at {self.config_file}. API keys can be added via the web interface.")
            self.api_keys = []
            self.current_index = 0
    
    def save_keys(self):
        """
        Save API keys to JSON file.
        
        Raises:
            Exception: If there's an error writing to the file
        """
        data = {
            'api_keys': self.api_keys,
            'current_index': self.current_index,
            'last_updated': datetime.now().isoformat()
        }
        with open(self.config_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def add_key(self, api_key, name=None):
        """
        Add a new API key.
        
        Args:
            api_key (str): API key to add
            name (str): Optional name/description for the key
            
        Returns:
            bool: True if added successfully, False if key already exists or is empty
            
        Raises:
            Exception: If there's an error saving the key to file
        """
        with self.lock:
            if api_key and api_key not in [k['key'] for k in self.api_keys]:
                key_name = name or f"Key {len(self.api_keys) + 1}"
                self.api_keys.append({
                    'key': api_key,
                    'name': key_name,
                    'added_at': datetime.now().isoformat(),
                    'usage_count': 0
                })
                self.save_keys()  # This will raise an exception if save fails
                print(f"✓ API key '{key_name}' added successfully (key starts with: {api_key[:10]}...)")
                print(f"  Total keys in manager: {len(self.api_keys)}")
                return True
            elif api_key in [k['key'] for k in self.api_keys]:
                print(f"✗ API key already exists in manager")
                return False
            else:
                print(f"✗ Cannot add empty API key")
                return False
    
    def remove_key(self, index):
        """
        Remove an API key by index.
        
        Args:
            index (int): Index of the key to remove
            
        Returns:
            bool: True if removed successfully
        """
        with self.lock:
            if 0 <= index < len(self.api_keys):
                self.api_keys.pop(index)
                if self.current_index >= len(self.api_keys) and self.api_keys:
                    self.current_index = 0
                self.save_keys()
                return True
            return False
    
    def get_next_key(self):
        """
        Get the next API key using round-robin rotation.
        
        Returns:
            str: API key or None if no keys available
        """
        with self.lock:
            if not self.api_keys:
                print("No API keys available in the manager. Please add keys via the web interface or environment variable.")
                return None
            
            # Get current key
            key_info = self.api_keys[self.current_index]
            api_key = key_info['key']
            
            # Update usage count
            key_info['usage_count'] = key_info.get('usage_count', 0) + 1
            
            # Move to next key
            self.current_index = (self.current_index + 1) % len(self.api_keys)
            
            self.save_keys()
            return api_key
    
    def get_all_keys(self):
        """
        Get all API keys information.
        
        Returns:
            list: List of key information dictionaries
        """
        with self.lock:
            return [
                {
                    'index': i,
                    'name': k['name'],
                    'key': k['key'][:10] + '...' if len(k['key']) > 10 else k['key'],
                    'usage_count': k.get('usage_count', 0),
                    'added_at': k.get('added_at', 'Unknown')
                }
                for i, k in enumerate(self.api_keys)
            ]
    
    def get_stats(self):
        """
        Get statistics about API key usage.
        
        Returns:
            dict: Statistics dictionary
        """
        with self.lock:
            total_keys = len(self.api_keys)
            total_usage = sum(k.get('usage_count', 0) for k in self.api_keys)
            return {
                'total_keys': total_keys,
                'total_usage': total_usage,
                'current_key_index': self.current_index
            }
