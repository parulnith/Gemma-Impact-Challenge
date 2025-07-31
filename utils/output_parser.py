"""
Output Parser Module
Handles parsing of AI model outputs to extract structured field data.
"""

import re
import json
from typing import List, Dict, Any, Optional


class OutputParser:
    """Parses AI model outputs to extract form field names"""
    
    def __init__(self):
        """Initialize the output parser with regex patterns"""
        # Pattern to match numbered list items (1. Field Name, 2) Field Name, etc.)
        self.numbered_pattern = re.compile(r'^\s*(\d+)[\.|\)]?\s*(.+?)(?:\s*[:：](.*))?$')
        
        # Pattern to find JSON blocks in text
        self.json_pattern = re.compile(r'{.*}', re.DOTALL)
        
        # Filter words to skip in fallback extraction
        self.skip_words = {'extract', 'json', 'example', 'field', 'labels', 'form'}
    
    def extract_json_fields(self, text: str) -> List[str]:
        """
        Extract field names from JSON structure in AI output
        
        Args:
            text (str): Raw AI model output
            
        Returns:
            list: Extracted field names from JSON
        """
        json_match = self.json_pattern.search(text)
        if not json_match:
            return []
        
        json_str = json_match.group(0)
        
        try:
            parsed_json = json.loads(json_str)
            flat_json = self._flatten_json(parsed_json)
            fields = list(flat_json.keys())
            return [field.strip() for field in fields if field.strip()]
            
        except (json.JSONDecodeError, TypeError) as e:
            print(f"[OutputParser] JSON parsing failed: {e}")
            return []
    
    def _flatten_json(self, json_obj: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively flatten nested JSON structures
        
        Args:
            json_obj (dict): JSON object to flatten
            
        Returns:
            dict: Flattened JSON object
        """
        flat = {}
        
        for key, value in json_obj.items():
            if isinstance(value, dict):
                if "type" in value:
                    # Handle structured field definitions like {"name": {"type": "text"}}
                    flat[key] = value["type"]
                else:
                    # Recursively flatten nested objects
                    flat.update(self._flatten_json(value))
            else:
                flat[key] = value
                
        return flat
    
    def extract_fallback_fields(self, text: str) -> List[str]:
        """
        Extract field names using regex patterns as fallback method
        
        Args:
            text (str): Raw AI model output
            
        Returns:
            list: Extracted field names using pattern matching
        """
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        fields = []
        
        for line in lines:
            # Skip instructional text from AI
            if any(skip_word in line.lower() for skip_word in self.skip_words):
                continue
            
            field_name = self._extract_field_from_line(line)
            if field_name and self._is_valid_field_name(field_name):
                fields.append(field_name)
        
        return fields
    
    def _extract_field_from_line(self, line: str) -> Optional[str]:
        """
        Extract field name from a single line using various patterns
        
        Args:
            line (str): Single line of text
            
        Returns:
            str or None: Extracted field name or None
        """
        # Try numbered pattern first (1. Name, 2) Age, etc.)
        match = self.numbered_pattern.match(line)
        if match:
            raw_field = match.group(2).strip()
            return raw_field.rstrip(':：')
        
        # Try colon-separated pattern (Name: value)
        if ':' in line:
            return line.split(':', 1)[0].strip()
        
        # Try dash-separated pattern (- Name)
        if line.startswith('-'):
            return line[1:].strip().rstrip(':：')
        
        return None
    
    def _is_valid_field_name(self, field_name: str) -> bool:
        """
        Validate if extracted text is a reasonable field name
        
        Args:
            field_name (str): Candidate field name
            
        Returns:
            bool: True if valid field name
        """
        if not field_name:
            return False
        
        # Length check: reasonable field name length
        if not (2 < len(field_name) < 50):
            return False
        
        # Skip common instructional phrases
        lower_field = field_name.lower()
        invalid_phrases = {
            'here are', 'the following', 'extracted', 'fields are',
            'json object', 'form has', 'image contains'
        }
        
        if any(phrase in lower_field for phrase in invalid_phrases):
            return False
        
        return True
    
    def parse(self, text: str) -> Dict[str, Any]:
        """
        Main parsing method that tries JSON first, then fallback patterns
        
        Args:
            text (str): Raw AI model output
            
        Returns:
            dict: Parsing results with fields and metadata
        """
        # Try JSON extraction first
        json_fields = self.extract_json_fields(text)
        
        if json_fields:
            return {
                'fields': json_fields,
                'method': 'json',
                'success': True,
                'field_count': len(json_fields)
            }
        
        # Fallback to regex patterns
        fallback_fields = self.extract_fallback_fields(text)
        
        return {
            'fields': fallback_fields,
            'method': 'regex_fallback',
            'success': len(fallback_fields) > 0,
            'field_count': len(fallback_fields)
        }