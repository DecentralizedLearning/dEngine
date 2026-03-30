from typing import Optional, Dict
import re
from pathlib import Path
import yaml
import hashlib


def _string_to_color(s: str) -> str:
    md5_bytes = hashlib.md5(s.encode("utf-8")).digest()
    r, g, b = md5_bytes[:3]
    return f"#{r:02x}{g:02x}{b:02x}"


class RegexColorDict:
    def __init__(
        self,
        patterns: Optional[Dict[str, str]] = None,
        yaml_file_path: Optional[Path] = None,
        default: Optional[str] = None
    ):
        self._patterns = {}
        self.default = default

        if yaml_file_path:
            with yaml_file_path.open('r') as f:
                config = yaml.safe_load(f)
                self.default = config.get('default', self.default)
                patterns_from_yaml = config.get('patterns', {})
                self.extend(patterns_from_yaml)

        if patterns:
            self.extend(patterns)

    def extend(self, patterns: Dict[str, str]):
        for pattern, value in patterns.items():
            self.add(pattern, value)

    def add(self, pattern: str, value: str):
        compiled_pattern = re.compile(pattern)
        self._patterns[compiled_pattern] = value

    def get(self, key: str) -> str:
        for pattern, value in self._patterns.items():
            if pattern.match(key):
                return value
        if self.default:
            return self.default
        return _string_to_color(key)
