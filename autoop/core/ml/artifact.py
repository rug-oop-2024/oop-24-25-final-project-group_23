from pydantic import BaseModel, Field
import base64
from typing import Dict, List, Optional


class Artifact:

    def __init__(self, name: str, asset_path: str, data: bytes,
                 type: str, tags: Optional[List[str]] = None,
                 metadata: Optional[Dict] = None, version: str = "1.0.0"):
        self.name = name
        self.asset_path = asset_path
        self.data = data
        self.version = version
        self.type = type
        self.tags = tags
        self.metadata = metadata

    @property
    def id(self) -> str:
        """Derives the ID from the asset_path and version."""
        encoded_path = base64.b64encode(self.asset_path.encode()).decode()
        return f"{encoded_path}:{self.version}"

    def save(self) -> bytes:
        """Method to save the artifact's data."""
        return self.data

    def read(self) -> bytes:
        """Method to read the artifact's data."""
        return self.data
