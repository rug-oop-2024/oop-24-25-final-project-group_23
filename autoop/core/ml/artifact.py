from pydantic import BaseModel, Field
import base64
from typing import Any, Dict, List


class Artifact(BaseModel):
    name: str
    asset_path: str
    version: str
    data: bytes
    metadata: Dict[str, Any]
    type: str
    tags: List[str] = Field(default_factory=list)

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
