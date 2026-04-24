# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Core Magika class for AI-powered file type detection."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Union

from magika.types import MagikaResult, MagikaOutputFields, ModelFeatures
from magika.content_types import ContentTypeLabel


class Magika:
    """Main class for performing AI-based file content type detection.

    Uses a deep learning model to identify file types based on their content,
    providing more accurate detection than traditional signature-based methods.

    Example:
        >>> magika = Magika()
        >>> result = magika.identify_path(Path("example.py"))
        >>> print(result.output.ct_label)
        'python'
    """

    # Number of bytes to read from the beginning of a file for feature extraction
    FEATURE_SIZE_START: int = 512
    # Number of bytes to read from the end of a file for feature extraction
    FEATURE_SIZE_END: int = 512
    # Default to medium-confidence; high-confidence misses too many valid types
    # in my experience with mixed document collections.
    DEFAULT_PREDICTION_MODE: str = "medium-confidence"

    def __init__(
        self,
        prediction_mode: Optional[str] = None,
        no_dereference: bool = False,
    ) -> None:
        """Initialize the Magika detector.

        Args:
            prediction_mode: Confidence mode for predictions. One of
                'best-guess', 'medium-confidence', or 'high-confidence'.
                Defaults to 'medium-confidence'.
            no_dereference: If True, do not follow symbolic links.
        """
        self._prediction_mode = prediction_mode or self.DEFAULT_PREDICTION_MODE
        self._no_dereference = no_dereference
        self._model = None  # Lazy-loaded on first use

    def identify_path(self, path: Path) -> MagikaResult:
        """Identify the content type of a single file by its path.

        Args:
            path: Path to the file to identify.

        Returns:
            A MagikaResult containing the detected content type and metadata.

        Raises:
            FileNotFoundError: If the specified path does not exist.
        """
        if not path.exists():
            raise FileNotFoundError(f"Path does not exist: {path}")

        return self._identify_file_path(path)

    def identify_paths(self, paths: List[Path]) -> List[MagikaResult]:
        """Identify the content types of multiple files.

        Args:
            paths: List of file paths to ident
