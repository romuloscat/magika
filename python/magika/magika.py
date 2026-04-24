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
    # Minimum confidence threshold for model predictions
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
            paths: List of file paths to identify.

        Returns:
            A list of MagikaResult objects, one per input path.
        """
        return [self.identify_path(p) for p in paths]

    def identify_bytes(self, content: bytes) -> MagikaResult:
        """Identify the content type from raw bytes.

        Args:
            content: The raw bytes of the file content to identify.

        Returns:
            A MagikaResult containing the detected content type and metadata.
        """
        features = self._extract_features_from_bytes(content)
        return self._run_inference(features, path=None)

    def _identify_file_path(self, path: Path) -> MagikaResult:
        """Internal method to identify a file given a validated path."""
        if self._no_dereference and path.is_symlink():
            # Return a result indicating this is a symlink without following it
            return MagikaResult.make_symlink(path)

        try:
            content = path.read_bytes()
        except PermissionError:
            return MagikaResult.make_unreadable(path)

        features = self._extract_features_from_bytes(content)
        return self._run_inference(features, path=path)

    def _extract_features_from_bytes(self, content: bytes) -> ModelFeatures:
        """Extract model input features from raw bytes.

        Reads bytes from the beginning and end of the content to build
        the fixed-size feature vector expected by the model.

        Args:
            content: Raw file bytes.

        Returns:
            A ModelFeatures object with start/end byte sequences.
        """
        start_bytes = content[: self.FEATURE_SIZE_START]
        end_bytes = content[-self.FEATURE_SIZE_END :] if len(content) > self.FEATURE_SIZE_END else content

        return ModelFeatures(
            start=list(start_bytes),
            end=list(end_bytes),
            size=len(content),
        )

    def _run_inference(self, features: ModelFeatures, path: Optional[Path]) -> MagikaResult:
        """Run model inference on extracted features.

        Args:
            features: Extracted file features.
            path: Optional path for result metadata.

        Returns:
            A MagikaResult with the model's prediction.
        """
        # Lazy-load the model on first inference call
        if self._model is None:
            self._load_model()

        # Placeholder: actual model inference will be implemented with ONNX runtime
        ct_label = ContentTypeLabel.UNKNOWN
        score = 0.0

        output = MagikaOutputFields(
            ct_label=ct_label,
            mime_type="application/octet-stream",
            group="unknown",
            description="Unknown file type",
            extensions=[],
            is_text=False,
            score=score,
        )

        return MagikaResult(path=path, output=output)

    def _load_model(self) -> None:
        """Load the ONNX inference model from the bundled assets."""
        # Model loading will be implemented when model assets are added
        pass
