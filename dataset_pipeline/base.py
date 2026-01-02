"""
Base classes and interfaces for the dataset pipeline

Provides the abstract PipelinePhase class that all pipeline phases inherit from,
along with common functionality for logging, progress tracking, and execution.
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional
from datetime import datetime

from .config import PipelineConfig
from .utils import load_json, save_json, create_progress_tracker


# Configure logging
def setup_logging(
    log_file: Optional[Path] = None,
    level: int = logging.INFO
) -> logging.Logger:
    """
    Configure logging for the pipeline.

    Args:
        log_file: Optional path to log file (default: traffic_dataset/pipeline.log)
        level: Logging level (default: INFO)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger("dataset_pipeline")
    logger.setLevel(level)

    # Clear existing handlers
    logger.handlers = []

    # Format
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


class PipelinePhase(ABC):
    """
    Abstract base class for all pipeline phases.

    Each phase implements a specific step in the dataset creation pipeline,
    with support for resume capability, progress tracking, and error handling.
    """

    def __init__(
        self,
        config: PipelineConfig,
        phase_name: str,
        description: str
    ):
        """
        Initialize pipeline phase.

        Args:
            config: Pipeline configuration
            phase_name: Short name for this phase (e.g., "discover", "download")
            description: Human-readable description of what this phase does
        """
        self.config = config
        self.phase_name = phase_name
        self.description = description

        # Setup logging
        log_file = config.base_dir / "pipeline.log"
        self.logger = setup_logging(log_file=log_file)
        self.logger = logging.getLogger(f"dataset_pipeline.{phase_name}")

        # Progress tracking
        self.progress_file = config.base_dir / f"{phase_name}_progress.json"
        self.progress: Dict[str, Any] = {}

    def load_progress(self) -> Dict[str, Any]:
        """
        Load progress from file if it exists.

        Returns:
            Progress dictionary, or empty dict if no progress file exists
        """
        self.progress = load_json(self.progress_file, default={})
        return self.progress

    def save_progress(self) -> bool:
        """
        Save current progress to file.

        Returns:
            True if successful, False otherwise
        """
        self.progress["updated_at"] = datetime.now().isoformat()
        return save_json(self.progress, self.progress_file)

    def initialize_progress(
        self,
        total: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Initialize or reset progress tracking.

        Args:
            total: Total number of items to process (optional)
            **kwargs: Additional progress fields

        Returns:
            Initialized progress dictionary
        """
        if total is not None:
            self.progress = create_progress_tracker(total, self.description)
        else:
            self.progress = {
                "description": self.description,
                "started_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            }

        # Add any additional fields
        self.progress.update(kwargs)

        self.save_progress()
        return self.progress

    def can_resume(self) -> bool:
        """
        Check if this phase can be resumed from previous execution.

        Returns:
            True if progress file exists and phase can resume
        """
        return self.progress_file.exists()

    @abstractmethod
    def run(self, resume: bool = False) -> Any:
        """
        Execute this pipeline phase.

        Args:
            resume: If True, resume from previous progress (if available)

        Returns:
            Phase-specific output data

        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError(f"{self.__class__.__name__}.run() must be implemented")

    @abstractmethod
    def validate(self) -> bool:
        """
        Validate that this phase completed successfully.

        Returns:
            True if validation passed, False otherwise

        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError(f"{self.__class__.__name__}.validate() must be implemented")

    def get_status(self) -> Dict[str, Any]:
        """
        Get current status of this phase.

        Returns:
            Status dictionary with progress information
        """
        # Load latest progress
        self.load_progress()

        status = {
            "phase": self.phase_name,
            "description": self.description,
            "can_resume": self.can_resume(),
            "progress": self.progress
        }

        return status

    def execute(
        self,
        resume: bool = False,
        validate_after: bool = True
    ) -> Dict[str, Any]:
        """
        Execute the phase with error handling and validation.

        Args:
            resume: If True, attempt to resume from previous progress
            validate_after: If True, run validation after execution

        Returns:
            Execution result with status and data

        Raises:
            Exception: If phase execution fails
        """
        start_time = datetime.now()

        try:
            # Check resume capability
            if resume and not self.can_resume():
                self.logger.warning(
                    f"Resume requested but no progress found for {self.phase_name}"
                )
                resume = False

            # Log start
            if resume:
                self.logger.info(f"Resuming phase: {self.phase_name}")
            else:
                self.logger.info(f"Starting phase: {self.phase_name}")

            # Execute phase
            result = self.run(resume=resume)

            # Validate if requested
            if validate_after:
                self.logger.info(f"Validating phase: {self.phase_name}")
                validation_passed = self.validate()

                if not validation_passed:
                    self.logger.error(f"Validation failed for phase: {self.phase_name}")
                    return {
                        "phase": self.phase_name,
                        "status": "failed",
                        "reason": "validation_failed",
                        "data": result
                    }

            # Calculate duration
            duration = (datetime.now() - start_time).total_seconds()

            self.logger.info(
                f"Completed phase: {self.phase_name} in {duration:.1f} seconds"
            )

            return {
                "phase": self.phase_name,
                "status": "completed",
                "duration_seconds": duration,
                "data": result
            }

        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            self.logger.error(
                f"Phase {self.phase_name} failed after {duration:.1f} seconds: {e}",
                exc_info=True
            )

            return {
                "phase": self.phase_name,
                "status": "failed",
                "reason": str(e),
                "duration_seconds": duration
            }
