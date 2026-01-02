"""
Phase 1a: Structure Discovery

Maps the Azure Blob Storage container structure hierarchically without
listing all blobs. Discovers borders, directions, years, months, and days.
"""

import logging
from typing import Dict, List, Any
from tqdm import tqdm

from .base import PipelinePhase
from .config import PipelineConfig, AzureConfig
from .azure_client import AzureBlobClient
from .utils import save_json

logger = logging.getLogger(__name__)


class StructureDiscoveryPhase(PipelinePhase):
    """
    Phase 1a: Discover the Azure Blob Storage container structure.

    Maps the hierarchical structure of the container to understand what
    cameras, years, months, and days are available without listing all 18M blobs.
    """

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        azure_config: AzureConfig
    ):
        """
        Initialize structure discovery phase.

        Args:
            pipeline_config: Pipeline configuration
            azure_config: Azure configuration for blob access
        """
        super().__init__(
            config=pipeline_config,
            phase_name="discover",
            description="Discovering Azure Blob Storage structure"
        )

        self.azure_config = azure_config
        self.azure_client: AzureBlobClient = None

    def discover_structure(self) -> Dict[str, Any]:
        """
        Discover the complete container structure hierarchically.

        Hierarchy:
            Borders → Directions → Years → Months → Days

        Returns:
            Inventory dictionary mapping camera_id → year → month → [days]
        """
        inventory = {}

        # Initialize Azure client
        self.azure_client = AzureBlobClient(self.azure_config)

        try:
            # Level 1: Discover borders (e.g., "GRADINA/", "KELEBIJA/")
            self.logger.info("Discovering borders...")
            borders = self.azure_client.list_prefixes(delimiter="/")
            self.logger.info(f"Found {len(borders)} borders")

            # Progress bar for borders
            for border in tqdm(borders, desc="Processing borders"):
                border_name = border.rstrip("/")

                # Level 2: Discover directions (e.g., "GRADINA/U/", "GRADINA/I/")
                directions = self.azure_client.list_prefixes(
                    prefix=border,
                    delimiter="/"
                )

                for direction in directions:
                    direction_name = direction.split("/")[-2]  # Extract "U" or "I"
                    camera_id = f"{border_name}_{direction_name}"

                    self.logger.debug(f"Processing camera: {camera_id}")

                    # Initialize camera structure
                    inventory[camera_id] = {}

                    # Level 3: Discover years
                    years = self.azure_client.list_prefixes(
                        prefix=direction,
                        delimiter="/"
                    )

                    for year in years:
                        year_name = year.split("/")[-2]  # Extract year

                        # Initialize year structure
                        inventory[camera_id][year_name] = {}

                        # Level 4: Discover months
                        months = self.azure_client.list_prefixes(
                            prefix=year,
                            delimiter="/"
                        )

                        for month in months:
                            month_name = month.split("/")[-2]  # Extract month

                            # Level 5: Discover days
                            days = self.azure_client.list_prefixes(
                                prefix=month,
                                delimiter="/"
                            )

                            # Extract day names
                            day_names = [day.split("/")[-2] for day in days]

                            # Store days for this month
                            inventory[camera_id][year_name][month_name] = day_names

                            self.logger.debug(
                                f"  {camera_id}/{year_name}/{month_name}: "
                                f"{len(day_names)} days"
                            )

            self.logger.info(
                f"Discovery complete. Found {len(inventory)} cameras"
            )

            return inventory

        finally:
            # Clean up Azure client
            if self.azure_client:
                self.azure_client.close()

    def run(self, resume: bool = False) -> Dict[str, Any]:
        """
        Execute structure discovery phase.

        Args:
            resume: If True, skip discovery if inventory already exists

        Returns:
            Dictionary with inventory and statistics
        """
        inventory_file = self.config.get_path(self.config.inventory_file)

        # Check if we can resume
        if resume and inventory_file.exists():
            self.logger.info(f"Inventory already exists at {inventory_file}")
            from .utils import load_json
            inventory = load_json(inventory_file)

            if inventory:
                self.logger.info("Resuming with existing inventory")
                return {
                    "inventory": inventory,
                    "resumed": True
                }

        # Discover structure
        self.logger.info("Starting structure discovery...")
        inventory = self.discover_structure()

        # Calculate statistics
        total_cameras = len(inventory)
        total_months = sum(
            len(years)
            for camera in inventory.values()
            for years in camera.values()
        )
        total_days = sum(
            len(days)
            for camera in inventory.values()
            for year in camera.values()
            for days in year.values()
        )

        # Save inventory
        self.logger.info(f"Saving inventory to {inventory_file}")
        save_json(inventory, inventory_file)

        result = {
            "inventory": inventory,
            "statistics": {
                "cameras": total_cameras,
                "months": total_months,
                "days": total_days
            },
            "resumed": False
        }

        self.logger.info(
            f"Inventory saved: {total_cameras} cameras, "
            f"{total_months} camera-months, {total_days} camera-days"
        )

        return result

    def validate(self) -> bool:
        """
        Validate that structure discovery completed successfully.

        Returns:
            True if inventory file exists and contains data
        """
        inventory_file = self.config.get_path(self.config.inventory_file)

        if not inventory_file.exists():
            self.logger.error(f"Inventory file not found: {inventory_file}")
            return False

        from .utils import load_json
        inventory = load_json(inventory_file)

        if not inventory:
            self.logger.error("Inventory file is empty")
            return False

        # Check that we have reasonable number of cameras
        num_cameras = len(inventory)
        if num_cameras < 10:
            self.logger.warning(
                f"Only {num_cameras} cameras found, expected ~32"
            )
            return False

        self.logger.info(f"Validation passed: {num_cameras} cameras in inventory")
        return True


def main():
    """
    CLI entry point for Phase 1a: Structure Discovery.

    Usage:
        python -m dataset_pipeline.phase1_discover [--resume]
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Phase 1a: Discover Azure Blob Storage structure"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing inventory if available"
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only run validation"
    )

    args = parser.parse_args()

    # Load configurations
    pipeline_config = PipelineConfig()
    azure_config = AzureConfig.from_env()

    # Ensure directories exist
    pipeline_config.ensure_directories()

    # Create phase
    phase = StructureDiscoveryPhase(pipeline_config, azure_config)

    if args.validate_only:
        # Only run validation
        print("Running validation...")
        if phase.validate():
            print("✓ Validation passed")
            exit(0)
        else:
            print("✗ Validation failed")
            exit(1)
    else:
        # Execute phase
        result = phase.execute(resume=args.resume, validate_after=True)

        # Print results
        print("\n" + "=" * 60)
        print("Phase 1a: Structure Discovery - Complete")
        print("=" * 60)

        if result["status"] == "completed":
            data = result["data"]
            stats = data.get("statistics", {})

            print(f"Status: ✓ Success")
            print(f"Duration: {result['duration_seconds']:.1f} seconds")
            print(f"\nStatistics:")
            print(f"  Cameras: {stats.get('cameras', 0)}")
            print(f"  Camera-Months: {stats.get('months', 0)}")
            print(f"  Camera-Days: {stats.get('days', 0)}")

            if data.get("resumed"):
                print(f"\n(Resumed from existing inventory)")

            print("=" * 60)
            exit(0)
        else:
            print(f"Status: ✗ Failed")
            print(f"Reason: {result.get('reason', 'Unknown')}")
            print("=" * 60)
            exit(1)


if __name__ == "__main__":
    main()
