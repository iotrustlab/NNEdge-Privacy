#!/usr/bin/env python3
"""
Backup and Cleanup System for NNEdge-Privacy Experiments

This script provides comprehensive backup, cleanup, and organization functionality
for experiment results and model checkpoints to prevent overwriting and ensure
efficient storage management.
"""

import os
import shutil
import json
import time
from datetime import datetime
from pathlib import Path
import argparse
import logging
from typing import Dict, List, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('backup_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ExperimentBackupManager:
    """Manages backup, cleanup, and organization of experiment results and checkpoints."""
    
    def __init__(self, base_dir: str = "."):
        self.base_dir = Path(base_dir)
        self.results_dir = self.base_dir / "results"
        self.checkpoints_dir = self.base_dir / "checkpoints"
        self.backups_dir = self.base_dir / "backups"
        
        # Create directories if they don't exist
        self.backups_dir.mkdir(exist_ok=True)
        
        # Configuration
        self.max_backups_per_experiment = 5
        self.max_checkpoint_age_days = 30
        self.results_to_preserve = [
            "STM", "MMBind", "minigrid_v3", "experiments"
        ]
    
    def create_backup(self, experiment_name: str, backup_type: str = "auto") -> str:
        """
        Create a backup of experiment results and checkpoints.
        
        Args:
            experiment_name: Name of the experiment to backup
            backup_type: Type of backup ("auto", "manual", "critical")
            
        Returns:
            Path to the created backup
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{experiment_name}_{backup_type}_{timestamp}"
        backup_path = self.backups_dir / backup_name
        
        try:
            backup_path.mkdir(exist_ok=True)
            
            # Backup results
            results_source = self.results_dir / experiment_name
            if results_source.exists():
                results_backup = backup_path / "results"
                shutil.copytree(results_source, results_backup)
                logger.info(f"Backed up results from {results_source} to {results_backup}")
            
            # Backup checkpoints
            checkpoints_source = self.checkpoints_dir / experiment_name
            if checkpoints_source.exists():
                checkpoints_backup = backup_path / "checkpoints"
                shutil.copytree(checkpoints_source, checkpoints_backup)
                logger.info(f"Backed up checkpoints from {checkpoints_source} to {checkpoints_backup}")
            
            # Create backup metadata
            metadata = {
                "experiment_name": experiment_name,
                "backup_type": backup_type,
                "timestamp": timestamp,
                "created_at": datetime.now().isoformat(),
                "source_results": str(results_source) if results_source.exists() else None,
                "source_checkpoints": str(checkpoints_source) if checkpoints_source.exists() else None,
                "backup_size_mb": self._get_directory_size(backup_path)
            }
            
            with open(backup_path / "backup_metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Created backup: {backup_name}")
            return str(backup_path)
            
        except Exception as e:
            logger.error(f"Failed to create backup {backup_name}: {e}")
            if backup_path.exists():
                shutil.rmtree(backup_path)
            raise
    
    def cleanup_old_backups(self, max_age_days: int = 30) -> int:
        """
        Clean up old backups to save disk space.
        
        Args:
            max_age_days: Maximum age of backups to keep
            
        Returns:
            Number of backups removed
        """
        removed_count = 0
        current_time = time.time()
        max_age_seconds = max_age_days * 24 * 3600
        
        for backup_dir in self.backups_dir.iterdir():
            if not backup_dir.is_dir():
                continue
                
            # Check if it's a backup directory
            if not any(backup_dir.name.startswith(exp) for exp in self.results_to_preserve):
                continue
            
            # Check age
            backup_time = backup_dir.stat().st_mtime
            if current_time - backup_time > max_age_seconds:
                try:
                    shutil.rmtree(backup_dir)
                    logger.info(f"Removed old backup: {backup_dir.name}")
                    removed_count += 1
                except Exception as e:
                    logger.error(f"Failed to remove backup {backup_dir.name}: {e}")
        
        return removed_count
    
    def cleanup_old_checkpoints(self, max_age_days: int = 30) -> int:
        """
        Clean up old model checkpoints to save disk space.
        
        Args:
            max_age_days: Maximum age of checkpoints to keep
            
        Returns:
            Number of checkpoint directories removed
        """
        removed_count = 0
        current_time = time.time()
        max_age_seconds = max_age_days * 24 * 3600
        
        for checkpoint_dir in self.checkpoints_dir.iterdir():
            if not checkpoint_dir.is_dir():
                continue
            
            # Check age
            checkpoint_time = checkpoint_dir.stat().st_mtime
            if current_time - checkpoint_time > max_age_seconds:
                try:
                    shutil.rmtree(checkpoint_dir)
                    logger.info(f"Removed old checkpoint: {checkpoint_dir.name}")
                    removed_count += 1
                except Exception as e:
                    logger.error(f"Failed to remove checkpoint {checkpoint_dir.name}: {e}")
        
        return removed_count
    
    def organize_results(self) -> Dict[str, int]:
        """
        Organize results directory by moving old experiments to backups.
        
        Returns:
            Dictionary with organization statistics
        """
        stats = {"moved": 0, "cleaned": 0, "preserved": 0}
        
        for item in self.results_dir.iterdir():
            if not item.is_dir() or item.name.startswith('.'):
                continue
            
            # Skip preserved directories
            if item.name in self.results_to_preserve:
                stats["preserved"] += 1
                continue
            
            # Check if it's an old experiment (older than 7 days)
            item_age = time.time() - item.stat().st_mtime
            if item_age > 7 * 24 * 3600:  # 7 days
                try:
                    # Create backup before moving
                    self.create_backup(item.name, "auto")
                    
                    # Move to backups
                    backup_dest = self.backups_dir / f"auto_{item.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    shutil.move(str(item), str(backup_dest))
                    logger.info(f"Moved old experiment {item.name} to {backup_dest}")
                    stats["moved"] += 1
                except Exception as e:
                    logger.error(f"Failed to move experiment {item.name}: {e}")
            else:
                stats["cleaned"] += 1
        
        return stats
    
    def get_checkpoint_reuse_info(self, model_name: str) -> Optional[Dict]:
        """
        Get information about existing checkpoints for model reuse.
        
        Args:
            model_name: Name of the model to check
            
        Returns:
            Dictionary with checkpoint information or None if not found
        """
        checkpoint_dir = self.checkpoints_dir / model_name
        if not checkpoint_dir.exists():
            return None
        
        # Find the most recent checkpoint
        checkpoints = []
        for item in checkpoint_dir.rglob("*.pth"):
            checkpoints.append({
                "path": str(item),
                "size_mb": item.stat().st_size / (1024 * 1024),
                "modified": datetime.fromtimestamp(item.stat().st_mtime).isoformat(),
                "age_days": (time.time() - item.stat().st_mtime) / (24 * 3600)
            })
        
        if not checkpoints:
            return None
        
        # Sort by modification time (newest first)
        checkpoints.sort(key=lambda x: x["modified"], reverse=True)
        
        return {
            "model_name": model_name,
            "checkpoint_dir": str(checkpoint_dir),
            "latest_checkpoint": checkpoints[0],
            "total_checkpoints": len(checkpoints),
            "total_size_mb": sum(cp["size_mb"] for cp in checkpoints)
        }
    
    def ensure_no_overwriting(self, experiment_name: str) -> str:
        """
        Ensure no overwriting by creating a unique experiment name.
        
        Args:
            experiment_name: Base experiment name
            
        Returns:
            Unique experiment name
        """
        base_name = experiment_name
        counter = 1
        
        while (self.results_dir / experiment_name).exists():
            experiment_name = f"{base_name}_{counter}"
            counter += 1
        
        if experiment_name != base_name:
            logger.info(f"Experiment name changed from '{base_name}' to '{experiment_name}' to avoid overwriting")
        
        return experiment_name
    
    def list_backups(self) -> List[Dict]:
        """
        List all available backups with metadata.
        
        Returns:
            List of backup information dictionaries
        """
        backups = []
        
        for backup_dir in self.backups_dir.iterdir():
            if not backup_dir.is_dir():
                continue
            
            metadata_file = backup_dir / "backup_metadata.json"
            if metadata_file.exists():
                try:
                    with open(metadata_file, "r") as f:
                        metadata = json.load(f)
                    backups.append(metadata)
                except Exception as e:
                    logger.error(f"Failed to read metadata for {backup_dir.name}: {e}")
        else:
                # Fallback for backups without metadata
                backups.append({
                    "experiment_name": backup_dir.name,
                    "backup_type": "unknown",
                    "timestamp": "unknown",
                    "created_at": datetime.fromtimestamp(backup_dir.stat().st_mtime).isoformat(),
                    "backup_size_mb": self._get_directory_size(backup_dir)
                })
        
        return sorted(backups, key=lambda x: x.get("created_at", ""), reverse=True)
    
    def restore_backup(self, backup_name: str, restore_to: str = None) -> bool:
        """
        Restore a backup to the results/checkpoints directories.
        
        Args:
            backup_name: Name of the backup to restore
            restore_to: Custom restore location (optional)
            
        Returns:
            True if restoration was successful
        """
        backup_path = self.backups_dir / backup_name
        if not backup_path.exists():
            logger.error(f"Backup {backup_name} not found")
            return False
        
        try:
            # Read metadata
            metadata_file = backup_path / "backup_metadata.json"
            if metadata_file.exists():
                with open(metadata_file, "r") as f:
                    metadata = json.load(f)
                experiment_name = metadata.get("experiment_name", backup_name)
            else:
                experiment_name = backup_name
            
            # Determine restore location
            if restore_to:
                restore_base = Path(restore_to)
            else:
                restore_base = self.base_dir
            
            # Restore results
            results_backup = backup_path / "results"
            if results_backup.exists():
                restore_results = restore_base / "results" / experiment_name
                if restore_results.exists():
                    shutil.rmtree(restore_results)
                shutil.copytree(results_backup, restore_results)
                logger.info(f"Restored results to {restore_results}")
            
            # Restore checkpoints
            checkpoints_backup = backup_path / "checkpoints"
            if checkpoints_backup.exists():
                restore_checkpoints = restore_base / "checkpoints" / experiment_name
                if restore_checkpoints.exists():
                    shutil.rmtree(restore_checkpoints)
                shutil.copytree(checkpoints_backup, restore_checkpoints)
                logger.info(f"Restored checkpoints to {restore_checkpoints}")
            
            logger.info(f"Successfully restored backup {backup_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to restore backup {backup_name}: {e}")
            return False
    
    def _get_directory_size(self, path: Path) -> float:
        """Calculate directory size in MB."""
        total_size = 0
        try:
            for dirpath, dirnames, filenames in os.walk(path):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    total_size += os.path.getsize(filepath)
        except Exception:
            pass
        return total_size / (1024 * 1024)  # Convert to MB

def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Backup and Cleanup System for NNEdge-Privacy")
    parser.add_argument("action", choices=[
        "backup", "cleanup", "organize", "list", "restore", "checkpoint-info"
    ], help="Action to perform")
    parser.add_argument("--experiment", "-e", help="Experiment name")
    parser.add_argument("--backup-type", "-t", default="auto", 
                       choices=["auto", "manual", "critical"], help="Backup type")
    parser.add_argument("--max-age", "-m", type=int, default=30, 
                       help="Maximum age in days for cleanup")
    parser.add_argument("--backup-name", "-b", help="Backup name for restore")
    parser.add_argument("--model-name", help="Model name for checkpoint info")
    parser.add_argument("--restore-to", help="Custom restore location")
    
    args = parser.parse_args()
    
    manager = ExperimentBackupManager()
    
    if args.action == "backup":
        if not args.experiment:
            print("Error: --experiment is required for backup action")
            return
        backup_path = manager.create_backup(args.experiment, args.backup_type)
        print(f"Backup created: {backup_path}")
    
    elif args.action == "cleanup":
        removed_backups = manager.cleanup_old_backups(args.max_age)
        removed_checkpoints = manager.cleanup_old_checkpoints(args.max_age)
        print(f"Removed {removed_backups} old backups and {removed_checkpoints} old checkpoints")
    
    elif args.action == "organize":
        stats = manager.organize_results()
        print(f"Organization complete: {stats['moved']} moved, {stats['cleaned']} cleaned, {stats['preserved']} preserved")
    
    elif args.action == "list":
        backups = manager.list_backups()
        print(f"Found {len(backups)} backups:")
        for backup in backups:
            print(f"  {backup['experiment_name']} ({backup['backup_type']}) - {backup['created_at']} - {backup.get('backup_size_mb', 0):.1f}MB")
    
    elif args.action == "restore":
        if not args.backup_name:
            print("Error: --backup-name is required for restore action")
            return
        success = manager.restore_backup(args.backup_name, args.restore_to)
        print(f"Restore {'successful' if success else 'failed'}")
    
    elif args.action == "checkpoint-info":
        if not args.model_name:
            print("Error: --model-name is required for checkpoint-info action")
            return
        info = manager.get_checkpoint_reuse_info(args.model_name)
        if info:
            print(f"Checkpoint info for {args.model_name}:")
            print(f"  Latest: {info['latest_checkpoint']['path']}")
            print(f"  Age: {info['latest_checkpoint']['age_days']:.1f} days")
            print(f"  Size: {info['latest_checkpoint']['size_mb']:.1f}MB")
            print(f"  Total checkpoints: {info['total_checkpoints']}")
        else:
            print(f"No checkpoints found for {args.model_name}")

if __name__ == "__main__":
    main()
