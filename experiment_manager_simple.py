#!/usr/bin/env python3
"""
Simplified Experiment Manager for NNEdge-Privacy

This module provides basic experiment management without torch dependencies.
"""

import os
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging
from backup_system import ExperimentBackupManager

logger = logging.getLogger(__name__)

class SimpleExperimentManager:
    """Simplified experiment manager without torch dependencies."""
    
    def __init__(self, base_dir: str = "."):
        self.base_dir = Path(base_dir)
        self.results_dir = self.base_dir / "results"
        self.checkpoints_dir = self.base_dir / "checkpoints"
        self.backup_manager = ExperimentBackupManager(base_dir)
        
        # Create directories if they don't exist
        self.results_dir.mkdir(exist_ok=True)
        self.checkpoints_dir.mkdir(exist_ok=True)
        
        # Experiment registry
        self.registry_file = self.base_dir / "experiment_registry.json"
        self.experiments = self._load_registry()
    
    def create_experiment(self, 
                         base_name: str,
                         config: Dict[str, Any] = None,
                         description: str = None) -> str:
        """
        Create a new experiment with a unique name.
        
        Args:
            base_name: Base name for the experiment
            config: Experiment configuration
            description: Experiment description
            
        Returns:
            Unique experiment name
        """
        # Ensure unique name
        experiment_name = self.backup_manager.ensure_no_overwriting(base_name)
        
        # Create experiment directory
        experiment_dir = self.results_dir / experiment_name
        experiment_dir.mkdir(exist_ok=True)
        
        # Create experiment metadata
        metadata = {
            "experiment_name": experiment_name,
            "base_name": base_name,
            "created_at": datetime.now().isoformat(),
            "status": "created",
            "config": config or {},
            "description": description or "",
            "results_files": [],
            "checkpoints": []
        }
        
        # Save metadata
        metadata_file = experiment_dir / "experiment_metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)
        
        # Register experiment
        self.experiments[experiment_name] = metadata
        self._save_registry()
        
        logger.info(f"Created experiment: {experiment_name}")
        return experiment_name
    
    def get_experiment_info(self, experiment_name: str) -> Optional[Dict[str, Any]]:
        """Get information about an experiment."""
        return self.experiments.get(experiment_name)
    
    def list_experiments(self, status: str = None) -> List[Dict[str, Any]]:
        """List all experiments, optionally filtered by status."""
        experiments = list(self.experiments.values())
        
        if status:
            experiments = [exp for exp in experiments if exp.get("status") == status]
        
        return sorted(experiments, key=lambda x: x.get("created_at", ""), reverse=True)
    
    def update_experiment_status(self, experiment_name: str, status: str, 
                                additional_info: Dict[str, Any] = None):
        """Update experiment status and additional information."""
        if experiment_name not in self.experiments:
            logger.warning(f"Experiment {experiment_name} not found in registry")
            return
        
        # Update status
        self.experiments[experiment_name]["status"] = status
        
        # Update additional info if provided
        if additional_info:
            self.experiments[experiment_name].update(additional_info)
        
        # Update metadata file
        experiment_dir = self.results_dir / experiment_name
        metadata_file = experiment_dir / "experiment_metadata.json"
        
        if metadata_file.exists():
            with open(metadata_file, "w") as f:
                json.dump(self.experiments[experiment_name], f, indent=2, default=str)
        
        self._save_registry()
    
    def add_sub_experiment(self, parent_experiment: str, sub_experiment: str):
        """Add a sub-experiment to a parent experiment."""
        if parent_experiment not in self.experiments:
            logger.warning(f"Parent experiment {parent_experiment} not found in registry")
            return
        
        if sub_experiment not in self.experiments:
            logger.warning(f"Sub-experiment {sub_experiment} not found in registry")
            return
        
        # Initialize sub_experiments list if it doesn't exist
        if "sub_experiments" not in self.experiments[parent_experiment]:
            self.experiments[parent_experiment]["sub_experiments"] = []
        
        # Add sub-experiment if not already present
        if sub_experiment not in self.experiments[parent_experiment]["sub_experiments"]:
            self.experiments[parent_experiment]["sub_experiments"].append(sub_experiment)
            
            # Update metadata file
            experiment_dir = self.results_dir / parent_experiment
            metadata_file = experiment_dir / "experiment_metadata.json"
            
            if metadata_file.exists():
                with open(metadata_file, "w") as f:
                    json.dump(self.experiments[parent_experiment], f, indent=2, default=str)
            
            self._save_registry()
        
        logger.info(f"Added sub-experiment {sub_experiment} to {parent_experiment}")
    
    def add_result_file(self, experiment_name: str, file_path: str, 
                       file_type: str = "result"):
        """
        Add a result file to an experiment.
        
        Args:
            experiment_name: Name of the experiment
            file_path: Path to the result file
            file_type: Type of result file
        """
        if experiment_name not in self.experiments:
            logger.warning(f"Experiment {experiment_name} not found in registry")
            return
        
        result_info = {
            "path": file_path,
            "type": file_type,
            "added_at": datetime.now().isoformat(),
            "size_mb": Path(file_path).stat().st_size / (1024 * 1024) if Path(file_path).exists() else 0
        }
        
        if "results_files" not in self.experiments[experiment_name]:
            self.experiments[experiment_name]["results_files"] = []
        
        self.experiments[experiment_name]["results_files"].append(result_info)
        self._save_registry()
    
    def get_experiment_summary(self) -> Dict[str, Any]:
        """Get a summary of all experiments."""
        total_experiments = len(self.experiments)
        status_counts = {}
        
        for experiment in self.experiments.values():
            status = experiment.get("status", "unknown")
            status_counts[status] = status_counts.get(status, 0) + 1
        
        total_results_files = sum(len(exp.get("results_files", [])) for exp in self.experiments.values())
        total_checkpoints = sum(len(exp.get("checkpoints", [])) for exp in self.experiments.values())
        
        return {
            "total_experiments": total_experiments,
            "status_counts": status_counts,
            "total_results_files": total_results_files,
            "total_checkpoints": total_checkpoints,
            "recent_experiments": self.list_experiments()[:5]  # Last 5 experiments
        }
    
    def _load_registry(self) -> Dict[str, Any]:
        """Load experiment registry from file."""
        if self.registry_file.exists():
            try:
                with open(self.registry_file, "r") as f:
                    data = json.load(f)
                    return data.get("experiments", {})
            except Exception as e:
                logger.error(f"Failed to load experiment registry: {e}")
        
        return {}
    
    def _save_registry(self):
        """Save experiment registry to file."""
        try:
            registry_data = {
                "created_at": datetime.now().isoformat(),
                "version": "1.0",
                "description": "NNEdge-Privacy Experiment Registry",
                "experiments": self.experiments
            }
            with open(self.registry_file, "w") as f:
                json.dump(registry_data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save experiment registry: {e}")

def main():
    """Command-line interface for simplified experiment management."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Simplified Experiment Manager CLI")
    parser.add_argument("action", choices=[
        "create", "list", "info", "summary"
    ], help="Action to perform")
    parser.add_argument("--experiment", "-e", help="Experiment name")
    parser.add_argument("--base-name", "-b", help="Base name for new experiment")
    parser.add_argument("--config", "-c", help="Path to config file")
    parser.add_argument("--description", "-d", help="Experiment description")
    parser.add_argument("--status", "-s", help="Status filter for list")
    
    args = parser.parse_args()
    
    manager = SimpleExperimentManager()
    
    if args.action == "create":
        if not args.base_name:
            print("Error: --base-name is required for create action")
            return
        
        config = {}
        if args.config:
            with open(args.config, "r") as f:
                config = json.load(f)
        
        experiment_name = manager.create_experiment(
            args.base_name, config, args.description
        )
        print(f"Created experiment: {experiment_name}")
    
    elif args.action == "list":
        experiments = manager.list_experiments(args.status)
        print(f"Found {len(experiments)} experiments:")
        for exp in experiments:
            print(f"  {exp['experiment_name']} ({exp.get('status', 'unknown')}) - {exp.get('created_at', '')}")
    
    elif args.action == "info":
        if not args.experiment:
            print("Error: --experiment is required for info action")
            return
        
        info = manager.get_experiment_info(args.experiment)
        if info:
            print(f"Experiment info for {args.experiment}:")
            print(json.dumps(info, indent=2))
        else:
            print(f"Experiment {args.experiment} not found")
    
    elif args.action == "summary":
        summary = manager.get_experiment_summary()
        print("Experiment Summary:")
        print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
