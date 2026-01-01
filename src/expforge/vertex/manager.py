"""Vertex AI resource manager with CLI support."""

from __future__ import annotations

import sys

from expforge.vertex.bucket import get_or_create_bucket, check_bucket_access
from expforge.vertex.context import get_config
from expforge.vertex.experiment import get_or_create_experiment, check_experiment_access
from expforge.vertex.tensorboard import get_or_create_tensorboard, check_tensorboard_access


class VertexManager:
    """Manager for Vertex AI resources."""
    
    def __init__(self):
        """Initialize Vertex Manager."""
        self.config = get_config()
    
    def check(self) -> dict:
        """
        Check status of all Vertex AI resources.
        
        Returns:
            Dictionary with check results
        """
        project_ok = False
        project_error = None
        try:
            project_ok = True
        except Exception as e:
            error_msg = str(e)
            if "expired" in error_msg.lower() or "revoked" in error_msg.lower() or "invalid_grant" in error_msg:
                project_error = "Credentials expired or invalid"
            else:
                project_error = f"Failed to initialize project: {e}"
        
        if not project_ok:
            template = """1. Checking project access...
   ✗ {error}
   To fix: gcloud auth application-default login

Summary:
  ✗ Project access: FAILED
  ⚠ Cannot check other resources without valid credentials"""
            output = template.format(error=project_error)
            return {
                "config": self.config,
                "results": {
                    "project_access": False,
                    "bucket_access": False,
                    "experiment_access": False,
                    "tensorboard_access": False,
                    "all_passed": False,
                },
                "output": output,
            }
        
        bucket_ok, bucket_error = check_bucket_access(self.config)
        exp_ok, exp_error = check_experiment_access(self.config)
        tb_ok, tb_error = check_tensorboard_access(self.config)
        all_passed = project_ok and bucket_ok
        
        # Format output
        template = """1. Checking project access...
   ✓ Project initialized successfully

2. Checking GCS bucket...
   {bucket_status}

3. Checking Vertex AI Experiment...
   {experiment_status}

4. Checking TensorBoard resource...
   {tensorboard_status}

Summary:
  ✓ Project access: OK
  {bucket_summary}
  {experiment_summary}
  {tensorboard_summary}

{final_message}"""
        
        bucket_status = f"✓ Bucket '{self.config.bucket_name}' is accessible" if bucket_ok else f"✗ {bucket_error}"
        experiment_status = f"✓ Experiment '{self.config.experiment_name}' is accessible" if exp_ok else f"⚠ {exp_error} (will be created on first use)"
        tensorboard_status = f"✓ TensorBoard '{self.config.tensorboard_name}' is accessible" if tb_ok else f"⚠ {tb_error} (will be created on first use)"
        
        bucket_summary = f"✓ Bucket '{self.config.bucket_name}': OK" if bucket_ok else f"✗ Bucket '{self.config.bucket_name}': NEEDS FIXING"
        experiment_summary = f"✓ Experiment '{self.config.experiment_name}': OK" if exp_ok else f"⚠ Experiment '{self.config.experiment_name}': Not found (will be auto-created)"
        tensorboard_summary = f"✓ TensorBoard '{self.config.tensorboard_name}': OK" if tb_ok else f"⚠ TensorBoard '{self.config.tensorboard_name}': Not found (will be auto-created)"
        
        final_message = "✓ All critical resources are ready!" if all_passed else "⚠ Some resources need attention. See details above.\n  Run with --create to automatically create missing resources."
        
        output = template.format(
            bucket_status=bucket_status,
            experiment_status=experiment_status,
            tensorboard_status=tensorboard_status,
            bucket_summary=bucket_summary,
            experiment_summary=experiment_summary,
            tensorboard_summary=tensorboard_summary,
            final_message=final_message,
        )
        
        return {
            "config": self.config,
            "results": {
                "project_access": project_ok,
                "bucket_access": bucket_ok,
                "experiment_access": exp_ok,
                "tensorboard_access": tb_ok,
                "all_passed": all_passed,
            },
            "output": output,
        }
    
    def create(self) -> dict:
        """
        Create missing Vertex AI resources.
        
        Returns:
            Dictionary with creation results
        """
        project_ok = True
        project_error = None
        
        if not project_ok:
            template = "Initializing Vertex AI...\n   ✗ {error}\n"
            output = template.format(error=project_error)
            return {
                "config": self.config,
                "results": {"all_created": False},
                "output": output,
            }
        
        bucket, bucket_created = get_or_create_bucket(self.config, create=True)
        experiment, exp_created = get_or_create_experiment(self.config, create=True)
        tensorboard, tb_created = get_or_create_tensorboard(self.config, create=True)
        
        # Link tensorboard to experiment if both exist
        tensorboard_linked = False
        if experiment and tensorboard:
            try:
                experiment.assign_backing_tensorboard(tensorboard)
                tensorboard_linked = True
            except Exception:
                pass
        
        all_created = bucket and experiment and tensorboard
        
        # Format output
        template = """Initializing Vertex AI...
   ✓ Project initialized

Creating GCS bucket...
   {bucket_status}

Creating Vertex AI Experiment...
   {experiment_status}

Creating TensorBoard resource...
   {tensorboard_status}{tensorboard_link}

Summary:
  {bucket_summary}
  {experiment_summary}
  {tensorboard_summary}

{final_message}"""
        
        bucket_status = f"✓ Bucket '{self.config.bucket_name}' created" if bucket_created else f"✓ Bucket '{self.config.bucket_name}' already exists" if bucket else f"✗ Failed to create bucket '{self.config.bucket_name}'"
        
        experiment_status = f"✓ Experiment '{self.config.experiment_name}' created" if exp_created else f"✓ Experiment '{self.config.experiment_name}' already exists" if experiment else f"✗ Failed to create experiment '{self.config.experiment_name}'"
        
        tensorboard_status = f"✓ TensorBoard '{self.config.tensorboard_name}' created" if tb_created else f"✓ TensorBoard '{self.config.tensorboard_name}' already exists" if tensorboard else f"✗ Failed to create TensorBoard '{self.config.tensorboard_name}'"
        
        tensorboard_link = "\n   ✓ TensorBoard linked to experiment" if tensorboard_linked else ""
        
        bucket_summary = f"✓ Bucket '{self.config.bucket_name}': CREATED" if bucket_created else f"✓ Bucket '{self.config.bucket_name}': Already exists" if bucket else f"✗ Bucket '{self.config.bucket_name}': FAILED"
        
        experiment_summary = f"✓ Experiment '{self.config.experiment_name}': CREATED" if exp_created else f"✓ Experiment '{self.config.experiment_name}': Already exists" if experiment else f"✗ Experiment '{self.config.experiment_name}': FAILED"
        
        tensorboard_summary = f"✓ TensorBoard '{self.config.tensorboard_name}': CREATED" if tb_created else f"✓ TensorBoard '{self.config.tensorboard_name}': Already exists" if tensorboard else f"✗ TensorBoard '{self.config.tensorboard_name}': FAILED"
        
        final_message = "✓ All resources are ready!" if all_created else "⚠ Some resources could not be created. See details above."
        
        output = template.format(
            bucket_status=bucket_status,
            experiment_status=experiment_status,
            tensorboard_status=tensorboard_status,
            tensorboard_link=tensorboard_link,
            bucket_summary=bucket_summary,
            experiment_summary=experiment_summary,
            tensorboard_summary=tensorboard_summary,
            final_message=final_message,
        )
        
        return {
            "config": self.config,
            "results": {
                "bucket_created": bucket_created,
                "experiment_created": exp_created,
                "tensorboard_created": tb_created,
                "all_created": all_created,
            },
            "output": output,
        }


def main():
    """CLI entry point for vertex manager."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Manage Vertex AI resources")
    parser.add_argument("--check", action="store_true", help="Check resource status")
    parser.add_argument("--create", action="store_true", help="Create missing resources")
    args = parser.parse_args()
    
    manager = VertexManager()
    
    if args.check:
        result = manager.check()
        print(result["output"])
        sys.exit(0 if result["results"]["all_passed"] else 1)
    elif args.create:
        result = manager.create()
        print(result["output"])
        sys.exit(0 if result["results"]["all_created"] else 1)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

