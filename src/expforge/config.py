"""Configuration management for Vertex AI resources."""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional


@dataclass
class VertexAIConfig:
    """Vertex AI configuration settings."""
    
    project_id: str = "lucas-fievet-research"
    location: str = "us-central1"
    bucket_name: str = "triple-mnist"
    experiment_name: str = "triple-mnist-experiments"
    tensorboard_name: str = "triple-mnist-tensorboard"
    
    # Custom Training Job defaults
    machine_type: str = "n1-standard-4"
    accelerator_type: Optional[str] = None
    accelerator_count: int = 0
    container_uri: str = "gcr.io/cloud-aiplatform/training/tf-cpu.2-11:latest"
    serving_container_image_uri: str = "us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-11:latest"
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> VertexAIConfig:
        """Create from dictionary."""
        return cls(**data)
    
    def save(self, path: Path):
        """Save configuration to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> VertexAIConfig:
        """Load configuration from JSON file."""
        if not path.exists():
            # Create default config if it doesn't exist
            config = cls()
            config.save(path)
            return config
        
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)


def get_config_path() -> Path:
    """Get the default configuration file path."""
    # Look for config in project root or user home
    project_root = Path(__file__).parent.parent.parent
    config_path = project_root / "vertex_config.json"
    
    # Also check for .expforge directory in home
    home_config = Path.home() / ".expforge" / "vertex_config.json"
    
    # Prefer project root, fall back to home
    if config_path.exists():
        return config_path
    elif home_config.exists():
        return home_config
    else:
        # Default to project root
        return config_path


def load_vertex_config() -> VertexAIConfig:
    """Load Vertex AI configuration from file or create default."""
    config_path = get_config_path()
    return VertexAIConfig.load(config_path)


def save_vertex_config(config: VertexAIConfig):
    """Save Vertex AI configuration to file."""
    config_path = get_config_path()
    config.save(config_path)


def check_current_account() -> dict:
    """
    Check which Google account is currently authenticated.
    
    Returns:
        Dictionary with account information
    """
    import subprocess
    import os
    
    result = {
        "account": None,
        "project": None,
        "error": None,
    }
    
    # Priority 1: Check environment variable (highest priority for folder scoping)
    env_project = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GCLOUD_PROJECT")
    if env_project:
        result["project"] = env_project
    
    # Priority 2: Check local config file (folder-specific)
    try:
        config = load_vertex_config()
        if config.project_id:
            # Only use local config if no env var was set
            if not result["project"]:
                result["project"] = config.project_id
    except Exception:
        pass
    
    # Priority 3: Try to get current account from gcloud (global config)
    try:
        output = subprocess.run(
            ["gcloud", "config", "get-value", "account"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if output.returncode == 0:
            result["account"] = output.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
        pass
    
    # Priority 4: Try to get current project from gcloud (global config, lowest priority)
    # Only if we haven't found a project yet
    if not result["project"]:
        try:
            output = subprocess.run(
                ["gcloud", "config", "get-value", "project"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if output.returncode == 0:
                result["project"] = output.stdout.strip()
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
            pass
    
    return result


def check_vertex_resources(fix: bool = False, verbose: bool = False) -> dict:
    """
    Check and optionally fix Vertex AI resources.
    
    Args:
        fix: If True, attempt to create missing resources (bucket, TensorBoard)
        verbose: Show detailed test results and error traces
    
    Returns:
        Dictionary with check results and messages
    """
    from google.cloud import aiplatform, storage
    from google.cloud.aiplatform import Tensorboard
    from google.cloud.exceptions import NotFound, Forbidden
    
    config = load_vertex_config()
    
    results = {
        "project_access": False,
        "bucket_access": False,
        "tensorboard_exists": False,
        "experiment_access": False,
        "bucket_created": False,
        "tensorboard_created": False,
        "errors": [],
    }
    
    messages = []
    
    # Step 1: Check project access and credentials
    messages.append("1. Checking project access...")
    try:
        aiplatform.init(project=config.project_id, location=config.location)
        results["project_access"] = True
        messages.append("   ✓ Project initialized successfully")
    except Exception as e:
        error_msg = str(e)
        if "expired" in error_msg.lower() or "revoked" in error_msg.lower() or "invalid_grant" in error_msg:
            messages.append("   ✗ Credentials expired or invalid")
            messages.append("")
            messages.append("   To fix: Run the following command to re-authenticate:")
            messages.append("   gcloud auth application-default login")
        else:
            messages.append(f"   ✗ Failed to initialize project: {e}")
        if verbose:
            import traceback
            messages.append(traceback.format_exc())
        results["errors"].append("credentials_error")
        # Can't proceed without credentials
        messages.append("")
        messages.append("Summary:")
        messages.append("  ✗ Project access: FAILED")
        messages.append("  ⚠ Cannot check other resources without valid credentials")
        messages.append("")
        messages.append("⚠ Please fix credentials and try again.")
        return {
            "config": config,
            "results": results,
            "messages": messages,
            "all_passed": False,
        }
    
    # Step 2: Check/create GCS bucket
    messages.append("2. Checking GCS bucket...")
    try:
        client = storage.Client(project=config.project_id)
        bucket = client.bucket(config.bucket_name)
        
        try:
            bucket.reload()
            results["bucket_access"] = True
            messages.append(f"   ✓ Bucket '{config.bucket_name}' is accessible")
            if verbose:
                messages.append(f"      Location: {bucket.location}")
                messages.append(f"      Storage class: {bucket.storage_class}")
        except NotFound:
            if fix:
                messages.append(f"   Creating bucket '{config.bucket_name}'...")
                try:
                    bucket.create(location=config.location)
                    messages.append(f"   ✓ Bucket '{config.bucket_name}' created successfully")
                    results["bucket_created"] = True
                    results["bucket_access"] = True
                except Exception as e:
                    messages.append(f"   ✗ Failed to create bucket: {e}")
                    results["errors"].append(f"bucket_creation_failed: {e}")
                    if verbose:
                        import traceback
                        messages.append(traceback.format_exc())
            else:
                messages.append(f"   ✗ Bucket '{config.bucket_name}' not found")
                messages.append(f"   To create it, run: expforge check-vertex-resources --fix")
                results["errors"].append("bucket_not_found")
        except Forbidden:
            messages.append(f"   ✗ Access denied to bucket '{config.bucket_name}'")
            messages.append("   Check your IAM permissions for the bucket")
            results["errors"].append("bucket_access_denied")
    except Exception as e:
        error_str = str(e)
        # Check if this is a credential error
        if "invalid_grant" in error_str or "expired" in error_str.lower() or "revoked" in error_str.lower():
            messages.append(f"   ✗ Credential error: Token expired or revoked")
            messages.append("")
            messages.append("   To fix credentials:")
            messages.append("   1. Re-authenticate: gcloud auth application-default login")
            messages.append("   2. Verify account: expforge check-account")
            messages.append("   3. Re-run check: expforge check-vertex-resources")
            results["errors"].append("credentials_expired")
        else:
            messages.append(f"   ✗ Error checking bucket: {e}")
            results["errors"].append(f"bucket_error: {e}")
            if verbose:
                import traceback
                messages.append(traceback.format_exc())
    
    # Step 3: Check/create TensorBoard
    messages.append("3. Checking TensorBoard resource...")
    try:
        aiplatform.init(project=config.project_id, location=config.location)
        tensorboards = Tensorboard.list(filter=f'display_name="{config.tensorboard_name}"')
        
        if tensorboards:
            results["tensorboard_exists"] = True
            messages.append(f"   ✓ TensorBoard '{config.tensorboard_name}' exists")
            if verbose:
                tb = tensorboards[0]
                messages.append(f"      Resource name: {tb.name}")
        else:
            if fix:
                messages.append(f"   Creating TensorBoard '{config.tensorboard_name}'...")
                try:
                    tensorboard = Tensorboard.create(display_name=config.tensorboard_name)
                    messages.append(f"   ✓ TensorBoard '{config.tensorboard_name}' created successfully")
                    messages.append(f"      Resource name: {tensorboard.name}")
                    results["tensorboard_created"] = True
                    results["tensorboard_exists"] = True
                except Exception as e:
                    messages.append(f"   ✗ Failed to create TensorBoard: {e}")
                    results["errors"].append(f"tensorboard_creation_failed: {e}")
        if verbose:
            import traceback
            messages.append(traceback.format_exc())
        else:
            messages.append(f"   ⚠ TensorBoard '{config.tensorboard_name}' not found (will be created on first use)")
    except Exception as e:
        error_str = str(e)
        # Check if this is a credential error
        if "invalid_grant" in error_str or "expired" in error_str.lower() or "revoked" in error_str.lower():
            messages.append(f"   ✗ Credential error: Token expired or revoked")
            messages.append("   Re-authenticate: gcloud auth application-default login")
            results["errors"].append("credentials_expired")
        else:
            messages.append(f"   ✗ Error checking TensorBoard: {e}")
            results["errors"].append(f"tensorboard_error: {e}")
            if verbose:
                import traceback
                messages.append(traceback.format_exc())
    
    # Step 4: Check Vertex AI Experiments access
    messages.append("4. Checking Vertex AI Experiments access...")
    try:
        aiplatform.init(project=config.project_id, location=config.location)
        from google.cloud.aiplatform import Experiment
        exp_list = Experiment.list(filter=f'display_name="{config.experiment_name}"')
        results["experiment_access"] = True
        if exp_list:
            messages.append(f"   ✓ Experiment '{config.experiment_name}' exists")
            if verbose:
                exp = exp_list[0]
                messages.append(f"      Resource name: {exp.name}")
        else:
            messages.append(f"   ⚠ Experiment '{config.experiment_name}' not found (will be created on first use)")
    except ImportError as e:
        messages.append("   ⚠ Experiments API not available in this SDK version")
        messages.append("   This usually means the Vertex AI API is not enabled.")
        messages.append(f"   Enable it: gcloud services enable aiplatform.googleapis.com --project={config.project_id}")
        if verbose:
            messages.append(f"   Import error: {e}")
    except Exception as e:
        error_str = str(e)
        # Check if this is an API not enabled error
        if "not enabled" in error_str.lower() or "api" in error_str.lower() and "403" in error_str:
            messages.append(f"   ✗ Vertex AI API not enabled")
            messages.append(f"   Enable it: gcloud services enable aiplatform.googleapis.com --project={config.project_id}")
            messages.append(f"   Or via console: https://console.cloud.google.com/apis/library/aiplatform.googleapis.com?project={config.project_id}")
        else:
            messages.append(f"   ✗ Error checking experiments: {e}")
            results["errors"].append("experiment_api_error")
            if verbose:
                import traceback
                messages.append(traceback.format_exc())
    
    # Summary
    messages.append("")
    messages.append("Summary:")
    
    if results["project_access"]:
        messages.append("  ✓ Project access: OK")
    else:
        messages.append("  ✗ Project access: FAILED")
    
    if results["bucket_access"]:
        if results["bucket_created"]:
            messages.append(f"  ✓ Bucket '{config.bucket_name}': CREATED")
        else:
            messages.append(f"  ✓ Bucket '{config.bucket_name}': OK")
    else:
        messages.append(f"  ✗ Bucket '{config.bucket_name}': NEEDS FIXING")
    
    if results["tensorboard_exists"]:
        if results["tensorboard_created"]:
            messages.append(f"  ✓ TensorBoard '{config.tensorboard_name}': CREATED")
        else:
            messages.append(f"  ✓ TensorBoard '{config.tensorboard_name}': OK")
    else:
        messages.append(f"  ⚠ TensorBoard '{config.tensorboard_name}': Not found (will be auto-created)")
    
    if results["experiment_access"]:
        messages.append("  ✓ Experiments access: OK")
    else:
        messages.append("  ⚠ Experiments: Not found or API unavailable")
    
    messages.append("")
    all_passed = results["project_access"] and results["bucket_access"]
    if all_passed:
        messages.append("✓ All critical resources are ready!")
    else:
        messages.append("⚠ Some resources need attention. See details above.")
        if not fix:
            messages.append("  Run with --fix to automatically create missing resources.")
    
    return {
        "config": config,
        "results": results,
        "messages": messages,
        "all_passed": all_passed,
    }
