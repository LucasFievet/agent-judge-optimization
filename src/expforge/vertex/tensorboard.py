"""TensorBoard resource management for Vertex AI."""

from __future__ import annotations

from typing import Optional

from google.cloud.aiplatform import Tensorboard

from expforge.config import ExpforgeConfig


def get_or_create_tensorboard(config: ExpforgeConfig, create: bool = False) -> tuple[Optional[object], bool]:
    """
    Get or create TensorBoard resource.
    
    Args:
        config: Configuration
        create: If True, create TensorBoard if it doesn't exist
    
    Returns:
        Tuple of (tensorboard object or None, was_created: bool)
    """
    tensorboards = Tensorboard.list(filter=f'display_name="{config.tensorboard_name}"')
    
    if tensorboards:
        return tensorboards[0], False
    elif create:
        tensorboard = Tensorboard.create(display_name=config.tensorboard_name)
        return tensorboard, True
    return None, False


def check_tensorboard_access(config: ExpforgeConfig) -> tuple[bool, Optional[str]]:
    """
    Check if TensorBoard is accessible.
    
    Args:
        config: Configuration
    
    Returns:
        Tuple of (is_accessible: bool, error_message: Optional[str])
    """
    tensorboard, _ = get_or_create_tensorboard(config, create=False)
    if tensorboard:
        return True, None
    return False, f"TensorBoard '{config.tensorboard_name}' not accessible"

