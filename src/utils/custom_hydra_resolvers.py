from omegaconf import DictConfig, MissingMandatoryValue, OmegaConf


def _validate_root_config(_root_: DictConfig) -> tuple[bool, str, str]:
    """Validate root config and return (is_valid, model_name, error_msg)."""
    if not hasattr(_root_, "model"):
        return False, "", "{MARKER_ERR:model_config_not_found_at_root}"

    model_cfg = _root_.model
    if not hasattr(model_cfg, "model_name_or_path"):
        return False, "", "{MARKER_ERR:model_name_or_path_missing}"

    return True, model_cfg.model_name_or_path, ""


def _get_model_specific_config(
    model_cfg: DictConfig, model_name: str
) -> tuple[bool, DictConfig, str]:
    """Get model-specific config and return (is_valid, config, error_msg)."""
    specific_configs = getattr(
        model_cfg, "available_model_specific_configs", None
    )
    if not isinstance(specific_configs, DictConfig):
        err_p1 = "{{MARKER_ERR:available_model_specific_configs_missing_"
        err_p2 = f"or_invalid_for_{model_name}}}"
        return False, OmegaConf.create({}), err_p1 + err_p2

    current_model_specific_config = specific_configs.get(model_name)
    if current_model_specific_config is None:
        return (
            False,
            OmegaConf.create({}),
            f"{{MARKER_ERR:config_for_{model_name}_not_found}}",
        )

    return True, current_model_specific_config, ""


def _get_marker_value(
    config: DictConfig, marker_key: str, model_name: str
) -> tuple[bool, str, str]:
    """Get marker value and return (is_valid, value, error_msg)."""
    markers_cfg = getattr(config, "markers", None)
    if not isinstance(markers_cfg, DictConfig):
        err_p1 = "{{MARKER_ERR:markers_missing_or_invalid_for_"
        return False, "", f"{err_p1}{model_name}}}"

    marker_value = markers_cfg.get(marker_key)
    if marker_value is None:
        return (
            False,
            "",
            f"{{MARKER_ERR:{marker_key}_not_in_markers_for_{model_name}}}",
        )

    if not isinstance(marker_value, str):
        return (
            False,
            "",
            f"{{MARKER_ERR:{marker_key}_not_string_for_{model_name}}}",
        )

    return True, marker_value, ""


def get_marker_resolver(marker_key: str, *, _root_: DictConfig) -> str:
    """Resolves a model-specific marker string (e.g., reasoning_start).

    Accesses the root config via `_root_` to find the active model's
    defined markers.
    """
    try:
        # Validate root configuration
        is_valid, model_name, error_msg = _validate_root_config(_root_)
        if not is_valid:
            return error_msg

        # Get model-specific configuration
        is_valid, config, error_msg = _get_model_specific_config(
            _root_.model, model_name
        )
        if not is_valid:
            return error_msg

        # Get marker value
        is_valid, marker_value, error_msg = _get_marker_value(
            config, marker_key, model_name
        )
        if not is_valid:
            return error_msg

        return marker_value

    except MissingMandatoryValue as e:
        return f"{{MARKER_ERR:Missing_config_value_{e}}}"
    except Exception as e:
        # In a real application, consider logging this exception
        # logger.error(f"Unexpected error in get_marker_resolver: {e}")
        return f"{{MARKER_ERR:Unexpected_{type(e).__name__}}}"


def get_model_markers_dict_resolver(*, _root_: DictConfig) -> DictConfig:
    """Resolves the entire markers dictionary for the current model.

    Accesses `_root_.model.available_model_specific_configs` based on
    `_root_.model.model_name_or_path`.
    Returns an empty DictConfig if markers are not found or on error.
    """
    try:
        if not hasattr(_root_, "model"):
            # Should ideally log this error too
            return OmegaConf.create({})
        model_cfg = _root_.model

        if not hasattr(model_cfg, "model_name_or_path"):
            return OmegaConf.create({})
        model_name = model_cfg.model_name_or_path

        specific_configs = getattr(
            model_cfg, "available_model_specific_configs", None
        )
        if not isinstance(specific_configs, DictConfig):
            return OmegaConf.create({})

        current_model_specific_config = specific_configs.get(model_name)
        if current_model_specific_config is None:
            return OmegaConf.create({})

        markers_dict = getattr(current_model_specific_config, "markers", None)
        if not isinstance(markers_dict, DictConfig):
            return OmegaConf.create({})

        return markers_dict
    except Exception:
        # Log unexpected errors in a real application
        return OmegaConf.create({})  # Return empty dict on any error


def register_custom_resolvers() -> None:
    """Registers all custom OmegaConf resolvers for the application."""
    OmegaConf.register_new_resolver(
        "get_marker", get_marker_resolver, use_cache=False
    )
    OmegaConf.register_new_resolver(
        "get_model_markers_dict",
        get_model_markers_dict_resolver,
        use_cache=False,  # Must be False when using _root_
    )
