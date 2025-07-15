from typing import List, Any, Dict, Optional
import importlib
import inspect


def instantiate_visualization_hooks(viz_names: List[str], extra_args: Optional[Dict[str, Any]] = None) -> List[Any]:
    """
    Instantiate visualization components from the refrakt_viz registry.

    Args:
        viz_names: List of visualization names (as specified in YAML).
        extra_args: Optional dict of extra arguments to pass to each visualization component.

    Returns:
        List of instantiated visualization components.
    """
    if extra_args is None:
        extra_args = {}
    # Dynamically import all supervised modules to populate the registry
    try:
        importlib.import_module("refrakt_viz.supervised.loss_accuracy")
        importlib.import_module("refrakt_viz.supervised.confusion_matrix")
        importlib.import_module("refrakt_viz.supervised.sample_predictions")
    except Exception as e:
        print(f"Warning: Could not import one or more visualization modules: {e}")
    viz_registry_module = importlib.import_module("refrakt_viz.registry")
    get_viz = getattr(viz_registry_module, "get_viz")
    visualizations = []
    for name in viz_names:
        try:
            viz_cls = get_viz(name)
            sig = inspect.signature(viz_cls.__init__)
            params = list(sig.parameters.keys())
            # Remove 'self' from params
            params = [p for p in params if p != 'self']
            if 'class_names' in params and 'class_names' in extra_args:
                visualizations.append(viz_cls(class_names=extra_args['class_names']))
            else:
                visualizations.append(viz_cls())
        except Exception as e:
            print(f"Warning: Could not instantiate visualization '{name}': {e}")
    return visualizations


def instantiate_explainability_hooks(xai_names: List[str], extra_args: Optional[Dict[str, Any]] = None) -> List[Any]:
    """
    Instantiate explainability components from the refrakt_xai registry (future use).

    Args:
        xai_names: List of explainability hook names (as specified in YAML).
        extra_args: Optional dict of extra arguments to pass to each explainability component.

    Returns:
        List of instantiated explainability components.
    """
    if extra_args is None:
        extra_args = {}
    try:
        xai_registry_module = importlib.import_module("refrakt_xai.registry")
        get_xai = getattr(xai_registry_module, "get_xai")
    except (ModuleNotFoundError, AttributeError):
        print("Warning: refrakt_xai registry not found. Skipping explainability hooks.")
        return []
    explainability_hooks = []
    for name in xai_names:
        try:
            xai_cls = get_xai(name)
            explainability_hooks.append(xai_cls(**extra_args))
        except Exception as e:
            print(f"Warning: Could not instantiate explainability hook '{name}': {e}")
    return explainability_hooks 