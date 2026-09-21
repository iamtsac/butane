import dataclasses
import enum
import functools
import importlib
from pathlib import Path
from typing import Any

import numpy as np
import yaml

__all__ = ["dump_config", "load_config"]


class _ConfigDumper(yaml.SafeDumper): pass
class _ConfigLoader(yaml.SafeLoader): pass


def _import_path(obj: Any) -> str:
    path = f"{obj.__module__}:{obj.__qualname__}"
    if "<" in path:
        raise TypeError(f"{path} cannot be imported back, define it at module level")
    return path


def _import(path: str) -> Any:
    module, qualname = path.split(":")
    obj = importlib.import_module(module)
    for name in qualname.split("."):
        obj = getattr(obj, name)
    return obj


def _represent(dumper: _ConfigDumper, obj: Any) -> yaml.Node:
    """Every value plain YAML cannot hold gets a tag naming what rebuilds it, anything else is refused."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        fields = {f.name: getattr(obj, f.name) for f in dataclasses.fields(obj) if f.init}
        return dumper.represent_mapping(f"!dataclass:{_import_path(type(obj))}", fields)
    if isinstance(obj, tuple):
        return dumper.represent_sequence("!tuple", obj)
    if isinstance(obj, Path):
        return dumper.represent_scalar("!path", str(obj))
    if isinstance(obj, enum.Enum):
        return dumper.represent_scalar(f"!enum:{_import_path(type(obj))}", obj.name)
    if isinstance(obj, np.generic):
        return dumper.represent_data(obj.item())
    if isinstance(obj, np.ndarray):
        return dumper.represent_mapping("!ndarray", {"dtype": str(obj.dtype), "data": obj.tolist()})
    if isinstance(obj, functools.partial):
        return dumper.represent_mapping(
            "!partial", {"func": obj.func, "args": list(obj.args), "kwargs": obj.keywords}
        )
    if isinstance(obj, type) or callable(obj) and hasattr(obj, "__qualname__"):
        return dumper.represent_scalar(f"!type:{_import_path(obj)}", "")
    raise TypeError(f"no lossless YAML form for {type(obj).__name__}: {obj!r}")


def _construct_partial(loader: _ConfigLoader, node: yaml.Node) -> functools.partial:
    fields = loader.construct_mapping(node, deep=True)
    return functools.partial(fields["func"], *fields["args"], **fields["kwargs"])


def _construct_ndarray(loader: _ConfigLoader, node: yaml.Node) -> np.ndarray:
    fields = loader.construct_mapping(node, deep=True)
    return np.array(fields["data"], dtype=fields["dtype"])


def _multiline_str_presenter(dumper: _ConfigDumper, data: str) -> yaml.Node:
    style = "|" if "\n" in data else None
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style=style)


_ConfigDumper.add_representer(str, _multiline_str_presenter)
_ConfigDumper.add_representer(tuple, _represent)
_ConfigDumper.add_multi_representer(object, _represent)

_ConfigLoader.add_multi_constructor(
    "!dataclass:", lambda loader, path, node: _import(path)(**loader.construct_mapping(node, deep=True))
)
_ConfigLoader.add_constructor("!tuple", lambda loader, node: tuple(loader.construct_sequence(node, deep=True)))
_ConfigLoader.add_constructor("!path", lambda loader, node: Path(loader.construct_scalar(node)))
_ConfigLoader.add_multi_constructor(
    "!enum:", lambda loader, path, node: _import(path)[loader.construct_scalar(node)]
)
_ConfigLoader.add_constructor("!ndarray", _construct_ndarray)
_ConfigLoader.add_constructor("!partial", _construct_partial)
_ConfigLoader.add_multi_constructor("!type:", lambda loader, path, node: _import(path))


def dump_config(config: Any) -> str:
    """Lossless YAML of a config: dataclasses, tuples, paths, enums, arrays, partials and importable
    classes/functions are tagged so `load_config` rebuilds the same objects."""
    return yaml.dump(config, Dumper=_ConfigDumper, sort_keys=False, allow_unicode=True)


def load_config(path: str | Path) -> Any:
    """Rebuilds what `dump_config` wrote. It imports the modules the file names, so load trusted files only."""
    return yaml.load(Path(path).read_text(encoding="utf-8"), Loader=_ConfigLoader)
