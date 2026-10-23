"""Content-aware JSON caching for notebook report tables and printed output."""

from functools import wraps
from hashlib import sha256
from io import StringIO
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
from IPython.display import display

DEFAULT_RESULTS_DIR = (
    Path(__file__).resolve().parents[2] / "results" / "00_cleaning_and_preprocessing"
)


def _fingerprint(value):
    if isinstance(value, pd.DataFrame):
        digest = sha256()
        digest.update(repr(list(value.columns)).encode())
        digest.update(repr([str(dtype) for dtype in value.dtypes]).encode())
        digest.update(repr(value.index.names).encode())
        digest.update(pd.util.hash_pandas_object(value, index=True).values.tobytes())
        return {"dataframe": digest.hexdigest()}
    if isinstance(value, Path):
        return {"path": str(value.resolve()), "contents": sha256(value.read_bytes()).hexdigest() if value.is_file() else None}
    if isinstance(value, dict):
        return {str(k): _fingerprint(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_fingerprint(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _encode(value):
    if isinstance(value, pd.DataFrame):
        return {"__dataframe__": value.to_json(orient="table", date_format="iso", double_precision=15)}
    if isinstance(value, dict):
        return {key: _encode(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_encode(item) for item in value]
    return value


def _decode(value):
    if isinstance(value, dict):
        if "__dataframe__" in value:
            return pd.read_json(StringIO(value["__dataframe__"]), orient="table")
        return {key: _decode(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode(item) for item in value]
    return value


class ReportOutput:
    """Collect messages and tables so computed and cached reports display alike."""

    def __init__(self):
        self.events = []

    def message(self, *values, sep=" ", end="\n"):
        self.events.append({"kind": "message", "value": sep.join(map(str, values)) + end})

    def table(self, table):
        self.events.append({"kind": "table", "value": table})

    def replay(self):
        with pd.option_context("display.max_colwidth", None):
            for event in self.events:
                if event["kind"] == "message":
                    print(event["value"], end="")
                else:
                    display(event["value"])


def cached_report(function):
    """Cache report results by data, arguments, source code, and pandas version.

    Every report accepts cache_dir=None, use_cache=True, and
    force_refresh=False. Method defaults come from the DatasetEDA instance.
    Set use_cache=False to avoid both reading and writing cache files.
    """
    @wraps(function)
    def wrapper(*args, **kwargs):
        instance = args[0] if args and hasattr(args[0], "dataframe") else None
        cache_dir = kwargs.pop("cache_dir", getattr(instance, "cache_dir", DEFAULT_RESULTS_DIR))
        use_cache = kwargs.pop("use_cache", getattr(instance, "use_cache", True))
        force_refresh = kwargs.pop("force_refresh", getattr(instance, "force_refresh", False))
        report = ReportOutput()
        bound = inspect.signature(function).bind(*args, **kwargs)
        bound.apply_defaults()
        inputs = {}
        for name, value in bound.arguments.items():
            if name == "_report":
                continue
            if name == "self":
                value = value.dataframe
            # Include the description file's contents, even if passed as a string.
            if name == "description_path":
                value = Path(value)
            inputs[name] = _fingerprint(value)
        source = "".join(path.read_text() for path in sorted(Path(__file__).parent.glob("*.py")))
        key_data = {"inputs": inputs, "source": source, "pandas": pd.__version__, "function": function.__qualname__}
        key = sha256(json.dumps(key_data, sort_keys=True, default=str).encode()).hexdigest()
        folder = Path(cache_dir) if cache_dir is not None else DEFAULT_RESULTS_DIR
        path = folder / f"{function.__name__}_{key}.json"
        loaded = False
        if use_cache and not force_refresh and path.exists():
            try:
                payload = _decode(json.loads(path.read_text()))
                result = payload["result"]
                report.events = payload["events"]
                loaded = True
            except (ValueError, TypeError, KeyError):
                loaded = False
        if not loaded:
            result = function(*args, **kwargs, _report=report)
            if use_cache:
                folder.mkdir(parents=True, exist_ok=True)
                payload = {"result": result, "events": report.events}
                temporary = path.with_suffix(".tmp")
                temporary.write_text(json.dumps(_encode(payload)))
                temporary.replace(path)
                for index, event in enumerate(report.events, 1):
                    if event["kind"] == "table":
                        event["value"].to_csv(folder / f"{function.__name__}_{key}_table_{index}.csv", index=True)
        if use_cache:
            print(f"{'Loaded cached' if loaded else 'Saved'} report: {path.name}")
        report.replay()
        return result

    return wrapper
