import csv
import numbers
import shutil
import logging
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Union, Any

from .config_yaml import dump_config

class ArtifactManager:
    def __init__(self, work_dir: Path, logger: logging.Logger):
        self.work_dir = work_dir
        self.logger = logger
        self._config = {}

    def save_config(self, config: Any) -> None:
        """Dicts merge into what was saved before, any other config (e.g. a dataclass) replaces it."""
        if isinstance(config, dict) and isinstance(self._config, dict):
            self._config.update(config)
        else:
            self._config = config
        with open(self.work_dir / 'config.yaml', 'w', encoding='utf-8') as f:
            f.write(dump_config(self._config))

    def save_csv(self, name: str, data: Dict[str, List], analysis_dir: Optional[Path]) -> Optional[Path]:
        target_dir = analysis_dir if analysis_dir else (self.work_dir / "analysis")
        target_dir.mkdir(parents=True, exist_ok=True)
        csv_path = target_dir / f"{name}_analysis.csv"

        keys, values = list(data.keys()), list(data.values())
        if not values: return None

        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(keys)
            writer.writerows(list(zip(*values)))
        return csv_path

    def save_media(self, name: str, obj: Any, out_dir: Optional[Path], **kwargs) -> Optional[Path]:
        """ Mainly for plots and images thourgh matplotlib or seaborn """
        target_dir = out_dir if out_dir else (self.work_dir / "outputs")
        target_dir.mkdir(parents=True, exist_ok=True)
        file_path = target_dir / name

        if hasattr(obj, 'savefig'): obj.savefig(file_path, **kwargs)
        elif hasattr(obj, 'save'): obj.save(file_path, **kwargs)
        else:
            self.logger.error(f"Object {name} has no .save() or .savefig() method.")
            return None
        return file_path

    def save_video(self, name: str, video: Union[str, Path, np.ndarray], out_dir: Optional[Path], **imageio_kwargs) -> Optional[Path]:
        target_dir = out_dir if out_dir else (self.work_dir / "outputs")
        target_dir.mkdir(parents=True, exist_ok=True)

        target_path = target_dir / name
        if target_path.suffix != '.mp4': 
            target_path = target_path.with_suffix('.mp4')

        if isinstance(video, (str, Path)):
            source_path = Path(video)
            if not source_path.exists():
                self.logger.error(f"Video file {source_path} not found.")
                return None
            if source_path.absolute() != target_path.absolute():
                shutil.copy(source_path, target_path)
        else:
            import imageio
            if 'fps' not in imageio_kwargs: imageio_kwargs["fps"] = 30
            if 'quality' not in imageio_kwargs: imageio_kwargs["quality"] = 8
            imageio.mimwrite(target_path, video, **imageio_kwargs)

        return target_path

    @staticmethod
    def _sanitize_config(data: Any) -> Any:
        if isinstance(data, dict): return {k: ArtifactManager._sanitize_config(v) for k, v in data.items()}
        if isinstance(data, (list, tuple)): return [ArtifactManager._sanitize_config(v) for v in data]
        if isinstance(data, (numbers.Number, np.ndarray, bool, type(None))): return data
        return str(data)
