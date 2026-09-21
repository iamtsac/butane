import os
import time
import logging
from pathlib import Path
from typing import Optional, List, Dict, Union, Any

class WandbManager:
    def __init__(self, fpath: Path, overwrite: bool, resume: bool, logger: logging.Logger):
        import wandb
        self.wandb = wandb
        self.fpath = fpath
        self.logger = logger
        self.defined_metrics = set()

        self._init_run(overwrite, resume)

    def _init_run(self, overwrite: bool, resume: bool):
        from wandb.sdk.lib.runid import generate_id
        project = os.environ.get("WANDB_PROJECT")
        name = os.environ.get("WANDB_RUN")
        assert project is not None, "Set the WANDB_PROJECT env variable"

        id_file = self.fpath / "wandb_id.txt"
        lineage_ids = [line.strip() for line in id_file.read_text().split('\n') if line.strip()] if id_file.exists() else []

        if overwrite and lineage_ids:
            self._delete_cloud_lineage(project, lineage_ids)
            lineage_ids = []

        if resume and not overwrite and lineage_ids:
            run_name = lineage_ids[-1]
        else:
            run_id = generate_id()
            run_name = f"{self.fpath.name}_{run_id}" if name is None else name

        mode = "a" if (resume and not overwrite) else "w"
        prefix = "\n" if (resume and not overwrite and lineage_ids) else ""
        with open(id_file, mode) as f:
            f.write(f"{prefix}{run_name}")

        self.wandb.init(
            project=project, name=run_name, id=run_name, dir=str(self.fpath),
            settings=self.wandb.Settings(_disable_stats=True, _disable_meta=True)
        )
        self.wandb.define_metric("step", hidden=True)

    def _delete_cloud_lineage(self, project: str, lineage_ids: List[str]):
        self.logger.info(f"🗑️ Deleting {len(lineage_ids)} old WandB run(s)...")
        api = self.wandb.Api()
        for old_id in lineage_ids:
            try:
                api.run(f"{project}/{old_id}").delete()
            except Exception as e:
                self.logger.warning(f"Could not delete {old_id}: {e}")

    def define_metrics(self, flat_stats: dict):
        for k in flat_stats.keys():
            if k not in self.defined_metrics and k != "step":
                self.wandb.define_metric(k, step_metric="step")
                self.defined_metrics.add(k)

    def log_metrics(self, entry: dict, step: int):
        self.wandb.log(entry, step=step, commit=True)

    def update_config(self, config: dict):
        self.wandb.config.update(config, allow_val_change=True)

    def replay_history(self, history_rows: List[dict]):
        for i, row in enumerate(history_rows):
            step = row.get("step", i + 1)
            self.wandb.log(row, step=step)
        self.logger.info(f"✅ Replayed {len(history_rows)} steps of history into WandB.")

    def append_resume_step(self, step: int):
        if getattr(self.wandb, "run", None) is None:
            return

        current_name = self.wandb.run.name
        new_name = f"{current_name}_resume_from_step_{step}"
        self.wandb.run.name = new_name

        try:
            api = self.wandb.Api()
            server_run = api.run(f"{self.wandb.run.project}/{self.wandb.run.id}")
            server_run.name = new_name
            server_run.update()
        except Exception as e:
            self.logger.warning(f"Could not push name change to WandB server: {e}")

    # Artifact specific logs
    def log_plot(self, name: str, plot: Any, step: int): self.wandb.log({f"plot/{name}": plot}, step=step)
    def log_image(self, name: str, path: Path, step: int): self.wandb.log({f"images/{name}": self.wandb.Image(str(path))}, step=step)
    def log_video(self, name: str, path: Path, step: int): self.wandb.log({f"videos/{name}": self.wandb.Video(str(path), format="mp4")}, step=step)
    def log_table(self, name: str, data: dict, step: int):
        keys, rows = list(data.keys()), list(zip(*list(data.values())))
        self.wandb.log({f"analysis/{name}_{step}": self.wandb.Table(columns=keys, data=rows)}, step=step)
