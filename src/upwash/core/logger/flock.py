import copy
import wandb
import atexit

class FlockLogger():
    # `config` is just a placeholder now
    def __init__(self, project, name, config=None):
        self.run = wandb.init(
            project=project,
            name=name,
            config=config or {}
        )
        self._finished = False

        atexit.register(self.finish)

    def log(self, metrics, step, commit=True):
        if self._finished:
            raise RuntimeError("Logger is already finished")

        self.run.log(
            copy.deepcopy(metrics), 
            step=step, 
            commit=commit
        )

    def finish(self):
        if self._finished:
            return

        self._finished = True
        self.run.finish()