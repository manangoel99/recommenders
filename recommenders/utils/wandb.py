from typing import Optional

try:
    import wandb
    from wandb.wandb_run import Run
except ModuleNotFoundError:
    wandb, Run = None, None

class WandbLogger:
    def __init__(self,
        name: Optional[str] = None,
        save_dir: Optional[str] = None,
        offline: Optional[bool] = False,
        id: Optional[str] = None,
        anonymous: Optional[bool] = None,
        version: Optional[str] = None,
        project: Optional[str] = None,
        log_model: Optional[bool] = False,
        experiment=None,
        prefix: Optional[str] = "",
        **kwargs
    ):
        if wandb is None:
            raise ModuleNotFoundError(
                "You want to use `wandb` logger which is not installed yet,"
                " install it with `pip install wandb`."  # pragma: no-cover
            )
        self._offline = offline
        self._log_model = log_model
        self._prefix = prefix
        self._experiment = experiment
        self._logged_model_time = {}
        self._checkpoint_callback = None
        # set wandb init arguments
        anonymous_lut = {True: "allow", False: None}
        self._wandb_init = dict(
            name=name,
            project=project,
            id=version or id,
            dir=save_dir,
            resume="allow",
            anonymous=anonymous_lut.get(anonymous, anonymous),
        )
        self._wandb_init.update(**kwargs)
        # extract parameters
        self._save_dir = self._wandb_init.get("dir")
        self._name = self._wandb_init.get("name")
        self._id = self._wandb_init.get("id")
    
    def __getstate__(self):
        state = self.__dict__.copy()
        # args needed to reload correct experiment
        state["_id"] = self._experiment.id if self._experiment is not None else None

        # cannot be pickled
        state["_experiment"] = None
        return state

    def set_experiment(self, experiment):
        self._experiment = experiment

    @property
    def experiment(self):
        if self._experiment is None:
            self._experiment = wandb.init(**self._wandb_init)
        return self._experiment

    def watch(self, model):
        self.experiment.watch(model)

    def log_metrics(self, metrics):
        self.experiment.log(metrics)