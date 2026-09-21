from __future__ import annotations

import argparse
import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab_fpo.rl_cfg import FpoRslRlOnPolicyRunnerCfg


def add_fpo_args(parser: argparse.ArgumentParser):
    """Add FPO arguments to the parser."""
    arg_group = parser.add_argument_group("fpo", description="Arguments for FPO agent.")
    arg_group.add_argument(
        "--experiment_name", type=str, default=None, help="Name of the experiment folder where logs will be stored."
    )
    arg_group.add_argument(
        "--algorithm",
        type=str,
        default=None,
        choices=("fpo", "imf_fpo", "pmf_fpo", "fsppo"),
        help=(
            "Policy variant. 'imf_fpo' selects Improved MeanFlow and "
            "'pmf_fpo' selects Pixel MeanFlow; 'fsppo' adds a same-noise "
            "transport-map trust region; omitted keeps the task default."
        ),
    )
    arg_group.add_argument("--run_name", type=str, default=None, help="Run name suffix to the log directory.")
    arg_group.add_argument("--resume", action="store_true", default=False, help="Whether to resume from a checkpoint.")
    arg_group.add_argument("--load_run", type=str, default=None, help="Name of the run folder to resume from.")
    arg_group.add_argument("--checkpoint", type=str, default=None, help="Checkpoint file to resume from.")
    arg_group.add_argument(
        "--logger", type=str, default=None, choices={"wandb", "tensorboard", "neptune"}, help="Logger module to use."
    )
    arg_group.add_argument(
        "--log_project_name", type=str, default=None, help="Name of the logging project when using wandb or neptune."
    )


def parse_fpo_cfg(task_name: str, args_cli: argparse.Namespace) -> FpoRslRlOnPolicyRunnerCfg:
    """Parse configuration for FPO agent based on inputs.

    Looks up the task config from the isaaclab_fpo registry instead of gym kwargs.
    """
    from isaaclab_fpo.task_cfgs import TASK_CONFIGS

    # train receives a task ID directly while some play scripts pass an
    # IsaacLab namespace prefix.  The registry stores the bare Gym task ID.
    task_name = task_name.split(":")[-1]

    if task_name not in TASK_CONFIGS:
        raise KeyError(
            f"No FPO config registered for task '{task_name}'. "
            f"Available tasks: {sorted(TASK_CONFIGS.keys())}"
        )
    agent_cfg = TASK_CONFIGS[task_name]()
    agent_cfg = update_fpo_cfg(agent_cfg, args_cli)
    return agent_cfg


def update_fpo_cfg(agent_cfg: FpoRslRlOnPolicyRunnerCfg, args_cli: argparse.Namespace):
    """Update configuration for FPO agent based on inputs."""
    if hasattr(args_cli, "seed") and args_cli.seed is not None:
        if args_cli.seed == -1:
            args_cli.seed = random.randint(0, 10000)
        agent_cfg.seed = args_cli.seed
    if args_cli.resume is not None:
        agent_cfg.resume = args_cli.resume
    if args_cli.load_run is not None:
        agent_cfg.load_run = args_cli.load_run
    if args_cli.checkpoint is not None:
        agent_cfg.load_checkpoint = args_cli.checkpoint
    if args_cli.run_name is not None:
        agent_cfg.run_name = args_cli.run_name
    if args_cli.experiment_name is not None:
        agent_cfg.experiment_name = args_cli.experiment_name
    if args_cli.logger is not None:
        agent_cfg.logger = args_cli.logger
    if agent_cfg.logger in {"wandb", "neptune"} and args_cli.log_project_name:
        agent_cfg.wandb_project = args_cli.log_project_name
        agent_cfg.neptune_project = args_cli.log_project_name

    algorithm_variant = getattr(args_cli, "algorithm", None)
    if algorithm_variant == "fpo":
        agent_cfg.policy.class_name = "ActorCritic"
        agent_cfg.algorithm.class_name = "FPO"
    elif algorithm_variant == "imf_fpo":
        agent_cfg.policy.class_name = "IMFActorCritic"
        agent_cfg.algorithm.class_name = "IMFFPO"
        # iMF is a fast-forward policy by construction.  More steps remain a
        # supported ablation and can be overridden through agent.policy.
        agent_cfg.policy.sampling_steps = 1
        # Do not silently reinterpret FPO's CFM endpoint proxies as mean-flow
        # KL/entropy estimates.  IMFFPO validates these safe defaults too.
        agent_cfg.algorithm.schedule = "fixed"
        agent_cfg.algorithm.knn_entropy_coef = 0.0
        if args_cli.experiment_name is None:
            agent_cfg.experiment_name = f"{agent_cfg.experiment_name}_imf_fpo"
    elif algorithm_variant == "pmf_fpo":
        agent_cfg.policy.class_name = "PMFActorCritic"
        agent_cfg.algorithm.class_name = "PMFFPO"
        # pMF is designed for a direct one-step x prediction.  More steps are
        # still available as a solver/NFE ablation via agent.policy overrides.
        agent_cfg.policy.sampling_steps = 1
        agent_cfg.algorithm.schedule = "fixed"
        agent_cfg.algorithm.knn_entropy_coef = 0.0
        if args_cli.experiment_name is None:
            agent_cfg.experiment_name = f"{agent_cfg.experiment_name}_pmf_fpo"
    elif algorithm_variant == "fsppo":
        agent_cfg.policy.class_name = "PMFActorCritic"
        agent_cfg.algorithm.class_name = "FSPPO"
        # The explicit F_theta(s, eps) penalty uses pMF's direct t=1 -> r=0
        # transport map, so it must match the one-NFE rollout policy.
        agent_cfg.policy.sampling_steps = 1
        agent_cfg.algorithm.schedule = "fixed"
        agent_cfg.algorithm.knn_entropy_coef = 0.0
        if args_cli.experiment_name is None:
            agent_cfg.experiment_name = f"{agent_cfg.experiment_name}_fsppo"

    return agent_cfg
