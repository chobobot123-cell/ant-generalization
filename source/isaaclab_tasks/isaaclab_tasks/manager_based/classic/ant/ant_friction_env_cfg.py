"""Ant training environment with randomized contact friction."""

import torch

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils import configclass

from .ant_env_cfg import AntEnvCfg, EventCfg


def randomize_ant_friction(
    env,
    env_ids,
    friction_range=(0.1, 1.5),
    num_buckets=64,
):
    """Assign one friction value per Ant, preserving restitution."""
    robot = env.scene["robot"]
    view = robot.root_physx_view

    # PhysX material properties are edited with CPU tensors.
    materials = view.get_material_properties().clone()

    if env_ids is None:
        ids = torch.arange(env.num_envs, dtype=torch.long, device="cpu")
    else:
        ids = env_ids.to(device="cpu", dtype=torch.long)

    low, high = friction_range
    buckets = torch.linspace(
        low, high, num_buckets,
        dtype=materials.dtype, device="cpu"
    )
    choices = torch.randint(
        num_buckets, (len(ids),), device="cpu"
    )
    friction = buckets[choices]

    # Channels: static friction, dynamic friction, restitution.
    # All collision shapes of one Ant receive the same friction.
    materials[ids, :, 0] = friction[:, None]
    materials[ids, :, 1] = friction[:, None]
    view.set_material_properties(materials, ids)

    # Verify the assigned values through the physics view.
    actual = view.get_material_properties()
    expected = friction[:, None].expand_as(actual[ids, :, 0])
    if not torch.allclose(actual[ids, :, 0], expected):
        raise RuntimeError("Static friction assignment failed.")
    if not torch.allclose(actual[ids, :, 1], expected):
        raise RuntimeError("Dynamic friction assignment failed.")

    print(
        f"[FRICTION] Applied to {len(ids)} Ants; "
        f"robot material range={friction.min().item():.3f}"
        f"..{friction.max().item():.3f}; buckets={num_buckets}"
    )


@configclass
class AntFrictionEventCfg(EventCfg):
    randomize_friction = EventTerm(
        func=randomize_ant_friction,
        mode="startup",
        params={
            "friction_range": (0.1, 1.5),
            "num_buckets": 64,
        },
    )


@configclass
class AntFrictionEnvCfg(AntEnvCfg):
    events: AntFrictionEventCfg = AntFrictionEventCfg()
