"""Analytic ramp for controlled Ant evaluation."""

import math
import numpy as np
import trimesh

from isaaclab.utils import configclass
from isaaclab.terrains import TerrainGeneratorCfg, SubTerrainBaseCfg

from .ant_env_cfg import AntEnvCfg


def make_ramp_mesh(difficulty, cfg):
    length, width = cfg.size
    start = length / 2.0 + cfg.start_distance
    end = start + cfg.ramp_length

    if not (0.0 < start < end < length):
        raise ValueError("Ramp must fit inside the terrain.")

    slope = math.tan(math.radians(cfg.angle_deg))
    xs = [0.0, start, end, length]
    vertices = []

    for x in xs:
        run = min(max(x - start, 0.0), cfg.ramp_length)
        z = slope * run
        vertices.extend([(x, 0.0, z), (x, width, z)])

    # Upward-facing triangles.
    faces = []
    for i in range(len(xs) - 1):
        a = 2 * i
        faces.extend([
            (a, a + 2, a + 1),
            (a + 1, a + 2, a + 3),
        ])

    mesh = trimesh.Trimesh(
        vertices=np.asarray(vertices, dtype=np.float64),
        faces=np.asarray(faces, dtype=np.int64),
        process=False,
    )
    origin = np.array([length / 2.0, width / 2.0, 0.0])
    return [mesh], origin


@configclass
class AntRampTerrainCfg(SubTerrainBaseCfg):
    function = make_ramp_mesh
    angle_deg: float = 5.0
    start_distance: float = 2.0
    ramp_length: float = 220.0


def ramp_clearance(env):
    robot = env.scene["robot"]
    position = robot.data.root_pos_w
    origins = env.scene.env_origins

    generator = env.cfg.scene.terrain.terrain_generator
    cfg = generator.sub_terrains["ramp"]

    # This analytic formula requires one shared ramp tile.
    if generator.num_rows != 1 or generator.num_cols != 1:
        raise RuntimeError("Ramp evaluation requires a single terrain tile.")

    x = position[:, 0] - origins[:, 0]
    y = position[:, 1] - origins[:, 1]

    # Reject boundary exits rather than counting them as locomotion failures.
    outside = (
        (x.abs() > generator.size[0] / 2.0 - 2.0)
        | (y.abs() > generator.size[1] / 2.0 - 2.0)
    )
    if outside.any().item():
        raise RuntimeError(
            "Ant reached the ramp map boundary. "
            "Enlarge the map before reporting this evaluation."
        )

    run = (x - cfg.start_distance).clamp(
        min=0.0, max=cfg.ramp_length
    )
    slope = math.tan(math.radians(cfg.angle_deg))
    ground_z = origins[:, 2] + slope * run
    return position[:, 2] - ground_z


def ramp_height_observation(env):
    return ramp_clearance(env).unsqueeze(-1)


def ramp_height_termination(env, minimum_height):
    return ramp_clearance(env) < minimum_height


@configclass
class AntRampEvalEnvCfg(AntEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = TerrainGeneratorCfg(
            seed=20261020,
            size=(512.0, 128.0),
            border_width=0.0,
            num_rows=1,
            num_cols=1,
            curriculum=False,
            use_cache=False,
            sub_terrains={
                "ramp": AntRampTerrainCfg(proportion=1.0),
            },
        )
        self.scene.terrain.max_init_terrain_level = 0

        self.observations.policy.base_height.func = ramp_height_observation
        self.observations.policy.base_height.params = {}

        self.terminations.torso_height.func = ramp_height_termination
        self.terminations.torso_height.params = {"minimum_height": 0.31}
