"""Small rough terrains for Ant training and development."""

import isaaclab.terrains as terrain_gen
from isaaclab.utils import configclass

from .ant_env_cfg import AntEnvCfg
from .ant_friction_env_cfg import AntFrictionEnvCfg


def rough_patch(proportion, amplitude):
    return terrain_gen.HfRandomUniformTerrainCfg(
        proportion=proportion,
        noise_range=(-amplitude, amplitude),
        noise_step=0.005,
        downsampled_scale=0.4,
        border_width=0.4,
    )


def make_terrain(seed, preview=False):
    if preview:
        sub_terrains = {
            "rough_2cm": rough_patch(1.0, 0.02),
        }
        num_cols = 1
    else:
        sub_terrains = {
            "flat": rough_patch(0.25, 0.0),
            "rough_1cm": rough_patch(0.375, 0.01),
            "rough_2cm": rough_patch(0.375, 0.02),
        }
        num_cols = 8

    return terrain_gen.TerrainGeneratorCfg(
        seed=seed,
        size=(256.0, 64.0),
        border_width=10.0,
        num_rows=1,
        num_cols=num_cols,
        horizontal_scale=0.2,
        vertical_scale=0.005,
        slope_threshold=0.75,
        use_cache=False,
        # Fixed column layout; no performance-based promotion.
        curriculum=True,
        sub_terrains=sub_terrains,
    )


def configure_rough_scene(cfg, preview=False):
    cfg.scene.terrain.terrain_type = "generator"
    cfg.scene.terrain.terrain_generator = make_terrain(
        seed=20261003 if preview else 20261002,
        preview=preview,
    )
    cfg.scene.terrain.max_init_terrain_level = 0

    # Mesh terrain origins determine the initial robot locations.
    cfg.scene.terrain.debug_vis = False

    print(
        "[ROUGH] preview=" + str(preview)
        + "; patch size=(256m, 64m); height samples up to +/-2cm"
    )


@configclass
class AntRoughTrainEnvCfg(AntFrictionEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        configure_rough_scene(self, preview=False)


@configclass
class AntRoughPreviewEnvCfg(AntEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        configure_rough_scene(self, preview=True)
