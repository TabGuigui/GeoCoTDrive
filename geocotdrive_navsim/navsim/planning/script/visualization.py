import os
from pathlib import Path
import json

import hydra
from hydra.utils import instantiate
from hydra.core.global_hydra import GlobalHydra
import numpy as np
import matplotlib.pyplot as plt

from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import SceneFilter, SensorConfig
from tqdm import tqdm

from navsim.planning.training.dataset import Dataset

SPLIT = "test"  
FILTER = "navtest"

from navsim.visualization.plots import plot_bev_with_agent
from navsim.agents.abstract_agent import AbstractAgent
CONFIG_PATH = "config/pdm_scoring"
CONFIG_NAME = "default_run_pdm_score"


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def init_agent(cfg) -> None:

    print("initlize agent")
    cfg_agent = cfg.agent
    agent: AbstractAgent = instantiate(cfg_agent)
    agent.initialize()
    agent.to("cuda")

    print("initlize scene")
    GlobalHydra.instance().clear()

    hydra.initialize(config_path="./config/common/train_test_split/scene_filter")
    cfg2 = hydra.compose(config_name=FILTER)
    scene_filter: SceneFilter = instantiate(cfg2)
    openscene_data_root = Path(os.getenv("OPENSCENE_DATA_ROOT"))

    scene_loader = SceneLoader(
        sensor_blobs_path=Path(cfg.sensor_blobs_path),
        data_path=Path(cfg.navsim_log_path),
        scene_filter=scene_filter,
        sensor_config=agent.get_sensor_config(),
        load_image_path=True
    )

    tokens = scene_loader.tokens
    # for token in tokens:
    tokens = np.random.choice(tokens, 10)
    for token in tokens:
        scene = scene_loader.get_scene_from_token(token)
        frame_idx = scene.scene_metadata.num_history_frames - 1
        fig, ax = plot_bev_with_agent(scene, agent)
        fig.text(x=0.05, y=0.05, s=token,  fontsize=10,color="blue",ha="left", va="bottom")
        plt.savefig(f"/data/visual_nips/geocotdrive_{token}.jpg")
        print(f"/data/visual_nips/geocotdrive_{token}.jpg")

if __name__ == "__main__":
    init_agent()