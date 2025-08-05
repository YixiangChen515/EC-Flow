from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from gymnasium.spaces import Box

from metaworld.envs.asset_path_utils import full_v2_path_for
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import RenderMode, SawyerXYZEnv
from metaworld.types import InitConfigDict


class SawyerBlankEnvV2(SawyerXYZEnv):
    PAD_SUCCESS_MARGIN: float = 0.06
    TARGET_RADIUS: float = 0.08

    def __init__(
        self,
        render_mode: RenderMode | None = None,
        camera_name: str | None = None,
        camera_id: int | None = None,
        hand_init_pos: npt.NDArray[np.float32] | None = np.array((0, 0.6, 0.2), dtype=np.float32),
    ) -> None:
        hand_low = (-0.5, 0.40, 0.05)
        hand_high = (0.5, 1, 0.5)

        super().__init__(
            hand_low=hand_low,
            hand_high=hand_high,
            render_mode=render_mode,
            camera_name=camera_name,
            camera_id=camera_id,
        )

        self.init_config: InitConfigDict = {
            "hand_init_pos": hand_init_pos,
        }
        self.hand_init_pos = self.init_config["hand_init_pos"]

    @property
    def model_name(self) -> str:
        return full_v2_path_for("sawyer_xyz/sawyer_blank.xml")

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[npt.NDArray[np.float64], dict[str, Any]]:
        self.curr_path_length = 0
        self.reset_model()
        return None
    
    def reset_model(self) -> npt.NDArray[np.float64]:
        self._reset_hand()
        return self._get_obs()
    
    def _get_obs(self) -> npt.NDArray[np.float64]:
        return None
