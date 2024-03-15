import pickle
import onpolicy
import attr
import habitat
import habitat_sim
import habitat_sim.utils
import magnum as mn
import numpy as np
from habitat.sims.habitat_simulator.actions import (
    HabitatSimV0ActionSpaceConfiguration,
)
from habitat.tasks.nav.nav import SimulatorTaskAction
from habitat.sims.habitat_simulator.actions import HabitatSimActions
actuation_noise_fwd = pickle.load(open(onpolicy.__path__[0] + "/envs/habitat/model/noise_models/actuation_noise_fwd.pkl", 'rb'))
actuation_noise_right = pickle.load(open(onpolicy.__path__[0] + "/envs/habitat/model/noise_models/actuation_noise_right.pkl", 'rb'))
actuation_noise_left = pickle.load(open(onpolicy.__path__[0] + "/envs/habitat/model/noise_models/actuation_noise_left.pkl", 'rb'))


@attr.s(auto_attribs=True, slots=True)
class CustomActuationSpec:
    action: int


def _custom_action_impl(
        scene_node: habitat_sim.SceneNode,
        delta_dist: float,  # in metres
        delta_dist_angle: float,  # in degrees
        delta_angle: float  # in degrees
):
    forward_ax = (
            np.array(scene_node.absolute_transformation().rotation_scaling())
            @ habitat_sim.geo.FRONT
    )
    move_angle = np.deg2rad(delta_dist_angle)

    rotation = habitat_sim.utils.quat_from_angle_axis(
        move_angle, habitat_sim.geo.UP
    )
    move_ax = habitat_sim.utils.quat_rotate_vector(rotation, forward_ax)

    scene_node.translate_local(move_ax * delta_dist)
    scene_node.rotate_local(mn.Deg(delta_angle), habitat_sim.geo.UP)


def _noisy_action_impl(
        scene_node: habitat_sim.SceneNode,
        action: int
):
    if action == 0:  ## Forward
        dx, dy, do = actuation_noise_fwd.sample()[0][0]
    elif action == 1:  ## Right
        dx, dy, do = actuation_noise_right.sample()[0][0]
    elif action == 2:  ## Left
        dx, dy, do = actuation_noise_left.sample()[0][0]

    delta_dist = np.sqrt(dx ** 2 + dy ** 2)
    delta_dist_angle = np.rad2deg(np.arctan2(-dy, dx))
    delta_angle = -do

    _custom_action_impl(scene_node, delta_dist, delta_dist_angle, delta_angle)


@habitat_sim.registry.register_move_fn(body_action=True)
class NoisyHumanForward(habitat_sim.SceneNodeControl):
    def __call__(
            self,
            scene_node: habitat_sim.SceneNode,
            actuation_spec: CustomActuationSpec,
    ):
        _noisy_action_impl(
            scene_node,
            actuation_spec.action,
        )


@habitat_sim.registry.register_move_fn(body_action=True)
class NoisyHumanRight(habitat_sim.SceneNodeControl):
    def __call__(
            self,
            scene_node: habitat_sim.SceneNode,
            actuation_spec: CustomActuationSpec,
    ):
        _noisy_action_impl(
            scene_node,
            actuation_spec.action,
        )


@habitat_sim.registry.register_move_fn(body_action=True)
class NoisyHumanLeft(habitat_sim.SceneNodeControl):
    def __call__(
            self,
            scene_node: habitat_sim.SceneNode,
            actuation_spec: CustomActuationSpec,
    ):
        _noisy_action_impl(
            scene_node,
            actuation_spec.action,
        )


@habitat.registry.register_task_action
class NoisyRight(SimulatorTaskAction):
    def _get_uuid(self, *args, **kwargs) -> str:
        return "NOISY_RIGHT"

    def step(self, *args, **kwargs):
        return HabitatSimActions.NOISY_RIGHT

@habitat.registry.register_task_action
class NoisyLeft(SimulatorTaskAction):
    def _get_uuid(self, *args, **kwargs) -> str:
        return "NOISY_LEFT"

    def step(self, *args, **kwargs):
        return HabitatSimActions.NOISY_LEFT

@habitat.registry.register_task_action
class NoisyForward(SimulatorTaskAction):
    def _get_uuid(self, *args, **kwargs) -> str:
        return "NOISY_FORWARD"

    def step(self, *args, **kwargs):
        return HabitatSimActions.NOISY_FORWARD


@habitat.registry.register_action_space_configuration
class CustomActionSpaceConfiguration(HabitatSimV0ActionSpaceConfiguration):
    
    def get(self):
        config = super().get()

        config[HabitatSimActions.NOISY_FORWARD] = habitat_sim.ActionSpec(
            "noisy_human_forward",
            CustomActuationSpec(0),
        )
        config[HabitatSimActions.NOISY_RIGHT] = habitat_sim.ActionSpec(
            "noisy_human_right",
            CustomActuationSpec(1),
        )
        config[HabitatSimActions.NOISY_LEFT] = habitat_sim.ActionSpec(
            "noisy_human_left",
            CustomActuationSpec(2),
        )

        return config
