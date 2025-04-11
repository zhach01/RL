import os  # <-- as requested
import gym
from gym import spaces
import numpy as np
import random
import matplotlib.pyplot as plt
import torch

# Import your updated ArmModel that provides the new muscle-space functions.
from Robot_model import ArmModel


class IntegratedMuscleControlEnv(gym.Env):
    """
    A 2-DoF environment for RL that uses muscle activations as direct controls,
    with *no* activation dynamics and *no* fatigue.
    """

    def __init__(
        self,
        dt=0.01,
        max_steps=100,
        seed=None,
        curriculum_level=1,
        enable_domain_randomization=False,
        success_threshold=0.02,
        use_optimization=True,
    ):
        super(IntegratedMuscleControlEnv, self).__init__()

        self.dt = dt
        self.max_steps = max_steps
        self.step_count = 0
        self.curriculum_level = curriculum_level
        self.enable_domain_randomization = enable_domain_randomization
        self.success_threshold = success_threshold
        self.use_optimization = use_optimization

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            print(f"[Env] Random seed set to: {seed}")

        # Instantiate the new ArmModel (2-DoF, 6 muscles)
        self.robot = ArmModel(nd=2, version="V0", use_gravity=1)
        self.robot.Fmax = np.array([1000] * 6, dtype=float)
        # Optionally set explicit joint limits for the environment
        self.robot.min_joint_positions = -np.pi * np.ones(2)
        self.robot.max_joint_positions = np.pi * np.ones(2)

        ### CHANGED: Removed dynamic activation and fatigue parameters.

        # Define observation space (22 dimensions now).
        # Obs = [q(2), q_dot(2), x_ee(2), dot_ee(2), lm(6), lmd(6), target(2)] = 22
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(22,), dtype=np.float32)

        # Action space: muscle activation commands for 6 muscles (range [0,1])
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(6,), dtype=np.float32)

        # External state: joint angles (q) and velocities (q_dot).
        self.state_ext = np.zeros(4)  # [q1, q2, dq1, dq2]

        # Sample an initial target in the reachable workspace
        self.x_target = self._sample_target()

        # Last-info dictionary for debugging
        self.last_info = {}

        # (NEW) Keep track of distance in previous step for reward shaping
        self.prev_dist = None

        print("[Env] IntegratedMuscleControlEnv (no activation dynamics, no fatigue) initialized.")

    def _sample_target(self):
        """
        Sample a target position using polar coordinates, with optional
        domain randomization / curriculum-based radius fraction.
        """
        L1 = self.robot.L1
        L2 = self.robot.L2
        r_min = abs(L1 - L2) + 0.1
        r_max = L1 + L2 - 0.1

        fraction = max(0.1, min(1.0, self.curriculum_level * 0.2))
        r = np.random.uniform(r_min, r_min + fraction * (r_max - r_min))
        theta = np.random.uniform(0, 2 * np.pi)
        candidate = np.array([r * np.cos(theta), r * np.sin(theta)])

        # If candidate is outside allowed workspace, sample in a smaller rectangle.
        x_min, x_max, y_min, y_max = self.robot.allowed_workspace_bounds()
        if not (x_min <= candidate[0] <= x_max and y_min <= candidate[1] <= y_max):
            shrink = fraction
            x_range = (x_max - x_min) * shrink
            y_range = (y_max - y_min) * shrink
            center_x = (x_min + x_max) / 2.0
            center_y = (y_min + y_max) / 2.0
            candidate = np.array(
                [
                    np.random.uniform(center_x - x_range / 2, center_x + x_range / 2),
                    np.random.uniform(center_y - y_range / 2, center_y + y_range / 2),
                ]
            )
        return candidate

    def reset(self):
        """
        Reset the environment:
          - Clear step count
          - Possibly randomize link lengths
          - Sample new target
          - Initialize state (q, q_dot)
          - Initialize the ArmModel's trajectory logging
        """
        self.step_count = 0

        # Optional domain randomization on link lengths:
        if self.enable_domain_randomization:
            scale1 = np.random.uniform(0.95, 1.05)
            scale2 = np.random.uniform(0.95, 1.05)
            self.robot.L1 *= scale1
            self.robot.L2 *= scale2

        self.x_target = self._sample_target()

        # Small random initial state near zero angles
        self.state_ext[:2] = np.random.uniform(-0.3, 0.3, 2)
        self.state_ext[2:4] = np.random.uniform(-0.1, 0.1, 2)

        # Initialize the ArmModel's trajectory logging.
        q = self.state_ext[:2].copy()
        q_dot = self.state_ext[2:4].copy()
        self.robot.collect_trajectory_info(
            q=q,
            q_dot=q_dot,
            x_des=self.x_target,
            step=0,
            dt=self.dt,
            initialize=True,
        )

        # We'll store prev_dist = None so we can set it on first step
        self.prev_dist = None

        return self._get_observation()

    def _get_observation(self):
        """
        Build the environment's observation from the current state:
          [q(2), q_dot(2), x_ee(2), dot_ee(2), muscle_lengths(6), muscle_vels(6), target(2)]
        """
        q = self.state_ext[:2]
        q_dot = self.state_ext[2:4]

        x_ee = np.array(self.robot.forward_kinematics(q)).flatten()
        dot_ee = np.array(self.robot.forward_speed_kinematics(q, q_dot)).flatten()

        # Evaluate muscle lengths/velocities.
        muscle_lengths = self.robot.muscle_lengths_func(*q).flatten()
        muscle_vels = self.robot.muscle_velocities_func(q, q_dot).flatten()

        return np.concatenate(
            [
                q,                 # 2
                q_dot,             # 2
                x_ee,              # 2
                dot_ee,            # 2
                muscle_lengths,    # 6
                muscle_vels,       # 6
                self.x_target,     # 2
            ]
        ).astype(np.float32)

    def _compute_muscle_forces(self, u_cmd, q, q_dot):
        """
        Directly call the muscle_space_controller with the given action (u_cmd),
        returning muscle forces and resulting joint torque.
        """
        # We skip any activation or fatigue updates since there's no dynamic or fatigue now.
        q_des = np.array(self.robot.inverse_kinematics(self.x_target)).flatten()
        # Assume v_des is defined (e.g., desired end-effector velocity; here we use zeros)
        v_des = np.zeros_like(self.x_target)

        # Compute the desired joint velocities using the inverse speed kinematics function.
        q_dot_des =  np.array((self.robot.inverse_speed_kinematics(q, v_des, damping=0.01).detach().cpu().numpy().flatten())).flatten()

        tau = self.robot.muscle_space_controller(
            q=q,
            q_dot=q_dot,
            q_des=q_des,
            dot_q_des=q_dot_des,
            dt=self.dt,
            return_torque=True,
            use_optimization=self.use_optimization,
            u_cmd=u_cmd,  # <-- direct muscle activation
        )

        # Infer muscle forces from the resulting torque. Negative sign depends on your sign convention:
        R_num = self.robot.R_func(*q)
        fm = -np.linalg.pinv(R_num.T) @ tau

        return fm, tau
    

    def step(self, action):
        """
        Apply the muscle activation commands (action), compute the dynamics, update the environment.
        """
        self.step_count += 1

        # Action = muscle activation command in [0,1]^6
        u_cmd = np.clip(action, 0.0, 1.0)
        q = self.state_ext[:2]
        q_dot = self.state_ext[2:4]

        # 1) Compute muscle forces and resulting joint torque
        fm, tau = self._compute_muscle_forces(u_cmd, q, q_dot)

        # 2) Forward dynamics: ddq = M^-1 (tau - (C+G))
        M = np.array(self.robot.mass_matrix_func(*q), dtype=float)
        C = np.array(self.robot.coriolis_forces_func(q, q_dot), dtype=float).ravel()
        G = np.array(self.robot.gravitational_forces_func(*q), dtype=float).ravel()
        ddq = np.linalg.solve(M, tau - (C + G))

        # 3) Integrate to get next state
        q_dot_new = q_dot + ddq * self.dt
        q_new = q + q_dot_new * self.dt

        # 4) Enforce environment joint/velocity limits
        q_new = np.clip(q_new, self.robot.min_joint_positions, self.robot.max_joint_positions)
        q_dot_new = np.clip(q_dot_new, -self.robot.joint_vel_limits, self.robot.joint_vel_limits)

        # 5) Update environment state
        self.state_ext[:2] = q_new
        self.state_ext[2:4] = q_dot_new

        # 6) Build next observation
        obs = self._get_observation()
        x_ee = obs[4:6]
        dist = np.linalg.norm(x_ee - self.x_target)

        # 7a) Basic immediate penalty for distance + muscle usage
        force_penalty = 0.005 * np.linalg.norm(fm)
        activation_penalty = 0.5 * np.sum(u_cmd**2)
        immediate_reward = - (dist + force_penalty + activation_penalty)

        # 7b) Shaping reward to encourage distance reduction
        if self.prev_dist is None:
            shaping_reward = 0.0
        else:
            shaping_reward = 0.1 * (self.prev_dist - dist)
        reward = immediate_reward + shaping_reward

        # Add a success bonus if within threshold
        if dist < self.success_threshold:
            reward += 100.0

        self.prev_dist = dist

        # 8) For debug: muscle velocities/accelerations at the new state
        muscle_vels = self.robot.muscle_velocities_func(q_new, q_dot_new)
        muscle_accels = self.robot.muscle_acceleration_func(q_new, q_dot_new, ddq)

        # 9) Log debugging info
        self.last_info = {
            "raw_action": u_cmd,
            "current_state": np.concatenate([q_new, q_dot_new, ddq]),
            "muscle_forces": fm,
            "muscle_velocities": muscle_vels,
            "muscle_accelerations": muscle_accels,
            "tau": tau,
            "joint_torque": tau,
            "end_effector": x_ee,
            "target": self.x_target,
            "pos_error": dist,
            "immediate_reward": immediate_reward,
            "shaping_reward": shaping_reward,
        }

        # 10) Collect trajectory info in the ArmModel for playback
        self.robot.collect_trajectory_info(
            q=q_new,
            q_dot=q_dot_new,
            x_des=self.x_target,
            step=self.step_count,
            dt=self.dt,
            ddq=ddq,
            torque=tau,
        )

        done = (self.step_count >= self.max_steps) or (dist < self.success_threshold)
        return obs, reward, done, self.last_info

    def render(self, mode="human"):
        """
        Rely on the ArmModel's built-in draw_model method.
        """
        q = self.state_ext[:2]
        self.robot.draw_model(q, in_deg=False, info=self.last_info)

    def close(self):
        """
        Clean up any resources (close figure windows, etc.)
        """
        plt.close("all")
        print("[Env] close() called; all figures closed.")


# --------------------------------------------------------------------------
# Example usage/testing
# --------------------------------------------------------------------------
if __name__ == "__main__":
    env = IntegratedMuscleControlEnv()
    obs = env.reset()
    done = False

    # Simple loop: random actions until done
    while not done:
        action = env.action_space.sample()  # random muscle activations
        obs, reward, done, info = env.step(action)
        env.render()

    print("[Main] Episode done. Playing trajectory from ArmModel...")

    # Play back the trajectory just recorded, using the built-in ArmModel method
    env.robot.play_trajectory(playback_duration=2.0, decimation=2)

    env.close()
