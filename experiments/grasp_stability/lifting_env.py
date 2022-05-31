import numpy as np

class lifting():
    def __init__(self,
                 robot,
                 plane,
                 obj,
                 objStartPos,
                 objStartOrientation,
                 p,
                 reward_scale=1.0,
                 reward_shaping=False,
                 ):

        self.robot = robot
        self.obj = obj
        self.objStartPos = objStartPos
        self.objStartOrientation = objStartOrientation
        self.plane = plane

        self.num_joints = p.getNumJoints(self.robot)
        self.ee_index = 7               # TODO: check that this is the index
        self.ee_orientation = np.asarray([0.0, 0.0, 0.0, 1.0])

        self.totalNumJoints = p.getNumJoints(self.robot)
        self.jd = [0.01] * self.totalNumJoints              # allow to tune the IK solution using joint damping factors

        # reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        self.p = p

        print()

    def step(self, action):
        """
            The action is the desired EE position (not orientation at the moment) and gripper width to grasp
        """
        desired_ee_pos = action
        self.robot.go(desired_ee_pos, ori=None, width=None, wait=False, gripForce=20)


    def reward(self):
        """
                Reward function for the task.
                Sparse un-normalized reward:
                    - a discrete reward of 2.25 is provided if the cube is lifted
                Un-normalized summed components if using reward shaping:
                    - Reaching: in [0, 1], to encourage the arm to reach the cube
                    - Grasping: in {0, 0.25}, non-zero if arm is grasping the cube
                    - Lifting: in {0, 1}, non-zero if arm has lifted the cube
                The sparse reward only consists of the lifting component.
                Note that the final reward is normalized and scaled by
                reward_scale / 2.25 as well so that the max score is equal to reward_scale
                Args:
                    action (np array): [NOT USED]
                Returns:
                    float: reward value
                """
        reward = 0.0

        # sparse completion reward
        if self._check_success():
            reward = 2.25

        # use a shaping reward
        elif self.reward_shaping:

            # reaching reward
            obj_pos = self.p.getBasePositionAndOrientation(self.obj)
            gripper_pos = self.robot.get_ee_pose()
            dist = np.linalg.norm(gripper_pos - obj_pos)
            reaching_reward = 1 - np.tanh(10.0 * dist)
            reward += reaching_reward

            # TODO: understand how to check the grasp!
            # # grasping reward
            # if self._check_grasp(gripper=self.robots[0].gripper, object_geoms=self.cube):
            #     reward += 0.25

        # Scale reward if requested
        if self.reward_scale is not None:
            reward *= self.reward_scale / 2.25

        return reward

    def reset(self):
        # Reset environments: robot, and object
        self.robot.init_robot()
        self.p.resetBasePositionAndOrientation(self.obj, self.objStartPos, self.objStartOrientation)
        print()

    def _check_success(self):
        """
        From ROBOSIUTE
        Check if cube has been lifted.
        Returns:
            bool: True if cube has been lifted
        """
        obj_height = self.p.getBasePositionAndOrientation(self.obj)[2]      # TODO: sure that this is the correct one?
        table_height = self.p.getBasePositionAndOrientation(self.plane)[2]

        # cube is higher than the table top above a margin
        return obj_height > table_height + 0.04
