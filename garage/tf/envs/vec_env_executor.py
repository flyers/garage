import numpy as np

from garage.tf.misc import tensor_utils
from garage.misc.overrides import overrides


class VecEnvExecutor:
    def __init__(self, envs, max_path_length):
        self.envs = envs
        self._action_space = envs[0].action_space
        self._observation_space = envs[0].observation_space
        self.ts = np.zeros(len(self.envs), dtype='int')
        self.max_path_length = max_path_length

    def step(self, action_n):
        all_results = [env.step(a) for (a, env) in zip(action_n, self.envs)]
        obs, rewards, dones, env_infos = list(
            map(list, list(zip(*all_results))))
        dones = np.asarray(dones)
        rewards = np.asarray(rewards)
        self.ts += 1
        if self.max_path_length is not None:
            dones[self.ts >= self.max_path_length] = True
        for (i, done) in enumerate(dones):
            if done:
                obs[i] = self.envs[i].reset()
                self.ts[i] = 0
        return obs, rewards, dones, tensor_utils.stack_tensor_dict_list(
            env_infos)

    def reset(self):
        results = [env.reset() for env in self.envs]
        self.ts[:] = 0
        return results

    @property
    def num_envs(self):
        return len(self.envs)

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._observation_space

    def close(self):
        pass


class AirsimVecEnvExecutor(VecEnvExecutor):

    @overrides
    def reset(self):
        all_results = [env.reset() for env in self.envs]
        obses, env_infos = list(
            map(list, list(zip(*all_results))))
        self.ts[:] = 0
        return obses, env_infos

    @overrides
    def step(self, action_n):
        all_results = [env.step(a) for (a, env) in zip(action_n, self.envs)]
        obs, rewards, dones, env_infos = list(
            map(list, list(zip(*all_results))))
        dones = np.asarray(dones)
        rewards = np.asarray(rewards)
        self.ts += 1
        if self.max_path_length is not None:
            dones[self.ts >= self.max_path_length] = True
        for (i, done) in enumerate(dones):
            if done:
                obs[i], env_infos[i] = self.envs[i].reset()
                self.ts[i] = 0
        return obs, rewards, dones, tensor_utils.stack_tensor_dict_list(
            env_infos)