import numpy as np
import gym

class AlphaBetaAgent:
    def __init__(self, env, depth=2):
        self.env = env
        self.depth = depth

    def get_action(self, obs):
        _, best_action = self.alphabeta(self.env, obs, self.depth, -np.inf, np.inf, True)

        if best_action is None:
            best_action = np.array([0, 0])

        return best_action

    def alphabeta(self, env, obs, depth, alpha, beta, maximizing_player):
        if depth == 0:
            return self.evaluate(obs), np.array([0, 0])

        best_value = -np.inf if maximizing_player else np.inf
        best_action = None

        possible_actions = [
            np.array([0, 0]),   # no action
            np.array([1, 0]),   # right
            np.array([-1, 0]),  # left
            np.array([0, 1]),   # jump
            np.array([1, 1]),   # jump + right
            np.array([-1, 1])   # jump + left
        ]

        for action in possible_actions:
            try:
                env_copy = gym.make("SlimeVolley-v0")
                env_copy.reset()

                joint_action = np.hstack([action, env_copy.action_space.sample()])
                _, new_obs, _, done, _ = env_copy.step(joint_action)

                value, _ = self.alphabeta(env_copy, new_obs, depth - 1, alpha, beta, not maximizing_player)

                if maximizing_player:
                    if value > best_value:
                        best_value = value
                        best_action = action
                    alpha = max(alpha, best_value)
                else:
                    if value < best_value:
                        best_value = value
                        best_action = action
                    beta = min(beta, best_value)

                if beta <= alpha:
                    break  # Prune
            except:
                continue

        return best_value, best_action

    def evaluate(self, obs):
        agent_x = obs[0]
        ball_x = obs[4]
        ball_vx = obs[5]
        agent_score = obs[24]
        opponent_score = obs[25]

        dist = abs(agent_x - ball_x)
        direction_bonus = 5 if ball_vx > 0 else -5
        score_reward = (agent_score - opponent_score) * 100

        return -dist + direction_bonus + score_reward
