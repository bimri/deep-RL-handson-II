import gym


if __name__ == "__main__":
    env = gym.make("CartPole-v1")

    total_reward = 0.0
    total_steps = 0
    obs = env.reset()

    max_steps = 1000
    done = False
    while not done and total_steps < max_steps:
        action = env.action_space.sample()
        obs, reward, done, _ = env.step(action)[:4]
        total_reward += reward
        total_steps += 1

    print(f"Episode done in {total_steps} steps, total reward {total_reward:.2f}")
