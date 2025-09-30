# scripts/test_random.py
import time

import numpy as np

from envs.robobo_env import RoboboEnv


def main():
    env = RoboboEnv(ip="localhost", action_mode="continuous")
    try:
        obs, info = env.reset()
        print("Obs inicial:", obs)

        for t in range(150):
            action = np.clip(np.random.normal(0.0, 0.35, size=2), -1.0, 1.0)
            obs, rew, term, trunc, inf = env.step(action)
            print(f"[{t:03d}] rew={rew:.3f} term={term} trunc={trunc} dist={obs[0]:.3f}")
            time.sleep(0.1)
            if term or trunc:
                obs, info = env.reset()
                print("Obs reinicio:", obs)
    finally:
        env.close()
        time.sleep(0.5)


if __name__ == "__main__":
    main()
