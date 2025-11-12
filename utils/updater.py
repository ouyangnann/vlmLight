from utils.config import DIC_AGENTS
import pickle
import os
import time

class Updater:

    def __init__(self, cnt_round, dic_agent_conf, dic_traffic_env_conf, dic_path):

        self.cnt_round = cnt_round
        self.dic_path = dic_path
        self.dic_traffic_env_conf = dic_traffic_env_conf
        self.dic_agent_conf = dic_agent_conf
        self.agents = []
        self.sample_set_list = []
        self.sample_indexes = None

        for i in range(dic_traffic_env_conf['NUM_AGENTS']):
            agent_name = self.dic_traffic_env_conf["MODEL_NAME"]
            agent= DIC_AGENTS[agent_name](
                self.dic_agent_conf, self.dic_traffic_env_conf,
                self.dic_path, self.cnt_round, intersection_id=str(i))
            self.agents.append(agent)

    def load_sample_for_agents(self):
        start_time = time.time()
        missing_pattern = self.dic_traffic_env_conf["MISSING_PATTERN"]

        sample_base = "%s.pkl" % self.dic_traffic_env_conf["TRAFFIC_FILE"][:-5]
        sample_path = os.path.join(self.dic_path["PATH_TO_MEMO"], sample_base)

        # if sample file exists: load as before
        if os.path.exists(sample_path):
            sample_file = open(sample_path, "rb")
            if missing_pattern is not None:
                pattern, rate = missing_pattern.rsplit("_", 1)
                missing_pattern_path = os.path.join(self.dic_path["PATH_TO_MEMO"], pattern, "%s_%s.pkl" % (self.dic_traffic_env_conf["TRAFFIC_FILE"][:-5], pattern))
                missing_pattern_file = open(missing_pattern_path, "rb") if os.path.exists(missing_pattern_path) else None

            try:
                while True:
                    sample_set = pickle.load(sample_file)
                    if missing_pattern is not None and missing_pattern_file is not None:
                        mask_set = pickle.load(missing_pattern_file)
                    else:
                        mask_set = None
                    print("Samples loaded!")
                    # prepare_Xs_Y expected to consume a sample_set and optional mask_set
                    self.agents[0].prepare_Xs_Y(sample_set, mask_set)
            except EOFError:
                sample_file.close()
            return

        # fallback: no sample file on disk -> try online sampling for certain missing patterns
        print(f"Sample file not found: {sample_path}. Attempting online sampling fallback.")

        if missing_pattern is None:
            print("No missing_pattern specified and no memory file found. Skipping load_sample_for_agents.")
            return

        pattern, rate = missing_pattern.rsplit("_", 1)
        pattern = pattern.lower()

        # preferred: agent-provided online sampler
        if hasattr(self.agents[0], "generate_samples_online"):
            try:
                print("Using agent.generate_samples_online(...) to create samples.")
                # expected signature (num_samples, pattern, rate, save_path) - agent should implement accordingly
                num_samples = self.dic_traffic_env_conf.get("NUM_SAMPLES", 10)
                save_dir = self.dic_path["PATH_TO_MEMO"]
                self.agents[0].generate_samples_online(num_samples=num_samples, pattern=pattern, rate=rate, save_dir=save_dir)
                # after generation, try loading the created file
                if os.path.exists(sample_path):
                    with open(sample_path, "rb") as sf:
                        while True:
                            try:
                                sample_set = pickle.load(sf)
                                mask_set = None
                                if pattern in ("rm", "km"):
                                    # if agent generated masks next in file, try to load one more item
                                    try:
                                        mask_set = pickle.load(sf)
                                    except Exception:
                                        mask_set = None
                                self.agents[0].prepare_Xs_Y(sample_set, mask_set)
                            except EOFError:
                                break
                    return
            except Exception as e:
                print("agent.generate_samples_online failed:", e)

        # generic basic online sampling implementations for 'rm' and 'km'
        if pattern in ("rm", "km"):
            print(f"Attempting generic online sampling for pattern '{pattern}' with rate={rate}.")
            # try to collect episodes from agent's env if available
            env_attr = None
            for a in self.agents:
                if hasattr(a, "env"):
                    env_attr = a.env
                    break
                if hasattr(a, "traffic_env"):
                    env_attr = a.traffic_env
                    break

            if env_attr is None:
                print("No env found on agents; cannot perform generic online sampling. Please implement agent.generate_samples_online or provide memory files.")
                return

            num_episodes = self.dic_traffic_env_conf.get("NUM_SAMPLES", 10)
            run_counts = int(self.dic_traffic_env_conf.get("RUN_COUNTS", 3600))
            generated = []
            masks = []
            import random
            import numpy as np

            for ep in range(num_episodes):
                try:
                    obs = env_attr.reset()
                except TypeError:
                    obs = env_attr.reset()
                episode_samples = []
                for t in range(run_counts):
                    # try to use a policy if agent provides choose_action / act
                    action = None
                    agent0 = self.agents[0]
                    if hasattr(agent0, "choose_action"):
                        try:
                            action = agent0.choose_action(obs)
                        except Exception:
                            action = None
                    if action is None and hasattr(agent0, "act"):
                        try:
                            action = agent0.act(obs)
                        except Exception:
                            action = None
                    # fallback to env default action or zeros
                    if action is None:
                        try:
                            action = env_attr.default_action()
                        except Exception:
                            # best-effort: zeros
                            action = 0

                    try:
                        next_obs, reward, done, info = env_attr.step(action)
                    except Exception:
                        # some envs return different signatures
                        out = env_attr.step(action)
                        if len(out) == 4:
                            next_obs, reward, done, info = out
                        else:
                            next_obs = out[0]
                            reward = None
                            done = False
                            info = {}

                    episode_samples.append((obs, action, reward, next_obs, done))
                    obs = next_obs
                    if done:
                        break

                # create mask according to pattern
                if pattern == "rm":
                    # random missing: drop each element with probability = float(rate)
                    try:
                        r = float(rate)
                    except Exception:
                        r = 0.1
                    mask = []
                    for s in episode_samples:
                        # conservative: mask per timestep scalar (True=observed)
                        mask.append(np.random.rand() >= r)
                    masks.append(mask)
                elif pattern == "km":
                    # k-missing: remove contiguous chunks. interpret rate as k (number of contiguous missing frames)
                    try:
                        k = int(rate)
                    except Exception:
                        k = 1
                    mask = [True] * len(episode_samples)
                    if k >= 1 and len(mask) > k:
                        start = random.randint(0, max(0, len(mask) - k))
                        for i in range(start, min(start + k, len(mask))):
                            mask[i] = False
                    masks.append(mask)

                generated.append(episode_samples)

            # call prepare_Xs_Y for each generated sample + mask
            for s, m in zip(generated, masks):
                self.agents[0].prepare_Xs_Y(s, m)

            # save generated samples to disk so next run can reuse them
            try:
                os.makedirs(self.dic_path["PATH_TO_MEMO"], exist_ok=True)
                with open(sample_path, "wb") as sf:
                    for s, m in zip(generated, masks):
                        pickle.dump(s, sf)
                        pickle.dump(m, sf)
                print(f"Generated samples saved to {sample_path}")
            except Exception as e:
                print("Failed to save generated samples:", e)
            return

        # unsupported pattern
        print(f"Unsupported missing pattern '{pattern}'. Cannot perform online sampling fallback.")

    def update_network(self, i):
        self.agents[i].train_network()
        self.agents[i].save_network("round_{0}_inter_{1}".format(self.cnt_round, self.agents[i].intersection_id))
        if self.dic_traffic_env_conf["MODEL_NAME"] in ["BEAR", "TD3_BC"]:
            self.agents[i].save_network_bar("round_{0}_inter_{1}".format(self.cnt_round, self.agents[i].intersection_id))

    def update_network_for_agents(self):
        for i in range(self.dic_traffic_env_conf['NUM_AGENTS']):
            self.update_network(i)
