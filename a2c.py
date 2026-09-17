import gymnasium as gym
import torch, torch.nn as nn
from torch.distributions import Categorical

# ---- hyperparameters -------------------------------------------------------
N_ENVS, T_MAX = 8, 5          # parallel envs, rollout length
GAMMA, LR = 0.99, 7e-4
VF_COEF, ENT_COEF = 0.5, 0.01
TOTAL_STEPS = 500_000

# ---- network ---------------------------------------------------------------
class ActorCritic(nn.Module):
    def __init__(self, obs_dim=4, n_actions=2, hidden=64):
        super().__init__()
        self.trunk  = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(),
                                    nn.Linear(hidden, hidden), nn.Tanh())
        self.actor  = nn.Linear(hidden, n_actions)   # logits over actions
        self.critic = nn.Linear(hidden, 1)           # V(s)

    def forward(self, obs):
        h = self.trunk(obs)
        return Categorical(logits=self.actor(h)), self.critic(h).squeeze(-1)

# ---- setup -----------------------------------------------------------------
envs = gym.vector.SyncVectorEnv([lambda: gym.make("CartPole-v1")] * N_ENVS)
net = ActorCritic()
opt = torch.optim.Adam(net.parameters(), lr=LR)

obs, _ = envs.reset(seed=0)
obs = torch.as_tensor(obs, dtype=torch.float32)
ep_returns, cur_ret, steps = [], torch.zeros(N_ENVS), 0

# ---- training loop ---------------------------------------------------------
while steps < TOTAL_STEPS:
    # 1. rollout: T_MAX steps in all N_ENVS envs with the current policy
    obs_buf, act_buf, rew_buf, done_buf = [], [], [], []
    for _ in range(T_MAX):
        with torch.no_grad():
            dist, _ = net(obs)
            action = dist.sample()
        next_obs, reward, term, trunc, _ = envs.step(action.numpy())
        done = torch.as_tensor(term | trunc, dtype=torch.float32)

        obs_buf.append(obs); act_buf.append(action)
        rew_buf.append(torch.as_tensor(reward, dtype=torch.float32)); done_buf.append(done)

        cur_ret += torch.as_tensor(reward)
        for i in range(N_ENVS):
            if done[i]:
                ep_returns.append(cur_ret[i].item()); cur_ret[i] = 0
        obs = torch.as_tensor(next_obs, dtype=torch.float32)
        steps += N_ENVS

    # 2. returns: bootstrap from V(s_T), then work backwards
    with torch.no_grad():
        _, R = net(obs)                      # V of last state, shape [N_ENVS]
    returns = []
    for r, d in zip(reversed(rew_buf), reversed(done_buf)):
        R = r + GAMMA * (1 - d) * R
        returns.insert(0, R)

    # flatten [T, N] -> [T*N]
    b_obs = torch.cat(obs_buf); b_act = torch.cat(act_buf); b_ret = torch.cat(returns)

    # 3. advantages and 4. one gradient step on the combined loss
    dist, values = net(b_obs)
    adv = (b_ret - values).detach()          # no gradient flows through A
    policy_loss = -(dist.log_prob(b_act) * adv).mean()
    value_loss = (b_ret - values).pow(2).mean()
    entropy = dist.entropy().mean()
    loss = policy_loss + VF_COEF * value_loss - ENT_COEF * entropy

    opt.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(net.parameters(), 0.5)
    opt.step()

    if steps % 10_000 < N_ENVS * T_MAX and ep_returns:
        print(f"steps {steps:>7}  mean return (last 20 eps): {sum(ep_returns[-20:]) / len(ep_returns[-20:]):.1f}")