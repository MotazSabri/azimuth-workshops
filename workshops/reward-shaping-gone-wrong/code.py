# workshops/reward-shaping-gone-wrong/code.py
#
# Trains one agent three times on three reward functions and shows that the
# mis-specified one wins the game it was given while losing the job.
#
# REINFORCE with a baseline rather than PPO: the whole workshop turns on
# reading the reward function, and PPO puts a clipping objective, GAE and a
# value head between the reader and that idea. The papers explain PPO; this
# spine keeps the thing under test in view.

# --8<-- [start:setup]
import azimuth_nb as azimuth

env = azimuth.setup(SLUG, lang=LANG, profile=PROFILE)
# --8<-- [end:setup]


# --8<-- [start:rewards]
# THE THREE REWARDS. Read them before running anything.
#
# LunarLander's observation is
#   [x, y, vx, vy, angle, angular_velocity, leg_left_contact, leg_right_contact]
# so `-(x**2 + y**2) ** 0.5` is distance to the pad, which sits at the origin.

GAMMA = env.cfg["gamma"]


def potential(obs):
    """Φ(s): closer to the pad is higher. Used by BOTH shaped rewards, so the
    difference between them is purely HOW it is applied, not what it measures."""
    return -((obs[0] ** 2 + obs[1] ** 2) ** 0.5)


def reward_honest(reward, obs, next_obs, done):
    """The environment's own reward, untouched. The control."""
    return reward


def reward_shaped(reward, obs, next_obs, done):
    """POTENTIAL-BASED shaping (Ng, Harada & Russell, 1999).

    Rewards the CHANGE in potential, γ·Φ(s′) − Φ(s). Because the added terms
    telescope over an episode, they cannot change which policy is optimal —
    they only make the gradient less sparse. This is the safe form, and it is
    the only shaping with a proof attached.
    """
    return reward + 10.0 * (GAMMA * potential(next_obs) - potential(obs))


def reward_hacked(reward, obs, next_obs, done):
    """The mis-specification. Rewards the STATE, not the change.

    This is not a strawman. "Give it points for being close to the target" is
    the single most natural thing to write, it reads as obviously helpful, and
    it is wrong for a reason that is invisible until you watch the agent: a
    bonus paid every step for BEING somewhere is a bonus for STAYING there.
    Landing ends the episode, and ending the episode ends the income.
    """
    return reward + 10.0 * potential(next_obs)


REWARDS = {
    "honest": reward_honest,
    "shaped": reward_shaped,
    "hacked": reward_hacked,
}

if env.lang == "ar":
    print("ثلاث دوال مكافأة:", " · ".join(REWARDS))
    print("أيّها تظنّه سيهبط أكثر؟ قرّر قبل التشغيل.")
else:
    print("three reward functions:", " · ".join(REWARDS))
    print("which do you think lands most often? decide before you run.")
# --8<-- [end:rewards]


# --8<-- [start:train]
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn


class Policy(nn.Module):
    """A small categorical policy with a value baseline."""

    def __init__(self, n_obs, n_actions, hidden):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(n_obs, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh()
        )
        self.actor = nn.Linear(hidden, n_actions)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, obs):
        h = self.body(obs)
        return self.actor(h), self.critic(h).squeeze(-1)


def train(reward_fn, seed):
    """One agent, one reward function. Same seed for all three, so the only
    thing that differs between the runs is the function itself."""
    torch.manual_seed(seed)
    environment = gym.make("LunarLander-v3")
    net = Policy(
        environment.observation_space.shape[0], environment.action_space.n, env.cfg["hidden"]
    )
    optimizer = torch.optim.Adam(net.parameters(), lr=env.cfg["learningRate"])

    for episode in range(env.cfg["episodes"]):
        obs, _ = environment.reset(seed=seed + episode)
        log_probs, values, rewards = [], [], []
        done = False
        while not done:
            obs_t = torch.as_tensor(obs, dtype=torch.float32)
            logits, value = net(obs_t)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            next_obs, raw_reward, terminated, truncated, _ = environment.step(action.item())
            done = terminated or truncated

            log_probs.append(dist.log_prob(action))
            values.append(value)
            # THE ONLY LINE THAT DIFFERS between the three agents.
            rewards.append(reward_fn(raw_reward, obs, next_obs, done))
            obs = next_obs

        returns, running = [], 0.0
        for r in reversed(rewards):
            running = r + GAMMA * running
            returns.append(running)
        returns = torch.tensor(list(reversed(returns)), dtype=torch.float32)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        values_t = torch.stack(values)
        advantage = returns - values_t.detach()
        loss = -(torch.stack(log_probs) * advantage).sum() + nn.functional.mse_loss(
            values_t, returns, reduction="sum"
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if episode % 100 == 0:
            print(f"  {episode:5d}  return {sum(rewards):8.1f}")

    environment.close()
    return net


agents = {}
for name, fn in REWARDS.items():
    print(f"\n{name}:")
    agents[name] = train(fn, env.cfg["seed"])

episodes_run = env.cfg["episodes"] * len(REWARDS)
# --8<-- [end:train]


# --8<-- [start:scoreboard]
def evaluate(net, reward_fn, seed, n):
    """Two numbers per agent, deliberately.

    `shaped` is what the training loop optimised. `landed` is the job. Keeping
    them apart is the entire point: an agent can move one without the other,
    and reporting a single 'score' would hide exactly the effect being taught.
    """
    environment = gym.make("LunarLander-v3")
    shaped_total, landed = 0.0, 0
    for i in range(n):
        obs, _ = environment.reset(seed=10_000 + seed + i)
        done = False
        while not done:
            with torch.no_grad():
                logits, _ = net(torch.as_tensor(obs, dtype=torch.float32))
            action = int(torch.argmax(logits))
            next_obs, raw_reward, terminated, truncated, _ = environment.step(action)
            shaped_total += reward_fn(raw_reward, obs, next_obs, terminated or truncated)
            obs = next_obs
            done = terminated or truncated
        # Both legs down and the lander at rest: gymnasium pays +100 on a
        # successful landing, so the final raw reward is the honest verdict.
        if terminated and raw_reward >= 100:
            landed += 1
    environment.close()
    return shaped_total / n, landed / n


n_eval = env.cfg["evalEpisodes"]
results = {}
for name, net in agents.items():
    # Every agent is scored on the HACKED reward as well as its own, so the
    # columns are comparable — otherwise each agent is graded on its own exam.
    own_shaped, landed = evaluate(net, REWARDS[name], env.cfg["seed"], n_eval)
    hacked_shaped, _ = evaluate(net, reward_hacked, env.cfg["seed"], n_eval)
    results[name] = {"own": own_shaped, "hacked_scale": hacked_shaped, "landed": landed}

header = f"{'agent':10}{'own reward':>14}{'hacked reward':>16}{'landed':>10}"
print("\n" + header)
print("-" * len(header))
for name, r in results.items():
    print(f"{name:10}{r['own']:>14.1f}{r['hacked_scale']:>16.1f}{r['landed']:>9.0%}")

# The card's thumbnail, and the finding in one picture: the bar that wins the
# written objective is the bar that never lands. A table says it; a chart makes
# it impossible to miss at index size.
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6, 3))
names = list(results)
ax.bar(names, [results[n]["landed"] * 100 for n in names], color=["#2a9d8f", "#457b9d", "#e76f51"])
for i, n in enumerate(names):
    ax.text(
        i, results[n]["landed"] * 100 + 2, f"{results[n]['landed']:.0%}", ha="center", fontsize=9
    )
ax.set_ylabel("landed (%)" if env.lang == "en" else "نسبة الهبوط (%)")
ax.set_ylim(0, 105)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
plt.show()

honest_landing = results["honest"]["landed"]
hacked_landing = results["hacked"]["landed"]
# A DIFFERENCE, not a ratio. Returns here are negative — the shaping term is
# -distance summed over the episode — and a ratio over signed quantities is
# meaningless: -742.9 / 1372.6 came out as -0.54 and read as "the exploit lost"
# when it had in fact won by 630 points. Ratios need a positive denominator
# and a meaningful zero; neither holds for a return.
shaped_advantage = results["hacked"]["hacked_scale"] - results["honest"]["hacked_scale"]
# Landing rates ARE ratios of counts: bounded, non-negative, zero means zero.
landing_ratio = hacked_landing / max(honest_landing, 1e-9)
# --8<-- [end:scoreboard]


# --8<-- [start:watch]
# One trajectory from the hacked agent, summarised. The scoreboard says it
# scores well and lands rarely; this says what it is doing instead.
environment = gym.make("LunarLander-v3")
obs, _ = environment.reset(seed=99)
altitudes, steps, done = [], 0, False
while not done and steps < 1000:
    with torch.no_grad():
        logits, _ = agents["hacked"](torch.as_tensor(obs, dtype=torch.float32))
    obs, _, terminated, truncated, _ = environment.step(int(torch.argmax(logits)))
    altitudes.append(float(obs[1]))
    steps += 1
    done = terminated or truncated
environment.close()

# Compare against an honest episode, and let the NUMBERS say what happened.
# The first version of this cell asserted hovering — "income every step" —
# and was exactly wrong: `potential` is -distance, so it is always negative,
# the bonus is a per-step TAX, and the fastest way to stop paying it is to end
# the episode. The agent did not learn to loiter. It learned to die.
environment = gym.make("LunarLander-v3")
obs, _ = environment.reset(seed=99)
honest_steps, done = 0, False
while not done and honest_steps < 1000:
    with torch.no_grad():
        logits, _ = agents["honest"](torch.as_tensor(obs, dtype=torch.float32))
    obs, _, terminated, truncated, _ = environment.step(int(torch.argmax(logits)))
    honest_steps += 1
    done = terminated or truncated
environment.close()

if env.lang == "ar":
    print(f"المخترِق: {steps} خطوة · ارتفاع وسيط {np.median(altitudes):.2f}")
    print(f"الأمين:  {honest_steps} خطوة")
else:
    print(f"hacked: {steps} steps · median altitude {np.median(altitudes):.2f}")
    print(f"honest: {honest_steps} steps")

if steps < honest_steps * 0.7:
    verdict_en = (
        "The exploit is a SHORT episode. `potential` is negative everywhere, so the "
        "bonus is a tax charged every step, and the cheapest policy is to stop "
        "paying it — end the episode. The agent is not confused; it found the fastest "
        "exit from a reward you wrote."
    )
    verdict_ar = (
        "الثغرة حلقة قصيرة. دالة الجهد سالبة في كل مكان، فالمكافأة ضريبة تُجبى كل "
        "خطوة، وأرخص سياسة أن تكفّ عن دفعها — أي أن تنهي الحلقة. الوكيل ليس مرتبكاً؛ "
        "بل وجد أسرع مخرج من مكافأة كتبتَها أنت."
    )
elif steps > honest_steps * 1.3:
    verdict_en = (
        "The exploit is a LONG episode: it loiters where the bonus is largest and "
        "never risks the landing. Same mechanism, opposite sign."
    )
    verdict_ar = (
        "الثغرة حلقة طويلة: يتسكّع حيث المكافأة أكبر ولا يخاطر بالهبوط قط. الآلية "
        "ذاتها بإشارة معاكسة."
    )
else:
    verdict_en = "Episode lengths are similar — look at where it spends its altitude instead."
    verdict_ar = "أطوال الحلقات متقاربة — انظر إلى أين ينفق ارتفاعه بدلاً من ذلك."

print("\n" + (verdict_ar if env.lang == "ar" else verdict_en))
# --8<-- [end:watch]


# --8<-- [start:fix_it]
# YOUR TURN.
#
# Convert the proximity bonus to potential-based form and retrain. The safe
# shaping is already written above as `reward_shaped` — the exercise is to
# understand WHY it is safe, then answer what it actually bought you.
#
# Compare against the numbers above, and look at episodes-to-competence, not
# just the final landing rate.
YOUR_REWARD = reward_shaped  # try your own

fixed = train(YOUR_REWARD, env.cfg["seed"])
fixed_shaped, fixed_landed = evaluate(fixed, YOUR_REWARD, env.cfg["seed"], n_eval)
print(
    f"yours: landed {fixed_landed:.0%}  ·  honest {honest_landing:.0%}  ·  hacked {hacked_landing:.0%}"
)
# --8<-- [end:fix_it]


# --8<-- [start:verify]
# The control first. A comparison against an agent that never learned the task
# is not a comparison, and 0/0 quietly satisfies "the exploit lands less".
control_ok = env.check("honest-agent-works", honest_landing)
if not control_ok:
    if env.lang == "ar":
        print("  الشاهد لم يتعلّم الهبوط — ارفع episodes، ولا تخفض العتبة.")
    else:
        print("  the control never learned to land — raise `episodes`, do not lower the bar.")

hacked_ok = env.check("hacked-scores-higher", shaped_advantage)
landing_ok = env.check("hacked-lands-less", landing_ratio)
# --8<-- [end:verify]


# --8<-- [start:finish]
receipt = env.receipt()
# --8<-- [end:finish]
