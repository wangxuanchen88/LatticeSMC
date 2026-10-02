"""Compute counters (project rule 3). NFE counts denoiser forwards per sample row: a guided
step on N particles adds 2N. Reward and twist evaluations are counted separately and are not NFE."""


class Counter:
    def __init__(self, name):
        self.name = name
        self.n = 0

    def add(self, k=1):
        self.n += int(k)

    def reset(self):
        self.n = 0


class Counters:
    def __init__(self):
        self.nfe = Counter("nfe")
        self.reward_evals = Counter("reward_evals")
        self.twist_evals = Counter("twist_evals")

    def reset(self):
        for c in (self.nfe, self.reward_evals, self.twist_evals):
            c.reset()

    def as_dict(self):
        return {c.name: c.n for c in (self.nfe, self.reward_evals, self.twist_evals)}
