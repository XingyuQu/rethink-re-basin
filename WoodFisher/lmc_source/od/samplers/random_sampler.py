import numpy as np

from lmc_source.utils.logger import Logger


class RandomSampler:

    def __init__(self, model):
        self.model = model
        self.create_samples()
        self.counter = 0
        Logger.get().info(f"Widths: {self.widths}")
        Logger.get().info(f"ODLayers: {self.names}")

    def create_samples(self):
        self.n_layers = 0
        self.widths = []
        self.names = []
        for name, m in self.model.named_modules():
            if hasattr(m, 'od_layer') and m.od_layer:
                self.n_layers += 1
                self.widths.append(m.layer_width)
                self.names.append(name)
        self.samples_last = [None] * self.n_layers

        self.sample_s = []
        for i in range(self.n_layers):
            sample = [None] * self.widths[i]
            for j in range(1, self.widths[i] + 1):
                sample[j-1] = j / self.widths[i]
            sample = np.array(sample)
            np.random.shuffle(sample)
            self.sample_s.append(sample)
        self.pts = [0] * self.n_layers

    def __call__(self):
        next_sample = None
        if self.n_layers > 0:
            if self.counter == self.n_layers:
                self.counter = 0
            next_sample = self.sample_s[self.counter][self.pts[self.counter]]
            self.samples_last[self.counter] = next_sample
            self.pts[self.counter] += 1
            if self.pts[self.counter] == self.widths[self.counter]:
                self.pts[self.counter] = 0
                np.random.shuffle(self.sample_s[self.counter])
            self.counter += 1
        return next_sample
