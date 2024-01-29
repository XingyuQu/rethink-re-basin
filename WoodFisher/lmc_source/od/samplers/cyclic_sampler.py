import numpy as np

from lmc_source.utils.logger import Logger


class CyclicWidthSampler:

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

        sample_s = []
        for i in range(self.n_layers):
            for j in range(1, self.widths[i] + 1):
                sample = self.n_layers * [None]
                sample[i] = j / self.widths[i]
                sample_s.append(sample)
        self.samples = np.array(sample_s)
        self.init_iter()

    def init_iter(self):
        np.random.shuffle(self.samples)
        self.samples_last = [None] * self.n_layers
        self.counter = 0
        self.iter = iter(self.samples.flatten())

    def __call__(self):
        try:
            next_sample = next(self.iter)
        except StopIteration:
            self.init_iter()
            next_sample = next(self.iter)

        if self.n_layers > 0:
            if self.counter == self.n_layers:
                self.counter = 0
            self.samples_last[self.counter] = next_sample
            self.counter += 1
        return next_sample
