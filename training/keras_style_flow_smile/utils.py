class RunningLosses:

    def __init__(self, *labels):
        self.values = {
          l: 0. for l in labels
        }
        self.counter = 0

    def update(self, losses_pbar):
        self.values = {k: v + losses_pbar[k] for k, v in self.values.items()}
        self.counter += int(losses_pbar['count'])

    def get_losses(self):
        return list(
            map(lambda x: x[1] / max(self.counter, 1),
                self.values.items())
        )

    def reset(self):
        self.__init__()
