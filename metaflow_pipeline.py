from metaflow import FlowSpec, step

from train import train, test


class ExperimentalPyTorchFlow(FlowSpec):

    @step
    def start(self):
        self.epochs = 1
        self.next(self.model_train)

    @step
    def model_train(self):
        self.model = train(num_epochs=self.epochs)
        self.next(self.model_test)

    @step
    def model_test(self):
        self.outputs = test(self.model)
        self.next(self.end)

    @step
    def end(self):
        print(self.outputs)


if __name__ == '__main__':
    obj = ExperimentalPyTorchFlow()

