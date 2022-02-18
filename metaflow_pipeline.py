from metaflow import FlowSpec, step

from train import __train__, test_model


class ExperimentalPyTorchFlow(FlowSpec):

    @step
    def start(self):
        self.epochs = 1
        self.next(self.model_train)

    @step
    def model_train(self):
        self.model = __train__(epochs=self.epochs)
        self.next(self.model_test)

    @step
    def model_test(self):
        self.outputs = test_model(self.model)
        self.next(self.end)

    @step
    def end(self):
        print(self.outputs)


if __name__ == '__main__':
    obj = ExperimentalPyTorchFlow()

