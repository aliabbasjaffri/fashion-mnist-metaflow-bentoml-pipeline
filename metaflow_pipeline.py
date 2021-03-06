import bentoml
import mlflow
from metaflow import FlowSpec, step
from train import train, test


class FashionMNISTFlow(FlowSpec):
    def __init__(self, use_cli=True):
        super().__init__(use_cli)
        self.epochs = None
        self.model = None
        self.test_result = None

    @step
    def start(self):
        self.epochs = 5
        self.next(self.model_train)

    # @resources(memory=8196, cpu=8)
    @step
    def model_train(self):
        self.model = train(num_epochs=self.epochs)
        mlflow.pytorch.log_model(self.model, "fashion_mnist_classifier")
        self.next(self.model_test)

    @step
    def model_test(self):
        self.test_result = test(self.model)
        self.next(self.bentoml_deployment)

    @step
    def bentoml_deployment(self):
        model_name = "fashion_mnist_classifier"
        tag = bentoml.pytorch.save(
            name=model_name, model=self.model, metadata=self.test_result
        )
        print(tag)
        self.next(self.end)

    @step
    def end(self):
        print(self.test_result)


if __name__ == "__main__":
    mlflow.set_experiment("fashion-mnist-classifier")
    obj = FashionMNISTFlow()
