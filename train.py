import mlflow
from torch import torch, nn
from datasource import get_loader, output_label
from model import FashionMNISTConvnet
from torch.autograd import Variable
import matplotlib.pyplot as plt


def train(num_epochs: int = 1, learning_rate: float = 0.001) -> FashionMNISTConvnet:
    # moving model to gpu if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = FashionMNISTConvnet()
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    error = nn.CrossEntropyLoss()

    print(model)

    count = 0
    batch_size: int = 100

    # Lists for visualization of loss and accuracy
    loss_list = []
    iteration_list = []
    accuracy_list = []

    # Lists for knowing class-wise accuracy
    predictions_list = []
    labels_list = []

    mlflow.log_params({"epochs": num_epochs})
    mlflow.log_params({"learning_rate": learning_rate})
    mlflow.log_params({"batch_size": learning_rate})

    train_loader = get_loader(is_train_set=True)
    test_loader = get_loader(is_train_set=False)

    for epoch in range(num_epochs):
        for images, labels in train_loader:
            # Transfering images and labels to GPU if available
            _images, _labels = images.to(device), labels.to(device)

            train_images = Variable(_images.view(batch_size, 1, 28, 28))
            train_labels = Variable(_labels)

            # Forward pass
            outputs = model(train_images)
            loss = error(outputs, train_labels)

            # Initializing a gradient as 0 so there is no mixing of gradient among the batches
            # Propagating the error backward
            # Optimizing the parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            count += 1

            if not (count % 10):  # It's same as "if count % 10 == 0"
                total = 0
                correct = 0

                for __images, __labels in test_loader:
                    _images, _labels = __images.to(device), __labels.to(device)
                    labels_list.append(_labels)

                    test_images = Variable(images.view(batch_size, 1, 28, 28))
                    outputs = model(test_images)

                    predictions = torch.max(outputs, 1)[1].to(device)
                    predictions_list.append(predictions)
                    correct += (predictions == labels).sum()

                    total += len(labels)

                accuracy = correct * 100 / total
                loss_list.append(loss.data)
                iteration_list.append(count)
                accuracy_list.append(accuracy)

            if not (count % 50):
                print(
                    "Iteration: {}, Loss: {}, Accuracy: {}%".format(
                        count, loss.data, accuracy
                    )
                )

    plt.plot(iteration_list, loss_list)
    plt.xlabel("No. of Iteration")
    plt.ylabel("Loss")
    plt.title("Iterations vs Loss")
    plt.show()

    plt.plot(iteration_list, accuracy_list)
    plt.xlabel("No. of Iteration")
    plt.ylabel("Accuracy")
    plt.title("Iterations vs Accuracy")
    plt.show()

    return model


def test(model: FashionMNISTConvnet) -> dict:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    class_correct = [0.0 for _ in range(10)]
    total_correct = [0.0 for _ in range(10)]

    test_loader = get_loader(is_train_set=False)

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            test = Variable(images)
            outputs = model(test)
            predicted = torch.max(outputs, 1)[1]
            c = (predicted == labels).squeeze()

            for i in range(100):
                label = labels[i]
                class_correct[label] += c[i].item()
                total_correct[label] += 1

    test_results = {}
    for i in range(10):
        print(
            "Accuracy of {}: {:.2f}%".format(
                output_label(i), class_correct[i] * 100 / total_correct[i]
            )
        )
        test_results[output_label(i)] = class_correct[i] * 100 / total_correct[i]
        mlflow.log_metric(
            f"validation_accuracy_{output_label(i)}",
            class_correct[i] * 100 / total_correct[i],
        )

    print((sum(class_correct) * 100) / sum(total_correct))

    return test_results


if __name__ == "__main__":
    model = train()
    test(model=model)
