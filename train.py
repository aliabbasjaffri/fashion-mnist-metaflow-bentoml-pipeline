from torch import torch, nn
from torch.utils.data import DataLoader

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

    # Lists for visualization of loss and accuracy
    loss_list = []
    iteration_list = []
    accuracy_list = []

    # Lists for knowing class-wise accuracy
    predictions_list = []
    labels_list = []

    train_loader = get_loader(is_train_set=True)
    test_loader = get_loader(is_train_set=False)

    for epoch in range(num_epochs):
        for images, labels in train_loader:
            # Transfering images and labels to GPU if available
            _images, _labels = images.to(device), labels.to(device)

            train_images = Variable(_images.view(100, 1, 28, 28))
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

                    test_images = Variable(images.view(100, 1, 28, 28))
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
                print("Iteration: {}, Loss: {}, Accuracy: {}%".format(count, loss.data, accuracy))

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
    class_correct = [0. for _ in range(10)]
    total_correct = [0. for _ in range(10)]

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

    for i in range(10):
        print("Accuracy of {}: {:.2f}%".format(output_label(i), class_correct[i] * 100 / total_correct[i]))

    print((sum(class_correct) * 100) / sum(total_correct))


def train_epoch(
    model, optimizer, loss_function, train_loader, epoch, _device="cpu"
) -> None:
    # Mark training flag
    model.train()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(_device), targets.to(_device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, targets)
        loss.backward()
        optimizer.step()
        if batch_idx % 499 == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(inputs),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )


def __train__(
    epochs: int = 1, learning_rate: float = 1e-4, _device: str = "cpu"
) -> FashionMNISTConvnet:
    train_loader = get_loader(is_train_set=True)

    model = FashionMNISTConvnet()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        train_epoch(model, optimizer, loss_function, train_loader, epoch, _device)

    # mlflow.pytorch.log_model(model, "model")
    return model


def test_model(
    model: FashionMNISTConvnet, _test_loader: DataLoader = None, _device: str = "cpu"
) -> dict:
    _correct, _total = 0, 0

    if _test_loader is None:
        _test_loader = get_loader(is_train_set=False)

    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(_test_loader):
            inputs, targets = inputs.to(_device), targets.to(_device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            _total += targets.size(0)
            _correct += (predicted == targets).sum().item()

    # mlflow.log_metric("val_accuracy", (float(_correct) / _total) * 100)
    print((float(_correct) / _total) * 100)
    return {"correct": _correct, "total": _total}


if __name__ == "__main__":
    # model = train()
    # test(model=model)
    model = __train__(epochs=5)
    test_model(model)

