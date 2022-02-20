from typing import BinaryIO, List
import numpy as np
import bentoml
from PIL.Image import Image as PILImage
import torch
from torchvision import transforms

from bentoml.io import Image, NumpyNdarray, JSON

FASHION_MNIST_CLASSES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


fashion_mnist_runner = bentoml.pytorch.load_runner(
    "fashion_mnist_classifier", name="fashion_mnist_runner", predict_fn_name="predict"
)
svc = bentoml.Service(name="fashion_mnist_service", runners=[fashion_mnist_runner])


@svc.api(input=Image(), output=NumpyNdarray(dtype="str"))
async def predict_image(f: PILImage):
    assert isinstance(f, PILImage)
    arr = np.array(f) / 255.0
    assert arr.shape == (28, 28)

    arr = np.expand_dims(arr, 0).astype("float32")
    output_tensor = await fashion_mnist_runner.async_run(arr)
    print(output_tensor)
    return output_tensor.numpy()


# @bentoml.env(pip_packages=['torch', 'numpy', 'torchvision', 'scikit-learn'])
# class FashionMNISTClassifier(bentoml.BentoService):
#
#     @bentoml.utils.cached_property  # reuse transformer
#     def transform(self):
#         return transforms.Compose([transforms.CenterCrop((29, 29)), transforms.ToTensor()])
#
#     @bentoml.api(input=FileInput(), output=JsonOutput(), batch=True)
#     def predict(self, file_streams: List[BinaryIO]) -> List[str]:
#         img_tensors = []
#         for fs in file_streams:
#             img = Image.open(fs).convert(mode="L").resize((28, 28))
#             img_tensors.append(self.transform(img))
#         outputs = self.artifacts.classifier(torch.stack(img_tensors))
#         _, output_classes = outputs.max(dim=1)
#
#         return [FASHION_MNIST_CLASSES[output_class] for output_class in output_classes]