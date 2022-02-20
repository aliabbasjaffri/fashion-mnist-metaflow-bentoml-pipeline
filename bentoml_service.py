import numpy as np
import bentoml
from datasource import FASHION_MNIST_CLASSES
from PIL.Image import Image as PILImage
from bentoml.io import Image, JSON


fashion_mnist_runner = bentoml.pytorch.load_runner(
    "fashion_mnist_classifier", name="fashion_mnist_runner"
)
svc = bentoml.Service(name="fashion_mnist_service", runners=[fashion_mnist_runner])


@svc.api(input=Image(), output=JSON())
async def predict_image(image: PILImage):
    assert isinstance(image, PILImage)
    image = image.convert(mode="L").resize((28, 28))
    arr = np.array(image) / 255.0
    assert arr.shape == (28, 28)

    arr = np.expand_dims(arr, 0).astype("float32")
    output_tensor = await fashion_mnist_runner.async_run(arr)

    results = {}

    for i, value in zip(FASHION_MNIST_CLASSES, output_tensor):
        results[i] = value.item()

    return results
