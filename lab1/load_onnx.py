import onnx
import onnxruntime as ort
import numpy
import torch

onnx_model = onnx.load("wine_model.onnx")
onnx.checker.check_model(onnx_model)

onnx_input = torch.tensor([7.0, 0.27, 0.36, 20.7, 0.045, 45.0, 170.0, 1.0010, 3.00, 0.45, 8.8])
actual = 6.
ort_sess = ort.InferenceSession("wine_model.onnx")
output = ort_sess.run(None, {'input': onnx_input})

print(f'Predicted: "{output}", Actual: "{actual}"')
