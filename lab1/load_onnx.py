import onnx
import onnxruntime as ort
import numpy as np

onnx_model = onnx.load("wine_model.onnx")
onnx.checker.check_model(onnx_model)

onnx_input = np.array([7.0, 0.27, 0.36, 20.7, 0.045, 45.0, 170.0, 1.0010, 3.00, 0.45, 8.8],
                      [6.3, 0.30, 0.34, 1.6, 0.049, 14.0, 132.0, 0.9940, 3.30, 0.49, 9.5]).astype(np.float32)
actual = [6., 6.]
ort_sess = ort.InferenceSession("wine_model.onnx")
output = ort_sess.run(None, {'input': onnx_input})

print(f'Predicted: "{output}", Actual: "{actual}"')
