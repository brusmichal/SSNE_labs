import onnx
import onnxruntime as ort
import numpy as np
import pandas as pd

onnx_model = onnx.load("wine_model.onnx")
onnx.checker.check_model(onnx_model)

df = pd.read_csv("valid.csv")
x_val = df.values[:, :-1]
y_val = df.values[:, -1]
print(x_val.shape, y_val.shape)

onnx_input = x_val.sample(n=10).astype(np.float32)

actual = 6.
ort_sess = ort.InferenceSession("wine_model.onnx")
output = ort_sess.run(None, {'input': onnx_input})

print(f'Predicted: "{output}", Actual: "{actual}"')
