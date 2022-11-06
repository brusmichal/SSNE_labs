import onnx
import onnxruntime as ort
import numpy as np
import pandas as pd

onnx_model = onnx.load("wine_model.onnx")
onnx.checker.check_model(onnx_model)


df = pd.read_csv("valid.csv")
df.head()
x_val = df.values[:, :-1]
y_val = df.values[:, -1]

onnx_input = x_val[:10, :].astype(np.float32)
target = y_val[:10]
ort_sess = ort.InferenceSession("wine_model.onnx")
output = ort_sess.run(None, {'input': onnx_input})

print("\nOcena                    Ocena")
print(f'przewidywana:            prawdziwa:\n')
for o, t in zip(output[0], target):
    print(f"{o[0]:.2f}                     {t}")
