Allowlisted strategies: `baseline`, `ort_basic`, `ort_extended`, `ort_all`.
Unknown names are rejected. There is no free-form graph rewrite API.

API surface:

- `python -m compiler emit-fixture`
- `python -m compiler analyze <model.onnx>`
- `python -m compiler recommend <model.onnx>`
- `python -m compiler compile <model.onnx> --backend ort_cpu|tensorrt`
- `python -m compiler baseline`
- `python -m compiler probe`
