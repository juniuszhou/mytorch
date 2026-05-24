import json

import ezkl

# https://github.com/zkonduit/ezkl/tree/main/examples/notebooks

# 1. Export your PyTorch/TF model to ONNX (example)
# Assume you have a trained model saved as "model.onnx"

# 2. Generate settings (circuit parameters)
settings_path = "settings.json"
ezkl.gen_settings("model.onnx", settings_path)

# Optional: Calibrate settings using sample data for better performance
data_path = "witness_data.json"  # input data for calibration
ezkl.calibrate_settings(data_path, "model.onnx", settings_path, "resources")

# 3. Compile the model into a circuit
compiled_model = "compiled_model.ezkl"
ezkl.compile_circuit("model.onnx", compiled_model, settings_path)

# 4. Prepare witness (input data + expected output)
# Example witness creation (you can also use ezkl helpers)
witness = {
    "input_data": [[1.0, 2.0, 3.0]],  # your input tensor(s)
    # ... other fields
}
with open("witness.json", "w") as f:
    json.dump(witness, f)

# 5. Setup keys (one-time)
vk_path = "vk.key"
pk_path = "pk.key"
ezkl.setup(compiled_model, vk_path, pk_path)  # or use gen_srs + setup

# 6. Generate proof
proof_path = "proof.json"
ezkl.prove(
    witness="witness.json",
    model=compiled_model,
    pk_path=pk_path,
    proof_path=proof_path,
    # transcript="evm", etc.
)

# 7. Verify the proof
assert ezkl.verify(proof_path, vk_path)

print("Proof generated and verified successfully!")
