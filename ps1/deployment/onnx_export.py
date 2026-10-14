"""
ONNX Model Export & Verification Pipeline.
Exports OpenSetReIDNet to an optimized ONNX computational graph with dynamic batch dimensions.
Verifies numerical parity between PyTorch eager mode and ONNX Runtime.
"""

import os
import sys
import argparse
import yaml
import torch
import numpy as np

# Enable direct execution from any directory
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
root_dir = os.path.abspath(os.path.join(parent_dir, ".."))
for p in [parent_dir, root_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from models.open_set_reid_model import build_model
except ImportError:
    from ps1.models.open_set_reid_model import build_model


def export_to_onnx(
    model: torch.nn.Module,
    output_path: str = "./weights/open_set_reid.onnx",
    img_h: int = 256,
    img_w: int = 128,
    opset_version: int = 14
):
    """
    Exports feature extractor to ONNX format with dynamic batching.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    model.eval()

    # Wrap model to only expose extract_features during inference
    class FeatureExtractorWrapper(torch.nn.Module):
        def __init__(self, net):
            super().__init__()
            self.net = net

        def forward(self, x):
            return self.net.extract_features(x)

    export_wrapper = FeatureExtractorWrapper(model)
    dummy_input = torch.randn(1, 3, img_h, img_w, dtype=torch.float32)

    print(f"[ONNX Export] Exporting model to {output_path} (Opset {opset_version})...")
    torch.onnx.export(
        export_wrapper,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input_images"],
        output_names=["embeddings"],
        dynamic_axes={
            "input_images": {0: "batch_size"},
            "embeddings": {0: "batch_size"}
        }
    )
    print(f"[ONNX Export] Successfully saved ONNX model. File size: {os.path.getsize(output_path) / (1024*1024):.2f} MB")

    # Verify model with ONNX Runtime if available
    try:
        import onnxruntime as ort
        session = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])
        ort_inputs = {session.get_inputs()[0].name: dummy_input.numpy()}
        ort_outs = session.run(None, ort_inputs)[0]

        with torch.no_grad():
            torch_outs = export_wrapper(dummy_input).numpy()

        diff = np.max(np.abs(torch_outs - ort_outs))
        print(f"[ONNX Export] Numerical parity test: Max Absolute Difference = {diff:.6e}")
        assert diff < 1e-4, "High discrepancy between PyTorch and ONNX outputs!"
        print("[ONNX Export] Numerical verification PASSED!")
    except ImportError:
        print("[ONNX Export] onnxruntime not installed, skipping runtime verification.")


def main():
    parser = argparse.ArgumentParser(description="Export OpenSetReID to ONNX")
    parser.add_argument("--config", type=str, default="./configs/default_config.yaml")
    parser.add_argument("--checkpoint", type=str, default="./weights/best_openset_reid_model.pth")
    parser.add_argument("--output", type=str, default="./weights/open_set_reid.onnx")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    model = build_model(config)
    if os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(ckpt.get("model_state_dict", ckpt))
        print(f"[ONNX Export] Loaded checkpoint from {args.checkpoint}")
    else:
        print("[ONNX Export] Using initialized model architecture.")

    export_to_onnx(model, output_path=args.output)


if __name__ == "__main__":
    main()
