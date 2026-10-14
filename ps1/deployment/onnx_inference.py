"""
Real-Time ONNX Runtime Inference & Latency Profiler.
Benchmarking inference speed (ms/frame and FPS) under CPU, CUDA, and TensorRT providers.
Demonstrates sub-10ms real-time single-backbone execution.
"""

import time
import argparse
import numpy as np
import cv2


class ONNXReIDInferenceEngine:
    """
    Production-ready real-time inference engine using ONNX Runtime.
    """
    def __init__(self, model_path: str = "./weights/open_set_reid.onnx", device: str = "cpu"):
        import onnxruntime as ort

        providers = ["CPUExecutionProvider"]
        if device == "cuda":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        elif device == "tensorrt":
            providers = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]

        # Configure session options for high throughput
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 4

        self.session = ort.InferenceSession(model_path, sess_options=sess_options, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        # Normalization constants (ImageNet)
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

    def preprocess(self, img_bgr: np.ndarray, target_h: int = 256, target_w: int = 128) -> np.ndarray:
        """
        Preprocesses a BGR cropped bounding box into normalized NCHW float32 tensor.
        """
        # Resize to network input shape
        resized = cv2.resize(img_bgr, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        # Convert BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        # Normalize
        norm = (rgb - self.mean) / self.std
        # Transpose HWC -> CHW and add batch dimension -> NCHW
        tensor = np.transpose(norm, (2, 0, 1))[np.newaxis, :, :, :].astype(np.float32)
        return tensor

    def extract_embedding(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        Extracts L2-normalized 512-D embedding from a single crop.
        """
        inp_tensor = self.preprocess(img_bgr)
        outputs = self.session.run([self.output_name], {self.input_name: inp_tensor})
        embedding = outputs[0][0]  # [512]
        # Ensure exact unit norm
        norm = np.linalg.norm(embedding)
        return embedding / max(norm, 1e-8)

    def profile_latency(self, iterations: int = 200, warmup: int = 20):
        """
        Profiles latency distribution (mean, p50, p95, p99, FPS).
        """
        print(f"\n[Profiler] Running {iterations} benchmark iterations (warmup: {warmup})...")
        dummy_crop = np.random.randint(0, 255, (256, 128, 3), dtype=np.uint8)

        # Warmup
        for _ in range(warmup):
            _ = self.extract_embedding(dummy_crop)

        latencies = []
        for _ in range(iterations):
            t0 = time.perf_counter()
            _ = self.extract_embedding(dummy_crop)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)  # ms

        latencies = np.array(latencies)
        mean_lat = np.mean(latencies)
        median_lat = np.median(latencies)
        p95_lat = np.percentile(latencies, 95)
        p99_lat = np.percentile(latencies, 99)
        fps = 1000.0 / mean_lat

        print("=" * 55)
        print("          REAL-TIME INFERENCE LATENCY BENCHMARK")
        print("=" * 55)
        print(f" Mean Latency    : {mean_lat:.2f} ms")
        print(f" Median (p50)    : {median_lat:.2f} ms")
        print(f" 95th Percentile : {p95_lat:.2f} ms")
        print(f" 99th Percentile : {p99_lat:.2f} ms")
        print(f" Throughput      : {fps:.1f} FPS")
        print("=" * 55)
        return {
            "mean_ms": mean_lat,
            "p50_ms": median_lat,
            "p95_ms": p95_lat,
            "p99_ms": p99_lat,
            "fps": fps
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="./weights/open_set_reid.onnx")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--iters", type=int, default=150)
    args = parser.parse_args()

    engine = ONNXReIDInferenceEngine(model_path=args.model, device=args.device)
    engine.profile_latency(iterations=args.iters)
