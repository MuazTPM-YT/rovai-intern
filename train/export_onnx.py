import numpy as np
import onnx
import onnxruntime as ort
import torch

from train_sigmoid_head import DOMAINS, MODELS_DIR, SigmoidDomainHead


def main():
    print("=" * 50)
    print("Exporting to ONNX")
    print("=" * 50)

    model = SigmoidDomainHead()
    model.load_state_dict(torch.load(MODELS_DIR / "sigmoid_domain_head.pt", map_location="cpu", weights_only=True))
    model.eval()

    onnx_path = MODELS_DIR / "sigmoid_domain_head.onnx"
    torch.onnx.export(
        model, torch.randn(1, 384), str(onnx_path),
        input_names=["embedding"], output_names=["logits"],
        dynamic_axes={"embedding": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=18,
    )
    print(f"exported: {onnx_path}")

    onnx.checker.check_model(onnx.load(str(onnx_path)))
    print("onnx validation: ok")

    print("\nparity check...")
    sess = ort.InferenceSession(str(onnx_path))

    val_path = MODELS_DIR / "cache" / "val_embeddings.npy"
    samples = np.load(val_path)[:100] if val_path.exists() else np.random.randn(100, 384).astype(np.float32)

    with torch.no_grad():
        pt_out = model(torch.tensor(samples)).numpy()
    ox_out = sess.run(None, {"embedding": samples})[0]

    maxd = np.max(np.abs(pt_out - ox_out))
    print(f"  max diff: {maxd:.2e} {'PASS' if maxd < 1e-4 else 'WARN'}")
    print(f"  size: {onnx_path.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
