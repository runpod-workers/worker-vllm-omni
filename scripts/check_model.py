"""Submit-time check: can this HF repo deploy on worker-vllm-omni?"""
import json, sys, urllib.request

SUPPORTED = set(json.load(open(__import__("os").path.join(__import__("os").path.dirname(__file__), "omni_supported_archs.json"))))          # from omni's supported-models docs
LORA_FAMILIES = {"QwenImagePipeline", "QwenImageEditPipeline", "QwenImageEditPlusPipeline",
                 "WanPipeline", "WanImageToVideoPipeline", "SenseNovaU1Pipeline", "LTX2Pipeline"}

def check(repo_id: str) -> dict:
    req = urllib.request.Request(f"https://huggingface.co/api/models/{repo_id}")
    with urllib.request.urlopen(req, timeout=15) as r:
        m = json.load(r)

    tags = m.get("tags", [])
    cls = next((t.split(":", 1)[1] for t in tags if t.startswith("diffusers:")), None)

    # Fallback for diffusers repos the API hasn't tagged: read model_index.json directly
    if not cls and m.get("library_name") == "diffusers":
        try:
            with urllib.request.urlopen(
                f"https://huggingface.co/{repo_id}/resolve/main/model_index.json", timeout=15) as r:
                cls = json.load(r).get("_class_name")
        except Exception:
            pass
    # Transformers-format models (e.g. HunyuanImage-3.0): architectures from config.json
    if not cls and m.get("library_name") == "transformers":
        try:
            with urllib.request.urlopen(
                f"https://huggingface.co/{repo_id}/resolve/main/config.json", timeout=15) as r:
                archs = json.load(r).get("architectures") or []
                cls = next((a for a in archs if a in SUPPORTED), archs[0] if archs else None)
        except Exception:
            pass

    gated = m.get("gated")  # False | "auto" | "manual"

    if "lora" in tags:
        base = next((t.split(":", 2)[2] for t in tags if t.startswith("base_model:adapter:")), None)
        verdict = "DEPLOYABLE_AS_LORA (set MODEL_NAME=base + LORA_REPO)" if base else "NOT_DEPLOYABLE (lora, unknown base)"
        if base:
            base_cls = check(base)["pipeline_class"]
            if base_cls not in LORA_FAMILIES:
                verdict = f"BLOCKED (LoRA for {base_cls}: no omni LoRA loader for that family)"
        return {"repo": repo_id, "verdict": verdict, "pipeline_class": cls, "base": base, "gated": gated}

    if "gguf" in tags or m.get("library_name") == "diffusion-single-file":
        return {"repo": repo_id, "verdict": "NOT_DEPLOYABLE (single-file/GGUF packaging)", "pipeline_class": cls, "gated": gated}

    if cls in SUPPORTED:
        v = "DEPLOYABLE" + (" (requires HF_TOKEN + license acceptance)" if gated == "auto"
                            else " (manual gate: user must have access)" if gated == "manual" else "")
        return {"repo": repo_id, "verdict": v, "pipeline_class": cls, "gated": gated}

    return {"repo": repo_id, "verdict": f"NOT_DEPLOYABLE (architecture {cls or 'unknown'} not in omni allowlist)",
            "pipeline_class": cls, "gated": gated}

if __name__ == "__main__":
    for repo in sys.argv[1:]:
        r = check(repo)
        print(f"{r['repo']:50s} -> {r['verdict']}")
