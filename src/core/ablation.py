"""Module/round ablations using the same isolated, scored experiment runner."""
from __future__ import annotations
import asyncio
import json
from pathlib import Path
import uuid
from datetime import datetime


class AblationStudy:
    DEFAULT_MODULE_ABLATIONS = {
        "full": ("完整系统", {}),
        "no_adversarial": ("关闭对抗", {"adversarial": {"enabled": False}}),
        "no_compressor": ("关闭上下文压缩", {"compressor": {"enable_multilevel": False}}),
        "no_memory": ("关闭共享持久记忆", {"memory": {"enabled": False}}),
    }

    @staticmethod
    def override_config(config: dict, overrides: dict) -> dict:
        from evaluation.experiments import merge_config
        return merge_config(config, overrides)

    @classmethod
    def run_module_ablation(cls, config, questions, systems=None):
        from evaluation.experiments import run_experiment
        systems = systems or cls.DEFAULT_MODULE_ABLATIONS
        output = Path("outputs/evaluation") / f"ablation_{uuid.uuid4().hex}"
        result = asyncio.run(run_experiment(config,
            {"name": "ablation", "version": "content-hash", "examples": questions}, output,
            {name: overrides for name, (_, overrides) in systems.items()}))
        rows = result["details"]
        systems_result = [{
            "system_name": name, "description": desc, "num_questions": len(questions),
            "average_composite_score": result["summary"]["variants"][name]["average_composite"],
            "details": [r for r in rows if r["variant"] == name],
        } for name, (desc, _) in systems.items()]
        return {"evaluation_name": "DeepResearch 模块消融", "artifact_dir": str(output),
                "num_questions": len(questions), "systems": systems_result,
                "summary": {r["system_name"]: r["average_composite_score"] for r in systems_result}}

    @classmethod
    def run_rounds_ablation(cls, config, questions, max_rounds=3):
        if max_rounds < 0:
            raise ValueError("max_rounds must be nonnegative")
        systems = {f"adv_{n}": (f"对抗轮数上限 {n}", {"adversarial": {"max_rounds": n, "enabled": n > 0}})
                   for n in range(max_rounds + 1)}
        result = cls.run_module_ablation(config, questions, systems)
        return {"evaluation_name": "DeepResearch 对抗轮数消融", "summary": result["summary"],
                "details": {r["system_name"]: r["details"] for r in result["systems"]},
                "artifact_dir": result["artifact_dir"]}

    @staticmethod
    def save_results(data, output_dir, prefix="ablation"):
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return str(path)
