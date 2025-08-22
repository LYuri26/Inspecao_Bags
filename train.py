from __future__ import annotations

import argparse
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
import time
import json
import random

import numpy as np

# Evita warning de OpenMP em algumas distros
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("ULTRALYTICS_AGGREGATE", "1")

try:
    from ultralytics import YOLO
except Exception as e:
    print(
        "❌ Falha ao importar ultralytics. Instale com: pip install ultralytics",
        file=sys.stderr,
    )
    raise


# ---------------------------- Config --------------------------------- #
@dataclass
class TrainCLI:
    data: Path = Path("dataset_sacolas/data.yaml")
    project: Path = Path("runs/train")
    name: str = "detector_sacola"
    model: str = "yolov8s.pt"  # backbone desejado
    device: str = "cpu"  # força CPU
    epochs: int = 320  # longo para CPU
    batch: int = 4  # 4 cabe bem em 32GB e acelera
    imgsz: int = 736  # múltiplo de 32 (evita resize implícito)
    workers: int = 8  # threads de dataloader
    patience: int = 30  # não parar cedo
    seed: int = 42
    output_model: Path = Path("runs/train/detector_sacola/weights/best.pt")
    resume: bool = False  # retomar de ultimo treino na pasta
    save_period: int = 25  # checkpoint a cada N épocas


# ---------------------------- Utils ---------------------------------- #
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_paths(cfg: TrainCLI) -> None:
    cfg.project.mkdir(parents=True, exist_ok=True)
    out_dir = cfg.project / cfg.name
    out_dir.mkdir(parents=True, exist_ok=True)
    # garante que o caminho de saída existe
    dest = Path(cfg.output_model).parent
    dest.mkdir(parents=True, exist_ok=True)


def copy_best(src_dir: Path, dst_path: Path) -> None:
    best_src = src_dir / "weights" / "best.pt"
    last_src = src_dir / "weights" / "last.pt"
    src = best_src if best_src.exists() else last_src
    if src.exists():
        shutil.copy2(src, dst_path)
        print(f"📦 Copiado modelo final para: {dst_path}")
    else:
        print("⚠️ Não encontrei weights/best.pt nem weights/last.pt para copiar.")


# ---------------------------- Treino --------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="Treino YOLOv8s otimizado para CPU")
    parser.add_argument("--data", type=str, default=str(TrainCLI.data))
    parser.add_argument("--project", type=str, default=str(TrainCLI.project))
    parser.add_argument("--name", type=str, default=TrainCLI.name)
    parser.add_argument("--model", type=str, default=TrainCLI.model)
    parser.add_argument("--device", type=str, default=TrainCLI.device)
    parser.add_argument("--epochs", type=int, default=TrainCLI.epochs)
    parser.add_argument("--batch", type=int, default=TrainCLI.batch)
    parser.add_argument("--imgsz", type=int, default=TrainCLI.imgsz)
    parser.add_argument("--workers", type=int, default=TrainCLI.workers)
    parser.add_argument("--patience", type=int, default=TrainCLI.patience)
    parser.add_argument("--seed", type=int, default=TrainCLI.seed)
    parser.add_argument("--resume", action="store_true", default=TrainCLI.resume)
    parser.add_argument("--save-period", type=int, default=TrainCLI.save_period)
    parser.add_argument("--output-model", type=str, default=str(TrainCLI.output_model))
    args = parser.parse_args()

    cfg = TrainCLI(
        data=Path(args.data),
        project=Path(args.project),
        name=args.name,
        model=args.model,
        device=args.device,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        workers=args.workers,
        patience=args.patience,
        seed=args.seed,
        output_model=Path(args.output_model),
        resume=args.resume,
        save_period=args.save_period,
    )

    set_seed(cfg.seed)
    ensure_paths(cfg)

    # Sanidade básica
    if not cfg.data.exists():
        print(f"❌ data.yaml não encontrado: {cfg.data}")
        sys.exit(1)

    print("\n================= CONFIG =================")
    print(
        json.dumps(
            {
                "model": cfg.model,
                "data": str(cfg.data),
                "device": cfg.device,
                "epochs": cfg.epochs,
                "batch": cfg.batch,
                "imgsz": cfg.imgsz,
                "workers": cfg.workers,
                "patience": cfg.patience,
                "project/name": f"{cfg.project}/{cfg.name}",
                "resume": cfg.resume,
            },
            indent=2,
        )
    )
    print("==========================================\n")

    # Carrega modelo base
    model = YOLO(cfg.model)

    # Hiperparâmetros e overrides pensados para CPU
    overrides = {
        # básico
        "data": str(cfg.data),
        "epochs": cfg.epochs,
        "batch": cfg.batch,
        "imgsz": cfg.imgsz,
        "device": cfg.device,
        "project": str(cfg.project),
        "name": cfg.name,
        "workers": cfg.workers,
        "verbose": True,
        "save": True,
        "save_period": cfg.save_period,
        "exist_ok": True,
        "seed": cfg.seed,
        "patience": cfg.patience,  # early stopping mais paciente
        "resume": cfg.resume,
        # desempenho CPU
        "cache": "ram",  # cache em RAM (mais rápido na CPU)
        "pin_memory": True,
        "persist": True,
        "pretrained": True,  # mantém head randômica mas backbone com pesos base
        # otimizador + scheduler estáveis em treinos longos
        "optimizer": "AdamW",
        "cos_lr": True,
        "lr0": 0.001,  # lr inicial menor (estabilidade)
        "lrf": 0.01,  # lr final (cosine)
        "momentum": 0.9,  # usado por algumas políticas internas
        "weight_decay": 0.0005,
        # augmentações fortes para reduzir FNs
        "degrees": 5.0,
        "translate": 0.10,
        "scale": 0.75,  # permite downscale/upscale
        "shear": 2.0,
        "fliplr": 0.5,
        "flipud": 0.1,
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        # Técnicas de mistura – ajudam em datasets pequenos
        "mosaic": 1.0,  # habilitado (probabilidade)
        "mixup": 0.15,
        "copy_paste": 0.1,
        "close_mosaic": 10,  # desliga nas últimas N épocas para refinar
        # validação e artefatos
        "val": True,
        "plots": True,
        "save_json": False,  # COCO json (habilite se precisar)
    }

    t0 = time.time()
    print("🚀 Iniciando treino...")
    results = model.train(**overrides)
    dur = time.time() - t0
    print(f"✅ Treino finalizado em {dur/3600:.2f} h")

    # Avaliação final (val) — garante métricas atualizadas no final
    print("📊 Rodando avaliação final...")
    model.val(
        data=str(cfg.data),
        imgsz=cfg.imgsz,
        batch=cfg.batch,
        device=cfg.device,
        plots=True,
    )

    # Copia best.pt para destino pedido
    exp_dir = (
        Path(results.save_dir)
        if hasattr(results, "save_dir")
        else (cfg.project / cfg.name)
    )
    copy_best(exp_dir, cfg.output_model)

    print("\n🎉 Pronto! Modelos e artefatos em:", exp_dir)
    print("   Melhor peso em:", cfg.output_model)


if __name__ == "__main__":
    main()
