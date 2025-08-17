import os
import argparse
import shutil
from pathlib import Path
import logging
import sys
import cv2
import random
import numpy as np
from ultralytics import YOLO
import albumentations as A
from collections import defaultdict

# ---------------- Logging ----------------
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ---------------- Config padrão ----------------
DEFAULT_CONFIG = {
    "model": "yolov8n.pt",
    "epochs": 50,
    "imgsz": 640,
    "batch": 16,
    "project": "runs/train",
    "name": "detector_sacola",
    "yaml_path": "dataset_sacolas/sacolas.yaml",
    "patience": 15,
    "output_model": "modelos/detector_sacola.pt",
    "aug_factor": 5,  # número padrão de augmentations por imagem
    "balance_classes": True,  # ativa balanceamento de classes
}


# ---------------- Funções utilitárias ----------------
def verificar_estrutura(yaml_path: str):
    base_path = os.path.dirname(yaml_path)
    required_dirs = ["images/train", "labels/train", "images/val", "labels/val"]
    missing, empty = [], []

    for dir_path in required_dirs:
        full_path = os.path.join(base_path, dir_path)
        if not os.path.exists(full_path):
            missing.append(full_path)
        elif not os.listdir(full_path):
            empty.append(full_path)

    if missing or empty:
        msgs = []
        if missing:
            msgs.append(f"Diretórios não encontrados: {', '.join(missing)}")
        if empty:
            msgs.append(f"Diretórios vazios: {', '.join(empty)}")
        raise ValueError("\n".join(msgs))
    logger.info(f"✅ Estrutura de dados válida em: {base_path}")


def configurar_argumentos():
    parser = argparse.ArgumentParser(description="Treinador YOLOv8 para Sacolas")
    for k, v in DEFAULT_CONFIG.items():
        parser.add_argument(f"--{k}", type=type(v), default=v)
    parser.add_argument(
        "--reset", action="store_true", help="Limpar treinamentos anteriores"
    )
    parser.add_argument(
        "--augmentar_dataset", action="store_true", help="Aplicar data augmentation"
    )
    parser.add_argument(
        "--resume", action="store_true", help="Retomar treino de modelo existente"
    )
    return parser.parse_args()


def limpar_diretorios_anteriores(project_dir):
    if os.path.exists(project_dir):
        shutil.rmtree(project_dir, ignore_errors=True)
        logger.info("⚠️ Diretórios de treino anteriores removidos")


def preparar_diretorio_saida(output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    logger.info(f"✅ Diretório de saída preparado: {os.path.dirname(output_path)}")


# ---------------- Augmentation pipeline ----------------
def criar_augmenter():
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.ShiftScaleRotate(
                shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5
            ),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(
                hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.3
            ),
            A.MotionBlur(blur_limit=3, p=0.2),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
        ],
        bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]),
    )


def aplicar_augmentations(im_path, lbl_path, out_img_dir, out_lbl_dir, augmenter, idx):
    try:
        image = cv2.imread(im_path)
        if image is None:
            raise ValueError(f"Não foi possível ler a imagem: {im_path}")
        if not os.path.exists(lbl_path):
            logger.warning(f"Label não encontrado: {lbl_path}")
            return 0

        with open(lbl_path, "r") as f:
            lines = [l.strip().split() for l in f if l.strip()]
            classes = [int(l[0]) for l in lines]
            bboxes = [list(map(float, l[1:])) for l in lines]

        transformed = augmenter(image=image, bboxes=bboxes, class_labels=classes)

        out_img = os.path.join(out_img_dir, f"aug_{idx}_{os.path.basename(im_path)}")
        out_lbl = os.path.join(out_lbl_dir, f"aug_{idx}_{os.path.basename(lbl_path)}")
        cv2.imwrite(out_img, transformed["image"])
        with open(out_lbl, "w") as f:
            for cls, bbox in zip(transformed["class_labels"], transformed["bboxes"]):
                f.write(f"{cls} {' '.join(f'{x:.6f}' for x in bbox)}\n")
        return 1
    except Exception as e:
        logger.error(f"Erro augment {im_path}: {e}")
        return 0


# ---------------- Dataset balanceado ----------------


def coletar_imagens_por_classe(img_dir, lbl_dir):
    """Retorna um dicionário {classe: [(imagem_path, label_path), ...]}"""
    class_imgs = defaultdict(list)
    for f in os.listdir(img_dir):
        if not f.lower().endswith((".jpg", ".png")):
            continue
        lbl_file = os.path.join(lbl_dir, f.rsplit(".", 1)[0] + ".txt")
        if not os.path.exists(lbl_file):
            continue
        with open(lbl_file, "r") as lf:
            classes = [int(l.strip().split()[0]) for l in lf if l.strip()]
        for cls in classes:
            class_imgs[cls].append((os.path.join(img_dir, f), lbl_file))
    return class_imgs


def expandir_dataset_balanceado(yaml_path, aug_factor=5):
    """Expande o dataset balanceando classes com menos imagens"""
    base = os.path.dirname(yaml_path)
    img_dir = os.path.join(base, "images/train")
    lbl_dir = os.path.join(base, "labels/train")
    augmenter = criar_augmenter()
    total_aug = 0

    class_imgs = coletar_imagens_por_classe(img_dir, lbl_dir)
    if not class_imgs:
        raise ValueError("Nenhuma imagem com label encontrada para augmentation")

    max_count = max(len(v) for v in class_imgs.values())  # classe com mais imagens

    for cls, imgs in class_imgs.items():
        needed = max_count - len(imgs)
        imgs_to_aug = imgs * ((needed // len(imgs)) + 1)
        for i, (img_path, lbl_path) in enumerate(imgs_to_aug[:needed]):
            for n in range(aug_factor):
                total_aug += aplicar_augmentations(
                    img_path, lbl_path, img_dir, lbl_dir, augmenter, f"{cls}_{i}_{n}"
                )

    logger.info(f"✅ Dataset expandido balanceado com {total_aug} novas imagens")


def expandir_dataset_automatico(yaml_path, aug_factor=5):
    base = os.path.dirname(yaml_path)
    img_dir = os.path.join(base, "images/train")
    lbl_dir = os.path.join(base, "labels/train")
    augmenter = criar_augmenter()
    total_aug = 0

    imagens = [f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".png"))]
    if not imagens:
        raise ValueError("Nenhuma imagem encontrada para augmentation")

    for idx, img_file in enumerate(imagens):
        img_path = os.path.join(img_dir, img_file)
        lbl_path = os.path.join(lbl_dir, img_file.rsplit(".", 1)[0] + ".txt")
        if not os.path.exists(lbl_path):
            logger.warning(f"Label não encontrado: {lbl_path}")
            continue
        for n in range(aug_factor):
            total_aug += aplicar_augmentations(
                img_path, lbl_path, img_dir, lbl_dir, augmenter, f"{idx}_{n}"
            )

    logger.info(f"✅ Dataset expandido com {total_aug} novas imagens")


# ---------------- Treinamento ----------------
def treinar_modelo(cfg):
    try:
        if cfg.reset:
            limpar_diretorios_anteriores(cfg.project)
        verificar_estrutura(cfg.yaml_path)
        preparar_diretorio_saida(cfg.output_model)

        if cfg.augmentar_dataset:
            if cfg.balance_classes:
                expandir_dataset_balanceado(cfg.yaml_path, cfg.aug_factor)
            else:
                expandir_dataset_automatico(cfg.yaml_path, cfg.aug_factor)

        modelo = YOLO(cfg.model)
        overrides = {
            "data": cfg.yaml_path,
            "epochs": cfg.epochs,
            "imgsz": cfg.imgsz,
            "batch": cfg.batch,
            "patience": cfg.patience,
            "project": cfg.project,
            "name": cfg.name,
            "optimizer": "AdamW",
            "lr0": 0.001,
            "lrf": 0.01,
            "momentum": 0.937,
            "weight_decay": 0.0005,
            "fliplr": 0.5,
            "hsv_h": 0.01,
            "hsv_s": 0.5,
            "hsv_v": 0.3,
            "augment": True,
            "resume": cfg.resume,
        }

        resultados = modelo.train(**overrides)

        caminho_modelo = Path(cfg.project) / cfg.name / "weights" / "best.pt"
        shutil.copy(caminho_modelo, cfg.output_model)
        logger.info(f"✅ Modelo salvo: {cfg.output_model}")

        modelo.export(format="onnx", imgsz=cfg.imgsz)
        logger.info(
            f"✅ Modelo exportado ONNX: {cfg.output_model.replace('.pt','.onnx')}"
        )

        return resultados

    except Exception as e:
        logger.error(f"Erro treinamento: {e}")
        raise


# ---------------- Main ----------------
if __name__ == "__main__":
    try:
        import torch, albumentations as A, cv2
        from ultralytics import YOLO

        cfg = configurar_argumentos()
        resultados = treinar_modelo(cfg)
        logger.info("✅ Treinamento finalizado com sucesso!")

    except ImportError as e:
        logger.error(f"Dependências não encontradas: {e}")
        logger.info(
            "Instale: pip install torch albumentations opencv-python ultralytics"
        )
        sys.exit(1)

    except Exception as e:
        logger.error(f"Falha treinamento: {e}")
        sys.exit(1)
