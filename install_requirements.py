import torch
import subprocess
import sys


def run(cmd):
    print(f"➡️ Executando: {cmd}")
    subprocess.check_call(cmd, shell=True)


def main():
    print("🔍 Verificando suporte a CUDA...")
    has_cuda = torch.cuda.is_available()
    print(f"CUDA disponível? {has_cuda}")

    if has_cuda:
        print("✅ Instalando pacotes GPU (CUDA 12.1)...")
        run(f"{sys.executable} -m pip install -r requirements-gpu.txt")
    else:
        print("⚠️ CUDA não encontrada. Instalando versão CPU...")
        run(f"{sys.executable} -m pip install -r requirements-cpu.txt")


if __name__ == "__main__":
    main()
