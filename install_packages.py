import subprocess
import sys

required_packages = [
    "numpy",
    "opencv-python",
    "pillow",
    "torch>=1.7",
    "torchvision",
    "torchsummary",
    "albumentations",
    "matplotlib",
    "seaborn",
    "scikit-image",
    "scikit-learn",
    "pandas",
    "tqdm",
    "xlsxwriter",
    "pyinstaller",
    "PySide6",
]

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

if __name__ == "__main__":
    for package in required_packages:
        try:
            print(f"Installing: {package}")
            install(package)
        except Exception as e:
            print(f"Failed to install {package}: {e}")



