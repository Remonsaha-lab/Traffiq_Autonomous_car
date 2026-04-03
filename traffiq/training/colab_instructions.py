#!/usr/bin/env python3
"""Quick instructions for running training on Google Colab."""


def main() -> None:
    print("Google Colab setup steps:")
    print("1. Upload the `traffiq` folder to Google Drive.")
    print("2. Open a Colab notebook and mount Drive:")
    print("   from google.colab import drive")
    print("   drive.mount('/content/drive')")
    print("3. Change directory to your project:")
    print("   %cd /content/drive/MyDrive/traffiq")
    print("4. Install dependencies:")
    print("   !pip install numpy pandas pillow matplotlib scikit-learn tensorflow")
    print("5. Validate data:")
    print("   !python utils/check_dataset.py --data-dir data/raw")
    print("6. Train model:")
    print("   !python training/train_dave2.py --data-dir data/raw --epochs 12")


if __name__ == "__main__":
    main()
