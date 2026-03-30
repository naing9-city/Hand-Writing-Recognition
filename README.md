# Lightweight Offline Handwriting Recognition with Angular Diversity

A high-accuracy, CPU-optimized Offline Handwriting Recognition system designed for Windows. This project introduces **Angular Diversity Regularization**to solve fine-grained character confusion in a resource-constrained environment.

## Key Features

1.  **Angular Diversity Regularization (Novelty):**
    *   Implements a custom Geometry-Aware Loss Function.
    *   Maximizes the angular distance between class centroids in the latent feature space.
    *   Specifically targets and resolves confusion between visually similar classes (e.g., `64` vs `67`, `1` vs `7`).

2.  **CPU-Optimized Architecture:**
    *   Uses a streamlined 4-stage **Standard Conv2D** backbone (32 -> 64 -> 128 -> 128 filters).
    *   Designed for low latency on standard CPUs (no GPU required).
    *   Avoids heavy models like ResNet/EfficientNet in favor of a custom, shallow design.

3.  **Robust Augmentation Pipeline:**
    *   **Post-Segmentation Augmentation:** Geometric transforms (Rotation, Zoom) are applied *after* segmenting characters to preserve structural integrity.
    *   Ensures the model is robust to varying slants, strokes, and writing styles.

4.  **Deployment Ready:**
    *   `train.py`: Full training pipeline with validation and model saving.
    *   `run.py`: One-click inference script for batch processing test images.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/AnseedX/handwriting.git
    cd handwriting
    ```

2.  **Create a virtual environment (Recommended):**
    ```bash
    python -m venv venv
    .\venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install tensorflow opencv-python numpy tqdm
    ```

## Usage

### 1. Training the Model
To train the model from scratch (recommended to enable Angular Diversity):
```bash
python train.py --train_dir train_dir --model_out model.keras --labels_out labels.json --epochs 200
```
*   **--train_dir:** Path to your training dataset images.
*   **--epochs:** Default is 200 (Early stopping is enabled).

### 2. Running Inference (Testing)
To test the model on a folder of unseen images:
```bash
python run.py --test_dir test_dir --model model.keras --labels labels.json --out result.csv
```
*   **--test_dir:** Path to folder containing test images.
*   **Output:** Generates `result.csv` with predictions and accuracy metrics.

## Technical Details

| Component | Specification |
| :--- | :--- |
| **Input Size** | 64x64 (RGB) |
| **Backbone** | Custom 4-Block CNN |
| **Optimizer** | Adam (lr=1e-3) with ReduceLROnPlateau |
| **Loss Function** | Categorical Crossentropy + Angular Diversity (factor=0.01) |
| **Validation** | 10% Random Split |

## Results
*   **Average Accuracy:** **~88.57%**
*   **Inference Speed:** ~15ms per character on CPU.

## Author
**NAING NAING**

