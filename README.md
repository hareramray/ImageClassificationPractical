# Learn image classification by doing

A local, GPU-ready course of **15 Jupyter notebooks**, from your first image array
to transfer learning, transformers, explainability, and a complete project.
Lessons contain explanations, runnable code, plots, practical exercises, and
sample answers. Follow them in order on your first pass.

This course teaches **image classification**: assigning a class to an image.
Object detection and segmentation are different tasks and are outside this course.

## Start on this computer

The project environment and **Image Classification (GPU)** kernel have been
configured for the existing Python 3.13 and CUDA-enabled PyTorch installation.
The NVIDIA GeForce RTX 5050 Laptop GPU (about 8 GB VRAM) was detected and a CUDA
matrix multiplication was verified.

1. Double-click **`start_jupyter.cmd`** in this folder.
2. In JupyterLab, open **`notebooks/00_start_here_gpu_setup.ipynb`**.
3. Select **Image Classification (GPU)** if prompted for a kernel.
4. Run one cell at a time with **Shift+Enter**.

Or launch from PowerShell in this folder:

```powershell
.\.venv\Scripts\python.exe -m jupyterlab
```

Run one training notebook at a time. When finished, use **Kernel → Shut Down
Kernel** to release GPU memory. Stop the Jupyter server with Ctrl+C in its terminal.

## Course roadmap

| # | Notebook | Practical result | Level |
|---|---|---|---|
| 00 | [Start here and GPU setup](notebooks/00_start_here_gpu_setup.ipynb) | Run cells and verify actual GPU use | Beginner |
| 01 | [Image pixels and NumPy](notebooks/01_image_pixels_numpy.ipynb) | Manipulate and visualize images | Beginner |
| 02 | [Classical ML baselines](notebooks/02_classical_ml_baselines.ipynb) | Train a classifier with proper data splits | Beginner |
| 03 | [Tensors and autograd](notebooks/03_pytorch_tensors_autograd.ipynb) | Understand gradients and optimization | Beginner |
| 04 | [First neural classifier](notebooks/04_first_neural_classifier.ipynb) | Write an MLP training loop | Beginner |
| 05 | [Convolutional neural networks](notebooks/05_convolutional_neural_networks.ipynb) | Train a CNN and inspect feature shapes | Intermediate |
| 06 | [Augmentation and regularization](notebooks/06_augmentation_regularization.ipynb) | Compare generalization experiments | Intermediate |
| 07 | [Transfer learning with ResNet](notebooks/07_transfer_learning_resnet.ipynb) | Freeze and fine-tune a pretrained network | Intermediate |
| 08 | [Your own image dataset](notebooks/08_your_own_image_dataset.ipynb) | Train from organized image folders | Intermediate |
| 09 | [Evaluation, imbalance, calibration](notebooks/09_evaluation_imbalance_calibration.ipynb) | Understand errors and probabilities | Advanced |
| 10 | [Experiments and checkpoints](notebooks/10_experiments_checkpoints.ipynb) | Select, save, and resume models | Advanced |
| 11 | [AMP, mixup, distillation](notebooks/11_mixed_precision_mixup_distillation.ipynb) | Apply efficient training techniques | Advanced |
| 12 | [Vision transformer](notebooks/12_vision_transformer.ipynb) | Build a small patch-based transformer | Advanced |
| 13 | [Explainability and inference](notebooks/13_explainability_and_inference.ipynb) | Visualize explanations and export a predictor | Advanced |
| 14 | [Capstone project](notebooks/14_capstone_project.ipynb) | Deliver a model with evaluation evidence | Independent practice |

Allow 45–120 minutes per lesson including experiments, and several sessions for
your own capstone. Advanced techniques are introductions to expert practice;
completing notebooks alone does not establish expertise.

Each notebook runs independently from a fresh kernel. No earlier notebook's
variables or checkpoints are required. Source notebooks start with empty outputs;
executed verification copies are saved under `artifacts/verification/`.

## A useful study routine

Read → predict the output → run → change one setting → record the result → explain it.
Try the exercises before expanding the sample answers. Use **Restart Kernel and
Run All Cells** after edits to detect accidental reliance on old variables.

The first training lessons show the full loop. Later lessons reuse the readable
functions in `course_utils.py` so the new concept stays visible. Open that file
when you want to inspect splitting, batching, training, or evaluation.

Train on the training set, choose settings on validation data, and evaluate the
test set after choices are fixed. The repeated digit split is for learning;
reserve a fresh final test set for an independent project.

## Datasets and files

- **Digits:** bundled with scikit-learn; 1,797 grayscale 8×8 images and 10 classes.
  Most notebooks run without an internet connection after dependencies are installed.
- **CIFAR-10:** notebook 07 downloads about 170 MB compressed on its first normal
  run and uses modest subsets by default. ResNet-18 weights add about 45 MB.
  The expanded data needs additional disk space. Downloads are cached.
- **Your own images:** notebook 08 includes an executable demo and instructions
  for switching to your own `train/class_name`, `val/class_name`, and
  `test/class_name` folders. Demo files stay separate from your own dataset.
- **`data/`:** downloaded/generated datasets. **`artifacts/`:** models, figures,
  experiment tables, model reports, and execution records. Rerunning a lesson
  replaces its named output files; copy experiments you want to preserve.

Dataset references: [scikit-learn digits](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html),
[CIFAR-10 original dataset](https://www.cs.toronto.edu/~kriz/cifar.html).
Lessons link their relevant primary documentation and research papers.

## Recreate the environment

On this computer, the local venv reuses the working system CUDA packages through
`--system-site-packages`. It does not replace your existing PyTorch. It depends
on that base installation remaining available.

```powershell
python -m venv --system-site-packages .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m ipykernel install --prefix .venv --name image-classification --display-name "Image Classification (GPU)"
```

On a different computer, create a normal venv and first install a compatible
`torch` / `torchvision` pair using the command from the
[official PyTorch installation selector](https://pytorch.org/get-started/locally/)
for that operating system and GPU. Then install `requirements.txt` and register
the kernel as above. Torch is deliberately not in that file to avoid replacing
a working GPU build. The tested versions are recorded in `environment-tested.txt`;
they are a record, not a portable CUDA installation command.

## Troubleshooting

| Symptom | Action |
|---|---|
| `ModuleNotFoundError` | Select the project kernel; print `sys.executable` in notebook 00. |
| CUDA is false | Verify the selected kernel, NVIDIA driver, and CUDA-enabled PyTorch installation. |
| CUDA out of memory | Shut down other notebook kernels; restart this kernel and halve batch size. |
| A DataLoader hangs on Windows | Keep `num_workers=0`, as provided in the course. |
| Notebook 07 cannot download | Restore internet access and rerun; other lessons work offline. |
| Variable is undefined | Restart and run cells in order from the top. |
| Low accuracy | Check labels/shapes/splits and learning curves. Smoke runs intentionally train very little. |
| Jupyter kernel is missing in another Jupyter installation | Launch the project `.venv` Jupyter using `start_jupyter.cmd`. |

This installation can print a `triton/windows_utils.py` warning about a missing
CUDA Toolkit. CUDA tensor operations and course GPU training were checked
separately and work. The course uses ordinary PyTorch execution, without
`torch.compile`; check the actual tensor device in notebook 00 before interpreting
that optional compiler warning as a GPU failure.

If the official CIFAR server is slow on a future setup, the optional
`tools/prefetch_transfer_data.py` downloads a byte-identical mirror archive and
requires the original torchvision checksum before using it. The dataset and
pretrained weights are already cached on this computer.

## Verification and standalone prediction

Validate structure and local notebook links:

```powershell
.\.venv\Scripts\python.exe tools\validate_notebooks.py
```

Execute every notebook in a fresh kernel with a tiny GPU smoke run:

```powershell
.\.venv\Scripts\python.exe tools\validate_notebooks.py --execute --smoke --device cuda
```

Omit `--smoke` for normal training, or use `--device cpu` to verify the fallback.
`--pattern "07_*.ipynb"` selects one lesson. The execution report and output
notebooks go into `artifacts/verification/`. Smoke mode verifies code paths,
not model quality; notebook 07 explicitly substitutes random data and weights
only in that mode. Leave `COURSE_SMOKE_TEST` unset for learning runs.

After running notebook 13:

```powershell
.\.venv\Scripts\python.exe predict.py artifacts\13_demo_digit.png
```

This CLI supports the course's digit model only. Its expected input is a centered,
light digit on a dark background. Use `--invert` for a dark digit on a light
background. It is not a general-purpose image classifier.

The authoring scripts in `tools/build_*.py` regenerate source notebooks and
overwrite edits. You do not need to run those scripts to study the course.

## Verified on this computer

On 11 October 2026, all **15 notebooks passed both short GPU smoke checks and
normal execution in fresh Jupyter kernels**. The normal transfer-learning run
used real CIFAR-10 images and verified pretrained ResNet-18 weights. The advanced
training notebook also passed its CPU fallback check. Checkpoint reloads and the
standalone prediction CLI were verified, and representative plots were inspected.

Execution records: [normal runs](artifacts/verification/full_cuda/results.json),
[GPU smoke runs](artifacts/verification/smoke_cuda/results.json),
[CPU fallback](artifacts/verification/smoke_cpu/results.json).
Executed notebooks beside those reports contain example outputs. Normal course
settings use modest datasets and epoch budgets for teaching, not exhaustive tuning.
