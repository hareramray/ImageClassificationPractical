from notebook_builder import md, code, bootstrap, write_notebook


def build_start():
    write_notebook("00_start_here_gpu_setup.ipynb", [
        md("""
        # 00 · Start here: your practical image classification course

        **Level:** absolute beginner · **Study time:** 30–45 minutes

        Your goal is to turn an image into a class label, such as `cat` or `dog`.
        Classification assigns a label to an entire image. Detection also locates
        objects with boxes; segmentation labels individual pixels. This course
        teaches classification, even though the project folder has “ObjectDetection” in its name.

        You will run real experiments, inspect mistakes, and eventually build
        your own classifier. Advanced notebooks introduce expert techniques;
        expertise comes from applying them to new data and explaining the results.

        **You will learn:** how notebooks work, how to confirm GPU use, and how
        to carry out a small experiment. No prior deep learning knowledge is needed.
        """),
        md("""
        ## 1. Run your first cell

        Click the cell below, then press **Shift+Enter**. Markdown cells explain;
        code cells execute Python. A number on the left means a code cell has run.
        `[*]` means it is busy. Variables remain in the kernel's memory.

        Use **Kernel → Restart Kernel and Run All Cells** to confirm your work
        runs in order. Save a copy before attempting exercises.
        """),
        code('message = "I am learning image classification by doing!"\nprint(message)\nprint("Two plus three is", 2 + 3)'),
        md("""
        ## 2. Confirm the Python environment

        Select **Image Classification (GPU)** as the notebook kernel. The path
        printed below should end in `.venv\\Scripts\\python.exe` for this project.
        The environment reuses your existing CUDA-enabled PyTorch installation.
        Notebook 01 explains the imports and Python basics in detail.
        """),
        code('import sys\nimport platform\nprint("Python:", sys.version)\nprint("Executable:", sys.executable)\nprint("Operating system:", platform.platform())'),
        bootstrap(),
        code("""
        import torch
        import torchvision
        import numpy as np
        import matplotlib.pyplot as plt
        import sklearn
        print("PyTorch:", torch.__version__)
        print("Torchvision:", torchvision.__version__)
        print("Scikit-learn:", sklearn.__version__)
        print("CUDA runtime bundled with PyTorch:", torch.version.cuda)
        print("CUDA available:", torch.cuda.is_available())
        if torch.cuda.is_available():
            properties = torch.cuda.get_device_properties(0)
            print("GPU:", properties.name)
            print("GPU memory (GiB):", round(properties.total_memory / 1024**3, 1))
        """),
        md("""
        ## 3. Actually use the GPU

        A visible GPU does not automatically move calculations to it. Both the
        model and its input tensors must be on the same device. Here we create
        a matrix on the selected device and multiply it. On your configured
        computer, expect `cuda:0` and an NVIDIA GeForce RTX 5050 Laptop GPU.
        A CPU fallback is supported on other machines.

        On this installation an optional `triton/windows_utils.py` warning may
        mention a missing CUDA Toolkit. The course's ordinary PyTorch GPU
        execution has been checked separately; it does not use `torch.compile`.
        Verify the actual tensor device below rather than inferring device use
        from that compiler warning.
        """),
        code("""
        matrix = torch.randn(256, 256, device=device)
        product = matrix @ matrix.T
        print("Result shape:", tuple(product.shape))
        print("Actual tensor device:", product.device)
        assert torch.isfinite(product).all()
        """),
        md("""
        ## 4. See the first dataset

        Most lessons use scikit-learn's bundled handwritten digits: 1,797 small
        images, 10 labels (0–9), and 8×8 grayscale pixels. There is no dataset
        download. Small data makes the entire pipeline easy to inspect.
        Notebook 07 moves to color photographs with CIFAR-10 and pretrained weights.
        """),
        code("""
        from sklearn.datasets import load_digits
        digits = load_digits()
        fig, axes = plt.subplots(1, 6, figsize=(10, 2))
        for index, ax in enumerate(axes):
            ax.imshow(digits.images[index], cmap="gray", vmin=0, vmax=16)
            ax.set_title(f"Label: {digits.target[index]}")
            ax.axis("off")
        plt.tight_layout()
        plt.show()
        print("Image count:", len(digits.images))
        """),
        md("""
        ## 5. Follow the learning path

        | Notebook | What you will build |
        |---|---|
        | [01 · Pixels and Python](01_image_pixels_numpy.ipynb) | Image manipulation with NumPy and Pillow |
        | [02 · Classical ML](02_classical_ml_baselines.ipynb) | A baseline classifier with honest splits |
        | [03 · PyTorch fundamentals](03_pytorch_tensors_autograd.ipynb) | Tensors, gradients, and optimization |
        | [04 · First neural classifier](04_first_neural_classifier.ipynb) | An MLP with an explicit training loop |
        | [05 · CNNs](05_convolutional_neural_networks.ipynb) | A network that learns spatial features |
        | [06 · Generalization](06_augmentation_regularization.ipynb) | Augmentation and regularization experiments |
        | [07 · Transfer learning](07_transfer_learning_resnet.ipynb) | Frozen and fine-tuned ResNet-18 |
        | [08 · Your own data](08_your_own_image_dataset.ipynb) | A classifier trained from image folders |
        | [09 · Evaluation](09_evaluation_imbalance_calibration.ipynb) | Imbalance metrics and calibrated probabilities |
        | [10 · Experiments](10_experiments_checkpoints.ipynb) | Reproducible model selection and checkpoints |
        | [11 · Advanced training](11_mixed_precision_mixup_distillation.ipynb) | Mixed precision, mixup, and distillation |
        | [12 · Vision transformers](12_vision_transformer.ipynb) | A small transformer built from image patches |
        | [13 · Explain and export](13_explainability_and_inference.ipynb) | Saliency, Grad-CAM, and saved-model inference |
        | [14 · Capstone](14_capstone_project.ipynb) | A complete project with a final model report |

        Study in order the first time. Every notebook loads its own data and
        creates its own models; you do not need another notebook's kernel state.
        Reserve 45–120 minutes per lesson for exercises. Runtime is usually
        much shorter than study time; dataset downloads depend on your connection.
        """),
        md("""
        ## 6. Use a repeatable learning routine

        1. Read the explanation and predict the output before running a cell.
        2. Run the unchanged notebook once.
        3. Change one setting and record what happened on the validation set.
        4. Explain a mistake using an actual image.
        5. Restart the kernel and run all cells again.

        **Training set:** fit parameters. **Validation set:** choose settings.
        **Test set:** final measurement after decisions are fixed. Repeatedly
        choosing settings from test scores makes that score optimistic. The digit
        split is reused for teaching; for a final project, use fresh held-out data.

        This course uses `num_workers=0` for reliable Windows notebook loading.
        If GPU memory runs out, restart the kernel and halve the batch size.
        Run one training notebook at a time to avoid competing for GPU memory.
        """),
        code("""
        experiment_notes = {
            "question": "What will happen if I double the batch size?",
            "prediction": "Write my prediction before trying it.",
            "changed_setting": "batch_size",
            "validation_result": None,
            "what_i_learned": "Fill this after the experiment.",
        }
        experiment_notes
        """),
        md("""
        ## 7. Files, downloads, and short verification runs

        `data/` stores datasets and demo images. `artifacts/` stores experiment
        tables, plots, checkpoints, and executed verification copies. Source
        notebooks remain clear of precomputed output so you can run them yourself.
        Most notebooks need no internet. Notebook 07 downloads CIFAR-10
        (about 170 MB compressed) and ResNet-18 weights (about 45 MB) once.

        `COURSE_SMOKE_TEST=1` is an automated wiring check with tiny datasets and
        short training. It is **not** a meaningful accuracy benchmark. Normal
        learning runs leave it unset. In notebook 07 only, smoke mode uses fake
        images and random model weights to avoid downloads during that check.
        """),
        md("""
        ## Practice

        **A.** Change the greeting, run it, and save the notebook.
        **B.** Display digit image number 25. Predict its shape before printing it.
        **C.** Create a tensor on `device`; verify its `.device` and `.shape`.
        """),
        code("""
        # Exercise B: change this index and rerun.
        image_index = 25
        plt.imshow(digits.images[image_index], cmap="gray")
        plt.title(f"True label: {digits.target[image_index]}")
        plt.show()
        print(digits.images[image_index].shape)
        """),
        code("""
        # Exercise C: try a different shape.
        practice_tensor = torch.zeros(2, 3, device=device)
        print(practice_tensor.device, practice_tensor.shape)
        """),
        md("""
        <details><summary>Sample answers and self-check</summary>

        Image 25 has shape `(8, 8)`. A tensor created using `device=device`
        prints `cuda:0` on the configured GPU system. The first dimension of
        `torch.zeros(2, 3)` has size 2; the second has size 3.
        If changing code appears to do nothing, rerun that cell.
        </details>

        **Ready to continue when:** you can run a cell, restart the kernel, name
        your active Python executable, and verify the tensor's actual device.

        **References:** [JupyterLab notebooks](https://jupyterlab.readthedocs.io/en/stable/user/notebook.html),
        [PyTorch installation](https://pytorch.org/get-started/locally/),
        [scikit-learn digits](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html).

        **Next:** [01 · Pixels and Python](01_image_pixels_numpy.ipynb).
        """),
    ])


def build_explain():
    write_notebook("13_explainability_and_inference.ipynb", [
        md("""
        # 13 · Explain predictions and run a saved model

        **Level:** advanced · **Prerequisites:** 04, 05, 09, 10 · **Study time:** 90 minutes

        You will train a small CNN, visualize input gradients and Grad-CAM,
        export weights with their preprocessing contract, and make predictions
        from a separate Python script. Everything runs independently of earlier notebooks.

        A heatmap indicates a model sensitivity or a spatial association. It does
        not prove causal reasoning, fairness, or correctness. Our 8×8 images and
        small feature maps are especially coarse; treat the visualizations as
        debugging tools and compare them with actual errors.
        """),
        bootstrap(),
        code("""
        import copy
        import json
        import time
        import numpy as np
        import torch
        import matplotlib.pyplot as plt
        from PIL import Image
        from course_utils import TinyCNN, get_digits_loaders, fit, evaluate
        train_loader, val_loader, test_loader = get_digits_loaders()
        model = TinyCNN().to(device)
        history = fit(model, train_loader, val_loader,
                      epochs=1 if SMOKE_TEST else 12, device=device)
        """),
        md("""
        ## 1. Pick a validation image and explain its predicted class

        Use validation examples while developing explanations. The test set
        stays untouched here. The model returns **logits**. We apply softmax
        only for displayed probabilities, not before cross-entropy training.
        """),
        code("""
        val_images, val_labels = next(iter(val_loader))
        example_index = 0
        x = val_images[example_index:example_index + 1].to(device)
        model.eval()
        with torch.no_grad():
            probabilities = model(x).softmax(1)
        predicted_class = int(probabilities.argmax(1).item())
        print("True:", int(val_labels[example_index]), "Predicted:", predicted_class)
        print("Probability of predicted class:", float(probabilities[0, predicted_class]))
        plt.imshow(x[0, 0].cpu(), cmap="gray")
        plt.title("Validation example")
        plt.show()
        """),
        md("""
        ## 2. Input-gradient saliency

        Ask: if a pixel changed slightly, how much would the chosen class score
        change? Enable gradients on the input, backpropagate one **logit**, and
        inspect the magnitude of the input gradient. Do not put this inside
        `no_grad()` or `inference_mode()`; those intentionally disable gradients.
        """),
        code("""
        saliency_input = x.detach().clone().requires_grad_(True)
        model.zero_grad(set_to_none=True)
        class_score = model(saliency_input)[0, predicted_class]
        class_score.backward()
        saliency = saliency_input.grad.detach().abs()[0, 0].cpu()
        saliency = saliency / saliency.max().clamp_min(1e-8)
        fig, axes = plt.subplots(1, 2, figsize=(7, 3))
        axes[0].imshow(x[0, 0].cpu(), cmap="gray")
        axes[0].set_title("Input")
        axes[1].imshow(saliency, cmap="magma", vmin=0, vmax=1)
        axes[1].set_title("Absolute input gradient")
        plt.tight_layout()
        plt.show()
        """),
        md("""
        ## 3. Grad-CAM from the final convolution

        A forward hook captures feature maps `A[k]`. Gradients of a selected
        logit with respect to each map give channel importance weights:
        `alpha[k] = mean(d score / d A[k])` over spatial positions.
        Compute `ReLU(sum(alpha[k] * A[k]))`, then resize it for display.
        We remove the hook in a `finally` block so rerunning the cell cannot stack hooks.
        """),
        code("""
        def grad_cam(model, image, target_class):
            captured = {}
            def capture(module, inputs, output):
                captured["features"] = output
                output.retain_grad()
            handle = model.features[3].register_forward_hook(capture)
            try:
                model.zero_grad(set_to_none=True)
                score = model(image)[0, target_class]
                score.backward()
                features = captured["features"]
                weights = features.grad.mean(dim=(2, 3), keepdim=True)
                heatmap = (weights * features).sum(dim=1, keepdim=True).relu()
                heatmap = torch.nn.functional.interpolate(
                    heatmap, size=image.shape[-2:], mode="bilinear", align_corners=False)
                heatmap = heatmap[0, 0].detach().cpu()
                return heatmap / heatmap.max().clamp_min(1e-8)
            finally:
                handle.remove()

        cam = grad_cam(model, x, predicted_class)
        plt.imshow(x[0, 0].cpu(), cmap="gray")
        plt.imshow(cam, cmap="jet", alpha=0.45, vmin=0, vmax=1)
        plt.title(f"Grad-CAM for class {predicted_class}")
        plt.colorbar(label="Relative intensity")
        plt.show()
        """),
        md("""
        A blank map can occur when the weighted evidence is nonpositive before
        ReLU. It is not an error to hide by inventing a heatmap. Try another
        image or class and report what actually happens. Colormaps are normalized
        per image here, so their brightness cannot be compared as absolute confidence.

        ## 4. Probe a small distribution shift

        Add noise to the same validation batch and compare accuracy. This is one
        narrow corruption test, not a robustness certificate. We clamp to the
        legal [0,1] input range and keep labels unchanged.
        """),
        code("""
        model.eval()
        with torch.no_grad():
            batch = val_images.to(device)
            labels = val_labels.to(device)
            noisy = (batch + 0.20 * torch.randn_like(batch)).clamp(0, 1)
            clean_accuracy = (model(batch).argmax(1) == labels).float().mean().item()
            noisy_accuracy = (model(noisy).argmax(1) == labels).float().mean().item()
        print(f"One validation batch: clean={clean_accuracy:.1%}, noisy={noisy_accuracy:.1%}")
        """),
        md("""
        ## 5. Save a complete inference contract

        Weights alone do not specify class order, color conversion, size, or
        normalization. Save those details together. This bundle uses only tensors
        and simple Python values and loads with `weights_only=True`. Load files
        you created or trust. Keep the `TinyCNN` architecture code with the bundle.
        """),
        code("""
        checkpoint_path = ARTIFACTS / "13_digit_classifier.pt"
        bundle = {
            "architecture": "TinyCNN-v1",
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "classes": [str(i) for i in range(10)],
            "image_size": 8,
            "channels": 1,
            "preprocessing": "grayscale, bilinear resize to 8x8, divide uint8 pixels by 255",
            "expected_input": "centered light digit on dark background",
            "training_data": "sklearn digits; original pixels divided by 16",
            "seed": 42,
            "smoke_test": SMOKE_TEST,
        }
        torch.save(bundle, checkpoint_path)
        print("Saved:", checkpoint_path)
        """),
        md("""
        ## 6. Verify the save/load round trip

        Compare identical inputs on the same device before and after loading.
        Saving should preserve logits. This check catches architecture or weight
        mistakes; it does not measure classification quality.
        """),
        code("""
        from predict import load_bundle, predict_image, image_to_tensor
        restored, restored_bundle = load_bundle(checkpoint_path, device="cpu")
        original_cpu = copy.deepcopy(model).cpu().eval()
        probe = val_images[:4].cpu()
        with torch.inference_mode():
            before = original_cpu(probe)
            after = restored(probe)
        torch.testing.assert_close(before, after)
        print("Round-trip logits match.")
        """),
        md("""
        ## 7. Classify an actual PNG through the inference function

        Export one validation digit as an image, then load it using Pillow. The
        conversion to uint8 introduces a little quantization, so exact logits may
        differ from the original floating point sample. Phone photos are a much
        larger domain shift: this tiny digit model is not a general photo classifier.
        """),
        code("""
        demo_path = ARTIFACTS / "13_demo_digit.png"
        png_pixels = (val_images[example_index, 0].numpy() * 255).round().astype(np.uint8)
        Image.fromarray(png_pixels).save(demo_path)
        top_three = predict_image(restored, restored_bundle, demo_path, top_k=3)
        print(json.dumps(top_three, indent=2))
        print("Image:", demo_path)
        """),
        md("""
        From PowerShell in the project folder, the same prediction works without
        Jupyter:

        ```powershell
        .\\.venv\\Scripts\\python.exe predict.py artifacts\\13_demo_digit.png
        ```

        For your own **dark digit on a light background**, add `--invert`.
        Crop and center the digit first. A high softmax value is not a guarantee
        of correctness and does not detect unrelated images automatically.
        """),
        md("""
        ## 8. Measure inference latency honestly

        Warm up the model, time repeated forward passes, and state what the timer
        excludes. This CPU measurement excludes image decoding, disk I/O, and
        network serving. For GPU timing, synchronize CUDA before and after timing;
        GPU operations are asynchronous.
        """),
        code("""
        input_tensor = image_to_tensor(demo_path)
        repeats = 10 if SMOKE_TEST else 100
        with torch.inference_mode():
            for _ in range(5):
                restored(input_tensor)
            started = time.perf_counter()
            for _ in range(repeats):
                restored(input_tensor)
        milliseconds = 1000 * (time.perf_counter() - started) / repeats
        print(f"CPU forward-only mean latency: {milliseconds:.3f} ms/image")
        """),
        md("""
        ## Practice: turn explanations into experiments

        **A.** Explain the true class and predicted class for a misclassified
        validation image. Does the map change? Record the class used for each map.

        **B.** Compare noise strengths 0.0, 0.1, and 0.3 on validation data. Keep
        a fixed noise realization while changing strength for a fair comparison.

        **C.** Export a second image and run `predict.py`. Try an upside-down
        version and explain why the confidence may still be high.
        """),
        code("""
        # Exercise A starter: locate mistakes without touching test data.
        validation = evaluate(model, val_loader, device)
        mistake_indices = (validation["logits"].argmax(1) != validation["targets"]).nonzero().flatten()
        print("First validation mistake indices:", mistake_indices[:10].tolist())
        """),
        code("""
        # Exercise B starter: extend the strength list, then compare accuracies.
        fixed_noise = torch.randn_like(val_images)
        for strength in [0.0, 0.1, 0.3]:
            corrupted = (val_images + strength * fixed_noise).clamp(0, 1).to(device)
            with torch.no_grad():
                accuracy = (model(corrupted).argmax(1).cpu() == val_labels).float().mean().item()
            print(strength, round(accuracy, 3))
        """),
        md("""
        <details><summary>Sample solution guidance</summary>

        Use `val_loader.dataset[index]` for a mistake index, because the index can
        extend beyond the first batch. Pass `image.unsqueeze(0).to(device)` to
        `grad_cam` so the batch dimension and device both match the model.
        Pass the actual label as `target_class` to explain the true class instead.
        Larger noise generally hurts performance, but a small batch need not be
        monotonic. Rotating an image changes the distribution; softmax must still
        allocate its probability mass among the ten known digits.
        </details>

        **You can now:** debug a prediction, describe the limits of an explanation,
        export a reproducible inference contract, and predict from a separate script.

        **References:** [Grad-CAM paper](https://arxiv.org/abs/1610.02391),
        [PyTorch saving/loading](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html).

        **Next:** [14 · Your capstone project](14_capstone_project.ipynb).
        """),
    ])


def build_capstone():
    write_notebook("14_capstone_project.ipynb", [
        md("""
        # 14 · Capstone: an image classifier you can defend

        **Level:** advanced practice · **Prerequisites:** 01–13 · **Study time:** 2–4 hours,
        then several sessions with your own data.

        The completed reference project below selects a CNN using validation
        results, performs one final test evaluation, and exports a checkpoint,
        a confusion matrix, an experiment table, and a model report. Run it once
        before replacing the dataset using the ImageFolder pattern from notebook 08.

        Your final challenge is to explain why a model is appropriate for its
        intended use, where it fails, and what evidence supports your decisions.
        A complicated model is not automatically a better model.
        """),
        bootstrap(),
        md("""
        ## 1. Write the task before training

        Specify the classes, image source, intended use, and a measurable success
        criterion. Decide your selection metric now. This reference project uses
        validation macro-F1 to give every digit equal importance.
        """),
        code("""
        PROJECT = {
            "title": "Handwritten digit classification",
            "intended_use": "Educational classification of centered 8x8 digit images",
            "out_of_scope": "Phone photos, handwriting recognition in documents, critical decisions",
            "classes": [str(i) for i in range(10)],
            "selection_metric": "validation macro-F1",
            "target_macro_f1": 0.90,
            "seed": 42,
        }
        PROJECT
        """),
        md("""
        ## 2. Audit the data and split before experiments

        For photographs, keep all images of the same person, product, scene, or
        capture session in one split. Near-duplicates can leak information even
        when filenames differ. This teaching dataset uses a stratified random
        70/15/15 split; it is not a writer-independent digit benchmark.
        The split was seen in earlier lessons, so a new project needs fresh test data.
        """),
        code("""
        import json
        import numpy as np
        import pandas as pd
        import torch
        import matplotlib.pyplot as plt
        from sklearn.metrics import f1_score, classification_report, ConfusionMatrixDisplay
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        from course_utils import digit_splits, get_digits_loaders, TinyCNN, fit, evaluate
        splits = digit_splits(PROJECT["seed"])
        audit_rows = []
        for name, (images, labels) in splits.items():
            assert images.ndim == 3 and images.shape[1:] == (8, 8)
            assert np.isfinite(images).all() and images.min() >= 0 and images.max() <= 1
            assert set(np.unique(labels)) == set(range(10))
            audit_rows.append({"split": name, "images": len(images),
                               "class_counts": np.bincount(labels, minlength=10).tolist()})
        pd.DataFrame(audit_rows)
        """),
        md("""
        ## 3. Establish a simple reference

        A linear classifier can work very well on this small dataset. Fit scaling
        only on training data through a pipeline. We record its validation result
        as a reference; the CNN search below selects between two fixed CNN
        configurations. This is a neural-network learning project, so we report
        openly if the linear reference is stronger.
        """),
        code("""
        train_x, train_y = splits["train"]
        val_x, val_y = splits["val"]
        baseline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        baseline.fit(train_x.reshape(len(train_x), -1), train_y)
        baseline_f1 = f1_score(val_y, baseline.predict(val_x.reshape(len(val_x), -1)), average="macro")
        print(f"Linear reference validation macro-F1: {baseline_f1:.3f}")
        """),
        md("""
        ## 4. Compare a small, predeclared set of experiments

        Recreate loaders and reset the seed for each run so initial weights and
        minibatch order match. Change only the learning rate. `fit` restores the
        epoch with the lowest validation loss within each run; macro-F1 chooses
        between the resulting models. This is a small teaching search, not an
        exhaustive claim that one learning rate is universally best.
        """),
        code("""
        configs = [{"lr": 0.001, "weight_decay": 0.0001},
                   {"lr": 0.003, "weight_decay": 0.0001}]
        epochs = 1 if SMOKE_TEST else 15
        experiments = []
        best_score, best_state, selected_config = -1.0, None, None
        for config in configs:
            seed_everything(PROJECT["seed"])
            train_loader, val_loader, test_loader = get_digits_loaders(seed=PROJECT["seed"])
            candidate = TinyCNN().to(device)
            history = fit(candidate, train_loader, val_loader, epochs=epochs,
                          device=device, **config)
            validation = evaluate(candidate, val_loader, device)
            macro_f1 = f1_score(validation["targets"].numpy(),
                                validation["logits"].argmax(1).numpy(), average="macro")
            experiments.append({**config, "epochs": epochs,
                                "val_loss": validation["loss"], "val_macro_f1": macro_f1})
            if macro_f1 > best_score:
                best_score, selected_config = float(macro_f1), dict(config)
                best_state = {k: v.detach().cpu().clone() for k, v in candidate.state_dict().items()}
        experiment_table = pd.DataFrame(experiments).sort_values("val_macro_f1", ascending=False)
        experiment_table
        """),
        md("""
        ## 5. Freeze the decision and evaluate the test set once

        After this point, do not change hyperparameters based on the test result.
        If you use errors to redesign the system, record that this set became
        development data and acquire a new final test set.
        """),
        code("""
        final_model = TinyCNN().to(device)
        final_model.load_state_dict(best_state)
        final_result = evaluate(final_model, test_loader, device)
        test_truth = final_result["targets"].numpy()
        test_predictions = final_result["logits"].argmax(1).numpy()
        test_f1 = f1_score(test_truth, test_predictions, average="macro")
        report = classification_report(test_truth, test_predictions,
                                       labels=list(range(10)), target_names=PROJECT["classes"],
                                       output_dict=True, zero_division=0)
        print(f"Final test accuracy: {final_result['accuracy']:.3f}")
        print(f"Final test macro-F1: {test_f1:.3f}")
        pd.DataFrame(report).T
        """),
        md("""
        ## 6. Show uncertainty and mistakes

        Bootstrap the final predictions to show how finite sample size affects
        accuracy. This approximate interval assumes independent examples and
        does not include variation from retraining, distribution shift, or
        repeated model selection. It cannot repair a leaked split.
        """),
        code("""
        rng = np.random.default_rng(42)
        correct = (test_truth == test_predictions).astype(float)
        bootstrap_accuracies = [rng.choice(correct, size=len(correct), replace=True).mean()
                                for _ in range(100 if SMOKE_TEST else 2000)]
        interval = np.quantile(bootstrap_accuracies, [0.025, 0.975]).tolist()
        print("Approximate 95% accuracy interval:", interval)
        output_dir = ARTIFACTS / "14_capstone"
        output_dir.mkdir(exist_ok=True)
        ConfusionMatrixDisplay.from_predictions(test_truth, test_predictions,
                                                labels=list(range(10)), cmap="Blues")
        plt.title("Final test confusion matrix")
        plt.tight_layout()
        plt.savefig(output_dir / "confusion_matrix.png", dpi=150)
        plt.show()
        """),
        code("""
        errors = np.flatnonzero(test_truth != test_predictions)
        print("Test mistakes:", len(errors))
        fig, axes = plt.subplots(1, 6, figsize=(12, 2.4))
        for ax in axes:
            ax.axis("off")
        for ax, index in zip(axes, errors[:6]):
            image, label = test_loader.dataset[int(index)]
            ax.imshow(image[0], cmap="gray")
            ax.set_title(f"True {int(label)} / pred {test_predictions[index]}")
        plt.tight_layout()
        plt.show()
        """),
        md("""
        ## 7. Export the project evidence

        The model report is JSON so it stays easy to read and compare. It records
        the task, dataset, split, chosen configuration, final metrics, preprocessing,
        and limitations. Add your own qualitative observations before sharing it.
        The checkpoint is compatible with `predict.py --checkpoint ...`.
        """),
        code("""
        model_report = {
            **PROJECT,
            "dataset": "scikit-learn bundled digits",
            "split": f"stratified 70/15/15, random_state={PROJECT['seed']}; tiny subset in smoke mode",
            "smoke_test": SMOKE_TEST,
            "selected_config": selected_config,
            "epochs_budget": epochs,
            "baseline_val_macro_f1": float(baseline_f1),
            "selected_val_macro_f1": best_score,
            "test_accuracy": final_result["accuracy"],
            "test_macro_f1": float(test_f1),
            "accuracy_bootstrap_interval": interval,
            "per_class_report": report,
            "torch_version": str(torch.__version__),
            "device": str(device),
            "preprocessing": "grayscale, bilinear resize to 8x8, light digit on dark background; uint8 pixels / 255",
            "limitations": ["Small teaching dataset", "Not writer-independent",
                            "Repeated course exposure to this split", "Not validated on photographs"],
            "my_error_analysis": "Add three observations grounded in the error images above.",
        }
        (output_dir / "model_report.json").write_text(json.dumps(model_report, indent=2), encoding="utf-8")
        experiment_table.to_csv(output_dir / "experiments.csv", index=False)
        torch.save({"architecture": "TinyCNN-v1", "state_dict": best_state,
                    "classes": PROJECT["classes"], "image_size": 8, "channels": 1,
                    "preprocessing": "grayscale, bilinear resize to 8x8, pixels / 255",
                    "smoke_test": SMOKE_TEST}, output_dir / "classifier.pt")
        from predict import load_bundle
        reloaded, _ = load_bundle(output_dir / "classifier.pt", device=device)
        probe = next(iter(val_loader))[0][:4].to(device)
        final_model.eval()
        with torch.inference_mode():
            torch.testing.assert_close(final_model(probe), reloaded(probe))
        print("Saved model reload verified on validation examples.")
        print("Exported files:", [p.name for p in output_dir.iterdir()])
        """),
        md("""
        ## 8. Your independent project

        Choose a small problem you can label reliably, such as three kinds of
        household object photographed against varied backgrounds.

        1. Write a task brief and a label guide. Collect lawful, appropriate images.
        2. Audit corruption, duplicates, class counts, and capture conditions.
        3. Create train/validation/test folders with the same class names. Keep
           related capture sessions together. Use notebook 08's loader and audit.
        4. Replace the data-loading cell. For RGB data use `TinyCNN(in_channels=3)`
           or adapt the pretrained ResNet workflow from notebook 07. Keep image
           transforms, class order, model architecture, and export code consistent.
        5. Compare a simple reference, a CNN, and a pretrained model using validation
           data. Select by a metric chosen before training.
        6. Freeze decisions, evaluate the test set once, inspect errors, and write
           a report with limitations and an explicit preprocessing contract.

        `predict.py` supports the **digit bundle only**. For an RGB/ResNet project,
        adapt it to load your architecture and use the exact saved transforms;
        do not silently resize photographs to 8×8 grayscale.
        """),
        md("""
        ## Practice and completion criteria

        **A.** Finish `my_error_analysis` with three concrete observations and
        rerun the export cell. Distinguish observations from proposed remedies.

        **B.** Before using any test result, run the validation experiment across
        three seeds. Report mean and spread, not just the best seed. For a fresh
        attempt, retain a new untouched final test set.

        **C.** Rebuild the project on your own image folders, then have someone
        else follow your README to make one prediction.
        """),
        code("""
        # Fill these notes from your evidence; no fabricated scores are needed.
        my_project_checklist = {
            "task_and_labels_documented": False,
            "split_and_duplicates_audited": False,
            "baseline_compared": False,
            "selection_uses_only_validation": False,
            "final_test_recorded_once": False,
            "three_error_examples_explained": False,
            "saved_model_reload_verified": False,
            "preprocessing_and_limitations_documented": False,
        }
        print("Completed:", sum(my_project_checklist.values()), "of", len(my_project_checklist))
        """),
        md("""
        <details><summary>Sample solution guidance</summary>

        Good error notes identify specific pairs such as true 8 predicted 3 and
        describe visible stroke ambiguity. A seed study repeats complete training
        and validation selection with each declared seed; it never picks the
        run with the best test score. For custom data, check class-to-index maps
        and inference normalization before interpreting any accuracy result.
        </details>

        **Assess your work:** can you justify the split, explain the training
        loop, reproduce your chosen model, show a failure, and state what your
        evidence does not establish? Those skills matter more than a single score.

        **References:** [scikit-learn common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html),
        [Model Cards paper](https://arxiv.org/abs/1810.03993).

        Return to [00 · Course roadmap](00_start_here_gpu_setup.ipynb) or extend
        [08 · Your own dataset](08_your_own_image_dataset.ipynb) with a new problem.
        """),
    ])


if __name__ == "__main__":
    build_start()
    build_explain()
    build_capstone()
