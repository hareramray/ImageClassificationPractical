"""Generate the four introductory, independently runnable course notebooks."""
from notebook_builder import md, code, bootstrap, write_notebook


def lesson01():
    return [
        md('''
        # 01 · From Python to image pixels

        **Level:** first steps · **Time:** 35–50 minutes · **Data:** bundled handwritten digits, no download.

        By the end you can use variables, loops and functions; inspect a NumPy image; plot it;
        and explain the difference between a pixel, an image and a class label.
        **Prerequisite:** open this notebook using the course kernel. No machine-learning experience needed.

        Run each cell with **Shift+Enter**, then change one value and rerun it. A notebook remembers
        variables in its kernel: after large edits, use **Restart Kernel and Run All** to check the full sequence.
        Blue/black text cells explain ideas; code cells execute Python. Errors are clues: read the last line first.
        ''') ,
        bootstrap(),
        md('''
        ## Step 1 — Give values names
        A variable keeps a value for later. Python infers its type: integers count things,
        floats hold decimal values, strings hold text, and booleans are `True` or `False`.
        A list stores an ordered sequence. Indexing starts at **zero**.
        ''') ,
        code('''
        image_height = 8                 # An integer
        pixel_fraction = 0.5             # A float
        dataset_name = "handwritten digits"
        is_grayscale = True              # One intensity channel
        class_names = [str(number) for number in range(10)]
        print(dataset_name, "|", image_height, "pixels high")
        print("First class:", class_names[0], "| last class:", class_names[-1])
        print(type(image_height).__name__, type(pixel_fraction).__name__)
        ''') ,
        md('''
        ## Step 2 — Repeat with a loop, reuse with a function
        Indentation marks which statements belong together. A `for` loop visits each value.
        A function accepts inputs and returns a result. Comments begin with `#` and do not run.
        Here, a conditional (`if`) makes a decision about each intensity.
        ''') ,
        code('''
        def describe_intensity(value):
            if value < 0.5:
                return "dark"
            return "bright"

        for value in [0.0, 0.25, 0.75, 1.0]:
            print(value, "is", describe_intensity(value))
        ''') ,
        md('''
        ## Step 3 — Load images and labels
        `load_digits` contains 1,797 tiny grayscale images of digits 0–9. Each image has
        8 rows × 8 columns. Its original intensities run from 0 to 16; its **label** tells us the digit.
        This classroom dataset is intentionally small. High scores here do not establish reliability
        on phone photographs, new handwriting styles or real applications.
        ''') ,
        code('''
        import numpy as np
        import matplotlib.pyplot as plt
        from PIL import Image
        from sklearn.datasets import load_digits

        digits = load_digits()
        raw_images = digits.images
        labels = digits.target
        print("Images:", raw_images.shape, "| labels:", labels.shape)
        print("Pixel range:", raw_images.min(), "to", raw_images.max())
        print("Classes:", np.unique(labels))
        ''') ,
        md('''
        ## Step 4 — Understand shape, dtype and indexing
        A NumPy array is a grid of values with one shared numeric type (`dtype`).
        Our shape is `(number_of_images, height, width)`. `images[0]` selects an image;
        `images[0, 2, 3]` selects its third row and fourth column.
        `:4` selects the first four entries. Make a `.copy()` before changing a slice
        if you want to preserve the original array.
        ''') ,
        code('''
        image_index = 0
        first_image = raw_images[image_index].copy()
        print("True digit:", labels[image_index])
        print("Image shape:", first_image.shape, "| dtype:", first_image.dtype)
        print("Top-left 4 by 4 pixels:\\n", first_image[:4, :4])
        print("Pixel at row 2, column 3:", first_image[2, 3])
        ''') ,
        md('''
        ## Step 5 — Normalize a known intensity range
        Divide every pixel by 16 to map this dataset's documented range to `[0, 1]`.
        `float32` uses 32 bits per number and is common for neural-network inputs.
        Array arithmetic applies to every element: no Python loop is needed.
        This fixed rescaling is different from **learning** a mean or variance from data;
        learned preprocessing must later be fitted on training examples only.
        ''') ,
        code('''
        images = raw_images.astype(np.float32) / 16.0
        print("Normalized:", images.dtype, images.min(), images.max())
        print("Mean intensity of first image:", round(float(images[0].mean()), 3))
        assert images.shape == (len(labels), 8, 8)
        assert 0 <= images.min() and images.max() <= 1
        ''') ,
        md('''
        ## Step 6 — See the numbers as pictures
        `imshow` assigns a color to each number; `cmap="gray"` uses black for low intensity
        and white for high intensity. Fixing `vmin` and `vmax` lets panels use the same brightness scale.
        Expect blocky but mostly recognizable handwritten digits.
        ''') ,
        code('''
        fig, axes = plt.subplots(2, 5, figsize=(10, 4))
        for index, ax in enumerate(axes.flat):
            ax.imshow(images[index], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            ax.set_title(f"Label: {labels[index]}")
            ax.axis("off")
        plt.tight_layout()
        plt.show()
        ''') ,
        md('''
        ## Step 7 — Convert between an array and a Pillow image
        Pillow reads and writes common image formats. An 8-bit grayscale PNG stores integer
        intensities 0–255. Resize with nearest-neighbor sampling to make the existing pixels
        easy to inspect; resizing does **not** recover new image detail.
        RGB photographs usually have shape `(height, width, 3)` in NumPy.
        ''') ,
        code('''
        pil_digit = Image.fromarray((images[0] * 255).round().astype(np.uint8))
        enlarged = pil_digit.resize((160, 160), resample=Image.Resampling.NEAREST)
        rgb_digit = pil_digit.convert("RGB")
        output_dir = ARTIFACTS / "01_pixels"
        output_dir.mkdir(parents=True, exist_ok=True)
        image_path = output_dir / "first_digit.png"
        enlarged.save(image_path)
        with Image.open(image_path) as reopened:
            reopened_array = np.asarray(reopened).copy()
        print("Pillow mode:", pil_digit.mode, "| RGB array shape:", np.asarray(rgb_digit).shape)
        print("Saved and reopened:", reopened_array.shape, image_path)
        plt.imshow(reopened_array, cmap="gray", vmin=0, vmax=255)
        plt.axis("off")
        plt.show()
        ''') ,
        md('''
        ## Step 8 — An image can become a feature vector
        Flattening an 8 × 8 grid creates 64 features in a fixed order. A traditional classifier
        can accept a table shaped `(examples, features)`. The values survive the reshape,
        but the model must learn useful relationships between positions.
        Convolutional networks later exploit the spatial grid directly.
        ''') ,
        code('''
        feature_table = images.reshape(len(images), -1)
        restored = feature_table[0].reshape(8, 8)
        print("Feature table:", feature_table.shape)
        print("Reshape preserved the pixels:", np.allclose(restored, images[0]))
        fig, ax = plt.subplots(figsize=(7, 2))
        ax.plot(feature_table[0], marker=".")
        ax.set(xlabel="Flattened pixel position", ylabel="Intensity", ylim=(-0.05, 1.05))
        plt.show()
        ''') ,
        md('''
        ## Practice 1 — Inspect a different digit
        Change `chosen_index` to 27. Print its label, mean and brightest pixel; show the image.
        Predict whether its average brightness will be closer to 0 or 1 before running.

        <details><summary>Sample answer — open after trying</summary>

        ```python
        chosen_index = 27
        chosen = images[chosen_index]
        print(labels[chosen_index], chosen.mean(), chosen.max())
        plt.imshow(chosen, cmap="gray", vmin=0, vmax=1)
        plt.show()
        ```

        Most pixels are dark background, so the average is usually closer to zero.
        </details>
        ''') ,
        code('''
        chosen_index = 12  # Change this; keep it between 0 and len(images) - 1.
        chosen = images[chosen_index]
        print("Your selected label:", labels[chosen_index])
        ''') ,
        md('''
        ## Practice 2 — Write a brightness function
        Write a function that returns the fraction of pixels brighter than a threshold.
        Apply it to three images using a loop. Hint: the mean of a boolean array is its fraction of `True` values.

        <details><summary>Sample answer</summary>

        ```python
        def bright_fraction(image, threshold=0.5):
            return float((image > threshold).mean())
        for index in [0, 1, 2]:
            print(index, bright_fraction(images[index]))
        ```

        `>` compares each pixel and produces a boolean mask with the same shape.
        </details>
        ''') ,
        code('''
        threshold = 0.5
        bright_mask = images[0] > threshold
        print("Starter mask shape:", bright_mask.shape)
        # Add your function and loop below.
        ''') ,
        md('''
        ## Practice 3 — Change pixels without changing your original
        Make an inverted copy, then make a copy with the top two rows black. Plot both.
        Explain why a label is separate from its pixel values: an edit can make an image ambiguous
        even when you keep its old label.

        <details><summary>Sample answer</summary>

        ```python
        inverted = 1.0 - images[0]
        cropped_ink = images[0].copy()
        cropped_ink[:2, :] = 0
        fig, axes = plt.subplots(1, 2)
        for ax, image in zip(axes, [inverted, cropped_ink]):
            ax.imshow(image, cmap="gray", vmin=0, vmax=1)
        plt.show()
        ```

        The label is supplied as supervision; the array itself contains no embedded class name.
        </details>
        ''') ,
        code('''
        edited_image = images[0].copy()
        print("Original still unchanged:", np.allclose(edited_image, images[0]))
        # Make and display your edits below.
        ''') ,
        md('''
        ## What you can now do
        You used Python variables, indexing, conditionals, loops and functions; inspected array shapes
        and dtypes; normalized, displayed, reshaped and saved an image. These operations are the
        input pipeline for a classifier. No classifier was trained in this lesson.

        **Checkpoint:** explain `(1797, 8, 8)` aloud, and explain why the number `7` as a label
        differs from a pixel whose intensity is `7`.

        **Next:** [02 · Classical machine-learning baselines](02_classical_ml_baselines.ipynb).

        Primary references: [Python tutorial](https://docs.python.org/3/tutorial/),
        [NumPy beginner guide](https://numpy.org/doc/stable/user/absolute_beginners.html),
        [digits dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html),
        [Matplotlib imshow](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imshow.html),
        [Pillow image API](https://pillow.readthedocs.io/en/stable/reference/Image.html).
        ''') ,
    ]


def lesson02():
    return [
        md('''
        # 02 · Honest baselines with classical machine learning

        **Level:** beginner · **Time:** 40–55 minutes · **Data:** bundled digits · **Hardware:** CPU is sufficient.

        You will split data before fitting, beat a trivial baseline, tune logistic regression using
        validation data, and read accuracy, macro F1 and a confusion matrix.
        **Prerequisite:** arrays and labels from notebook 01. This notebook creates its own data and models.

        A classifier learns a mapping from features to class labels. A useful experiment must also
        check performance on examples that did not influence fitting or model selection.
        ''') ,
        bootstrap(),
        md('''
        ## Step 1 — Set up the feature table
        Each row contains 64 pixel intensities; each target is an integer 0–9.
        Dividing by the dataset's known maximum is a fixed unit conversion, not a learned statistic.
        ''') ,
        code('''
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from sklearn.datasets import load_digits
        from sklearn.model_selection import train_test_split
        from sklearn.dummy import DummyClassifier
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import (accuracy_score, f1_score, classification_report,
                                     ConfusionMatrixDisplay)

        digits = load_digits()
        X = digits.data.astype(np.float32) / 16.0
        y = digits.target.astype(np.int64)
        print("Features:", X.shape, "| targets:", y.shape)
        ''') ,
        md('''
        ## Step 2 — Reserve training, validation and test sets
        We use about **70% training / 15% validation / 15% test**. Training fits parameters;
        validation selects choices such as regularization strength; the test set estimates final
        performance after those choices are fixed. `stratify` keeps class proportions similar.
        Splitting row indices lets us verify that the sets do not overlap.

        For this demonstration, random image splits are appropriate. For images grouped by person,
        patient, video or acquisition session, split by that group to avoid near-duplicate leakage.
        ''') ,
        code('''
        all_indices = np.arange(len(y))
        train_idx, holdout_idx = train_test_split(
            all_indices, test_size=0.30, stratify=y, random_state=42)
        val_idx, test_idx = train_test_split(
            holdout_idx, test_size=0.50,
            stratify=y[holdout_idx], random_state=42)

        if SMOKE_TEST:
            # Fast verification mode preserves all ten classes.
            train_idx = train_test_split(train_idx, train_size=128,
                                        stratify=y[train_idx], random_state=42)[0]
            val_idx = train_test_split(val_idx, train_size=64,
                                      stratify=y[val_idx], random_state=42)[0]
            test_idx = train_test_split(test_idx, train_size=64,
                                       stratify=y[test_idx], random_state=42)[0]

        assert not (set(train_idx) & set(val_idx) or set(train_idx) & set(test_idx)
                    or set(val_idx) & set(test_idx))
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        print("Train / validation / test:", len(y_train), len(y_val), len(y_test))
        ''') ,
        md('''
        ## Step 3 — Inspect training data, then choose metrics
        **Accuracy** is the fraction of correct predictions. **Macro F1** computes F1 separately
        for each class, then weights each class equally. F1 balances precision (how trustworthy a
        class prediction is) and recall (how many true members of a class we recover).
        Accuracy alone can conceal weak performance on rare classes.
        ''') ,
        code('''
        counts = np.bincount(y_train, minlength=10)
        plt.bar(np.arange(10), counts)
        plt.xticks(np.arange(10))
        plt.xlabel("Training class")
        plt.ylabel("Number of examples")
        plt.title("Training class balance")
        plt.show()

        def metric_row(name, target, predictions):
            return {"model": name, "accuracy": accuracy_score(target, predictions),
                    "macro_f1": f1_score(target, predictions, average="macro", zero_division=0)}
        ''') ,
        md('''
        ## Step 4 — Start with a deliberately simple baseline
        A `DummyClassifier` ignores the image and always predicts the most frequent training class.
        Expect roughly 10% accuracy because the ten classes are fairly balanced.
        A baseline checks whether a more elaborate model learns anything useful.
        ''') ,
        code('''
        dummy = DummyClassifier(strategy="most_frequent")
        dummy.fit(X_train, y_train)
        baseline_result = metric_row("Most frequent class", y_val, dummy.predict(X_val))
        pd.DataFrame([baseline_result])
        ''') ,
        md('''
        ## Step 5 — Put learned preprocessing inside a pipeline
        Logistic regression learns a linear score for each class and converts scores into class probabilities.
        Despite its name it is a **classification** method. `StandardScaler` learns one mean and
        standard deviation per feature. The pipeline fits both stages using **training data only**,
        then reuses those same statistics when predicting validation or test examples.

        Fitting the scaler on all images before splitting would leak information about held-out data.
        `C` controls inverse regularization strength: a smaller value penalizes large weights more.
        ''') ,
        code('''
        model = make_pipeline(StandardScaler(),
                              LogisticRegression(C=1.0, max_iter=2000, random_state=42))
        model.fit(X_train, y_train)
        val_predictions = model.predict(X_val)
        initial_result = metric_row("Logistic regression, C=1", y_val, val_predictions)
        print("Scaler fitted examples:", model.named_steps["standardscaler"].n_samples_seen_)
        pd.DataFrame([baseline_result, initial_result]).round(4)
        ''') ,
        md('''
        ## Step 6 — Compare a small, declared set of choices on validation data
        Try three values of `C`, choose the highest validation macro F1, and leave the test set alone.
        In normal mode this small dataset often gives high accuracy, but your exact numbers may vary.
        More validation trials can also overfit the validation set; record your choices.
        ''') ,
        code('''
        candidates = {}
        search_rows = []
        for strength in [0.1, 1.0, 10.0]:
            candidate = make_pipeline(
                StandardScaler(), LogisticRegression(C=strength, max_iter=2000, random_state=42))
            candidate.fit(X_train, y_train)
            candidates[strength] = candidate
            row = metric_row(f"C={strength}", y_val, candidate.predict(X_val))
            row["C"] = strength
            search_rows.append(row)

        search_table = pd.DataFrame(search_rows).sort_values(
            ["macro_f1", "C"], ascending=[False, True])
        best_C = float(search_table.iloc[0]["C"])
        final_model = candidates[best_C]
        print("Chosen C using validation only:", best_C)
        search_table.round(4)
        ''') ,
        md('''
        ## Step 7 — Evaluate the fixed model once on the test set
        The following predictions are the final evaluation for this experiment. We keep the chosen
        model fitted on training only, so the reported protocol is simple and explicit.
        Reusing these same stored predictions for a report, matrix and error gallery is one evaluation.
        Do not choose new hyperparameters from the test report.
        ''') ,
        code('''
        test_predictions = final_model.predict(X_test)
        final_metrics = metric_row(f"Final logistic regression C={best_C}", y_test, test_predictions)
        print(pd.DataFrame([final_metrics]).round(4).to_string(index=False))
        print(classification_report(y_test, test_predictions, labels=np.arange(10),
                                    target_names=[str(i) for i in range(10)], zero_division=0))
        ConfusionMatrixDisplay.from_predictions(
            y_test, test_predictions, labels=np.arange(10), cmap="Blues", colorbar=False)
        plt.title("Final test confusion matrix: rows=true, columns=predicted")
        plt.show()
        ''') ,
        md('''
        ## Step 8 — Inspect the mistakes
        Off-diagonal matrix entries are errors. Look at a few to distinguish ambiguous handwriting,
        preprocessing issues and systematic confusions. This is diagnosis; a new tuned experiment
        needs validation decisions and, for an unbiased final claim, fresh held-out evaluation data.
        ''') ,
        code('''
        mistakes = np.flatnonzero(test_predictions != y_test)
        fig, axes = plt.subplots(1, 6, figsize=(12, 2.5))
        for ax in axes:
            ax.axis("off")
        for ax, index in zip(axes, mistakes[:6]):
            ax.imshow(X_test[index].reshape(8, 8), cmap="gray", vmin=0, vmax=1)
            ax.set_title(f"True {y_test[index]} / predicted {test_predictions[index]}")
        plt.tight_layout()
        plt.show()
        print("Test mistakes:", len(mistakes), "out of", len(y_test))
        ''') ,
        md('''
        ## Practice 1 — Manually calculate baseline accuracy
        Count how often the dummy prediction matches a validation label. Compare your fraction to
        `baseline_result["accuracy"]`. Why is this model useful even though it is poor?

        <details><summary>Sample answer</summary>

        ```python
        guesses = dummy.predict(X_val)
        manual_accuracy = float((guesses == y_val).sum() / len(y_val))
        print(manual_accuracy, baseline_result["accuracy"])
        ```

        It reveals how much performance we get without using the input features.
        </details>
        ''') ,
        code('''
        guesses = dummy.predict(X_val)
        print("First ten guesses:", guesses[:10])
        # Calculate the fraction correct below.
        ''') ,
        md('''
        ## Practice 2 — Audit training-only scaling
        Compare the scaler's stored means to `X_train.mean(axis=0)`. Why should we avoid
        calling `fit_transform(X_val)`? Use `np.allclose` rather than exact float equality.

        <details><summary>Sample answer</summary>

        ```python
        scaler = final_model.named_steps["standardscaler"]
        print(np.allclose(scaler.mean_, X_train.mean(axis=0), atol=1e-6))
        scaled_val = scaler.transform(X_val)
        print(scaled_val.shape)
        ```

        Refitting on validation learns from held-out data and creates a different feature coordinate system.
        </details>
        ''') ,
        code('''
        scaler = final_model.named_steps["standardscaler"]
        print("Number of feature means:", len(scaler.mean_))
        # Audit the means and transform validation inputs below.
        ''') ,
        md('''
        ## Practice 3 — Explain one validation confusion
        Build a confusion matrix from **validation** predictions. Find the largest off-diagonal
        entry and inspect examples from that pair. Keep this as a validation-only diagnostic.

        <details><summary>Sample answer</summary>

        ```python
        from sklearn.metrics import confusion_matrix
        validation_guesses = final_model.predict(X_val)
        cm = confusion_matrix(y_val, validation_guesses, labels=np.arange(10))
        np.fill_diagonal(cm, 0)
        true_class, predicted_class = np.unravel_index(cm.argmax(), cm.shape)
        print(true_class, predicted_class, cm.max())
        ```

        If all off-diagonal counts are zero, there is no validation confusion to inspect.
        </details>
        ''') ,
        code('''
        validation_guesses = final_model.predict(X_val)
        print("Validation errors to investigate:", int((validation_guesses != y_val).sum()))
        # Build and inspect the matrix below; do not tune against the test results.
        ''') ,
        md('''
        ## What you learned
        You established a baseline, preserved separate data roles, used a leakage-resistant pipeline,
        selected regularization using validation macro F1 and inspected one final test evaluation.
        A neural network must earn its complexity by improving something useful: accuracy,
        robustness, deployment needs or the kinds of inputs it can represent.

        **Checkpoint:** explain why a 99% training accuracy alone is not evidence of generalization.

        **Next:** [03 · PyTorch tensors and automatic differentiation](03_pytorch_tensors_autograd.ipynb).

        Primary references: [data splitting](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html),
        [common pitfalls and leakage](https://scikit-learn.org/stable/common_pitfalls.html),
        [logistic regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html),
        [classification metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics).
        ''') ,
    ]


def lesson03():
    return [
        md('''
        # 03 · PyTorch tensors, gradients and your GPU

        **Level:** beginner to early intermediate · **Time:** 40–55 minutes · **Data:** tiny synthetic numbers and bundled digits.

        You will reason about tensor shapes, move data to a device, calculate gradients, and fit
        a straight line using gradient descent. These are the mechanics underneath neural networks.
        **Prerequisite:** notebook 01's arrays and basic functions. Everything is recreated here.

        Your local GPU can accelerate larger workloads. Tiny examples may run faster on the CPU
        because launching GPU work and transferring data also take time.
        ''') ,
        bootstrap(),
        md('''
        ## Step 1 — A tensor is a numeric array with extra abilities
        PyTorch tensors have a shape, dtype and device. They can participate in automatic
        differentiation and run on compatible accelerators. Float inputs commonly use `float32`;
        class labels for cross entropy use integer `int64` (also called `long`).
        ''') ,
        code('''
        import numpy as np
        import matplotlib.pyplot as plt
        import torch
        from course_utils import digit_splits

        scalar = torch.tensor(3.0)
        vector = torch.tensor([1.0, 2.0, 3.0])
        matrix = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        for name, tensor in [("scalar", scalar), ("vector", vector), ("matrix", matrix)]:
            print(name, "shape=", tuple(tensor.shape), "dtype=", tensor.dtype, "device=", tensor.device)
        ''') ,
        md('''
        ## Step 2 — Convert images and track every dimension
        A batch of grayscale images uses `(N, C, H, W)` in PyTorch: examples, channels, height,
        width. Our NumPy images have `(N, H, W)`, so `unsqueeze(1)` adds a channel dimension.
        `flatten(1)` preserves the batch dimension and flattens the remaining dimensions.
        ''') ,
        code('''
        splits = digit_splits()
        train_images, train_labels = splits["train"]
        batch_nhw = torch.from_numpy(train_images[:4].copy())
        batch_nchw = batch_nhw.unsqueeze(1)
        flat_batch = batch_nchw.flatten(1)
        batch_labels = torch.from_numpy(train_labels[:4].copy()).long()
        print("NHW:", tuple(batch_nhw.shape))
        print("NCHW:", tuple(batch_nchw.shape))
        print("Flattened:", tuple(flat_batch.shape), "| labels:", batch_labels.tolist())
        assert batch_nchw.shape == (4, 1, 8, 8)
        assert flat_batch.shape == (4, 64)
        ''') ,
        md('''
        ## Step 3 — Matrix multiplication creates class scores
        `(4, 64) @ (64, 10)` produces `(4, 10)`: ten scores for each of four images.
        A bias vector of shape `(10,)` broadcasts across all four rows.
        These random scores are **not trained predictions**. Elementwise `*` and matrix `@`
        are different operations.
        ''') ,
        code('''
        weights = torch.randn(64, 10) * 0.01
        bias = torch.zeros(10)
        scores = flat_batch @ weights + bias
        print("Class score shape:", tuple(scores.shape))
        print("First image scores:", scores[0])
        print("Elementwise square:", matrix * matrix)
        print("Matrix product:\\n", matrix @ matrix)
        ''') ,
        md('''
        ## Step 4 — Place compatible tensors on the same device
        `.to(device)` returns a tensor on that device; assign the returned value.
        Model parameters and their inputs must be colocated. `.cpu()` returns data to CPU for
        NumPy and plotting. When gradients are involved, `.detach()` first removes the graph connection.
        ''') ,
        code('''
        accelerator_batch = flat_batch.to(device)
        accelerator_weights = weights.to(device)
        accelerator_bias = bias.to(device)
        accelerator_scores = accelerator_batch @ accelerator_weights + accelerator_bias
        scores_numpy = accelerator_scores.detach().cpu().numpy()
        print("Calculation device:", accelerator_scores.device, "| NumPy shape:", scores_numpy.shape)
        if device.type == "cuda":
            print("GPU:", torch.cuda.get_device_name(0))
        print("CPU and selected-device results agree:", np.allclose(
            scores_numpy, scores.numpy(), atol=1e-5))
        ''') ,
        md('''
        ## Step 5 — Ask autograd for a derivative
        For `f(w) = (w − 3)²`, the derivative is `2(w − 3)`. At `w=0` it is −6.
        A negative derivative means increasing `w` locally reduces this loss.
        `requires_grad=True` records operations; `.backward()` computes derivatives and
        adds them into each leaf tensor's `.grad` field.
        ''') ,
        code('''
        w = torch.tensor(0.0, requires_grad=True, device=device)
        loss = (w - 3.0) ** 2
        loss.backward()
        print("Loss:", loss.item(), "| gradient:", w.grad.item())
        assert np.isclose(w.grad.item(), -6.0)
        ''') ,
        md('''
        ## Step 6 — Gradients accumulate unless you clear them
        A fresh forward calculation creates a fresh graph, but another backward call adds to
        existing gradients. Training normally clears gradients each step. The default graph
        is freed after backward, so we recompute the loss before another backward call.
        ''') ,
        code('''
        second_loss = (w - 3.0) ** 2
        second_loss.backward()
        print("After two backward calls:", w.grad.item())
        w.grad = None
        fresh_loss = (w - 3.0) ** 2
        fresh_loss.backward()
        print("After clearing and recomputing:", w.grad.item())
        ''') ,
        md('''
        ## Step 7 — Take gradient-descent steps
        Update a parameter with `parameter -= learning_rate * gradient`. The learning rate controls
        step size. We update inside `torch.no_grad()` because parameter updates themselves should
        not create a new differentiation graph. Watch the parameter approach 3 and the loss approach 0.
        ''') ,
        code('''
        w = torch.tensor(0.0, requires_grad=True, device=device)
        learning_rate = 0.1
        scalar_history = []
        for step in range(25):
            w.grad = None
            loss = (w - 3.0) ** 2
            loss.backward()
            scalar_history.append({"step": step, "w": w.item(), "loss": loss.item()})
            with torch.no_grad():
                w -= learning_rate * w.grad
        print("Final parameter:", round(w.item(), 4))
        plt.plot([entry["loss"] for entry in scalar_history], marker=".")
        plt.xlabel("Update step")
        plt.ylabel("Squared distance from 3")
        plt.show()
        ''') ,
        md('''
        ## Step 8 — Fit two parameters to noisy observations
        Synthetic targets follow approximately `y = 2x + 1`. The model predicts `a*x+b`;
        mean squared error averages the squared prediction errors.
        This is a mechanics demonstration, not a generalization experiment: we fit and display
        the same tiny training sample. The next notebook restores a proper validation/test protocol.
        ''') ,
        code('''
        seed_everything(42)
        x = torch.linspace(-1, 1, 80, device=device).unsqueeze(1)
        target = 2.0 * x + 1.0 + 0.08 * torch.randn_like(x)
        a = torch.zeros(1, device=device, requires_grad=True)
        b = torch.zeros(1, device=device, requires_grad=True)
        regression_losses = []

        for step in range(120):
            prediction = a * x + b
            loss = ((prediction - target) ** 2).mean()
            loss.backward()
            with torch.no_grad():
                a -= 0.1 * a.grad
                b -= 0.1 * b.grad
            a.grad = None
            b.grad = None
            regression_losses.append(loss.item())

        print(f"Learned a={a.item():.3f}, b={b.item():.3f}; generating values were 2 and 1")
        with torch.no_grad():
            fitted = a * x + b
        fig, axes = plt.subplots(1, 2, figsize=(10, 3))
        axes[0].scatter(x.cpu().numpy(), target.cpu().numpy(), s=12, label="Observations")
        axes[0].plot(x.cpu().numpy(), fitted.cpu().numpy(), color="orange", label="Fitted line")
        axes[0].legend()
        axes[0].set(xlabel="x", ylabel="y")
        axes[1].plot(regression_losses)
        axes[1].set(xlabel="Step", ylabel="Training mean squared error")
        plt.tight_layout()
        plt.show()
        ''') ,
        md('''
        ## Practice 1 — Predict a batch shape
        Convert the first seven training images to `(7, 1, 8, 8)` and then to `(7, 64)`.
        Write down the output shape before running each operation.

        <details><summary>Sample answer</summary>

        ```python
        seven = torch.from_numpy(train_images[:7].copy()).unsqueeze(1)
        print(seven.shape, seven.flatten(1).shape)
        ```

        Flattening from dimension 0 would merge examples together, losing the batch boundary.
        </details>
        ''') ,
        code('''
        seven_nhw = torch.from_numpy(train_images[:7].copy())
        print("Starting shape:", tuple(seven_nhw.shape))
        # Add the channel, then flatten only the feature dimensions.
        ''') ,
        md('''
        ## Practice 2 — Check a derivative by hand
        For `f(t)=t³+2t`, calculate the derivative at `t=2`, then check it with autograd.
        Which tensor needs `requires_grad=True`?

        <details><summary>Sample answer</summary>

        ```python
        t = torch.tensor(2.0, requires_grad=True)
        f = t**3 + 2*t
        f.backward()
        print(t.grad.item())  # 3*(2**2)+2 = 14
        ```

        The input parameter `t` requires gradients; `f` inherits a graph through its operations.
        </details>
        ''') ,
        code('''
        trial_value = 2.0
        manual_derivative = 3 * trial_value**2 + 2
        print("Hand calculation:", manual_derivative)
        # Confirm using a fresh tensor and backward().
        ''') ,
        md('''
        ## Practice 3 — Explore learning-rate behavior
        Repeat the scalar `w` experiment with rates `0.01`, `0.1` and `1.1` for 15 steps each.
        Record the loss **after** the last update. Which is slow, which converges, which grows?

        <details><summary>Sample answer</summary>

        ```python
        for lr in [0.01, 0.1, 1.1]:
            trial = torch.tensor(0.0, requires_grad=True)
            for step in range(15):
                trial.grad = None
                objective = (trial - 3)**2
                objective.backward()
                with torch.no_grad():
                    trial -= lr * trial.grad
            print(lr, float((trial.detach() - 3)**2))
        ```

        For this quadratic, `1.1` overshoots and diverges. A useful learning rate depends on the objective.
        </details>
        ''') ,
        code('''
        learning_rates_to_try = [0.01, 0.1, 1.1]
        print("Compare these rates:", learning_rates_to_try)
        # Build independent trial tensors so every run starts at zero.
        ''') ,
        md('''
        ## What you learned
        You tracked tensor dimensions, matrix multiplication and device placement, inspected
        `.grad`, cleared accumulated gradients, and used backward passes to learn parameters.
        Neural-network layers simply organize larger collections of these differentiable operations.

        **Checkpoint:** state the sequence: forward → loss → backward → update, and explain
        where gradient clearing belongs.

        **Next:** [04 · Your first neural image classifier](04_first_neural_classifier.ipynb).

        Primary references: [PyTorch tensors](https://docs.pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html),
        [autograd](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html),
        [optimization](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html),
        [no_grad](https://docs.pytorch.org/docs/stable/generated/torch.no_grad.html).
        ''') ,
    ]


def lesson04():
    return [
        md('''
        # 04 · Train your first neural image classifier

        **Level:** early intermediate · **Time:** 45–65 minutes · **Data:** bundled digits · **Hardware:** local GPU or CPU.

        Build a multilayer perceptron (MLP), follow an explicit training loop, inspect learning curves,
        and save a reproducible checkpoint. An MLP connects flattened pixels through learned linear
        layers and a nonlinear activation. **Prerequisite:** tensors and autograd from notebook 03.
        This notebook independently loads all data and trains its own model.

        Run the default experiment once before modifying it. Normal mode uses eight epochs;
        `COURSE_SMOKE_TEST=1` uses one epoch and much less data, so its scores are not learning results.
        ''') ,
        bootstrap(),
        md('''
        ## Step 1 — Load separate datasets in mini-batches
        A `DataLoader` groups examples into batches. Training shuffles their order; validation and
        test use a fixed order. Our shared loader uses `num_workers=0` for a simple Windows setup.
        Each image is `(1, 8, 8)` and each label is a scalar integer. This dataset needs no download.
        ''') ,
        code('''
        import copy
        import json
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import torch
        from torch import nn
        from sklearn.metrics import accuracy_score, f1_score, ConfusionMatrixDisplay
        from course_utils import get_digits_loaders

        seed_everything(42)
        train_loader, val_loader, test_loader = get_digits_loaders(batch_size=64)
        sample_images, sample_labels = next(iter(train_loader))
        print("Batch:", tuple(sample_images.shape), "labels:", tuple(sample_labels.shape))
        print("Examples:", {name: len(loader.dataset) for name, loader in
                            [("train", train_loader), ("validation", val_loader), ("test", test_loader)]})
        print("Dtypes:", sample_images.dtype, sample_labels.dtype)
        ''') ,
        md('''
        ## Step 2 — Build a small network and count its parameters
        `Flatten` converts each image into 64 inputs. `Linear(64, 64)` learns weighted combinations,
        `ReLU` keeps positive activations and replaces negative ones with zero, and `Linear(64, 10)`
        produces one raw score (a **logit**) per class. The nonlinearity lets stacked layers represent
        more than one linear mapping.
        ''') ,
        code('''
        class DigitMLP(nn.Module):
            def __init__(self, hidden_size=64):
                super().__init__()
                self.network = nn.Sequential(
                    nn.Flatten(), nn.Linear(8 * 8, hidden_size),
                    nn.ReLU(), nn.Linear(hidden_size, 10))

            def forward(self, images):
                return self.network(images)

        model = DigitMLP(hidden_size=64).to(device)
        print(model)
        print("Trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
        print("Expected: (64*64 + 64) + (64*10 + 10) =", 64*64 + 64 + 64*10 + 10)
        ''') ,
        md('''
        ## Step 3 — Keep logits for the loss; use softmax for inspection
        `CrossEntropyLoss` expects raw logits of shape `(batch, 10)` and integer targets of shape
        `(batch,)`. It performs the required normalization internally. **Do not add softmax to the
        model before this loss.** For display, softmax converts logits into probabilities that sum to 1.
        `argmax` picks the same class from either logits or softmax values.
        ''') ,
        code('''
        criterion = nn.CrossEntropyLoss()
        with torch.no_grad():
            sample_logits = model(sample_images.to(device))
            sample_probabilities = sample_logits.softmax(dim=1)
            starting_loss = criterion(sample_logits, sample_labels.to(device)).item()
        print("Logits shape:", tuple(sample_logits.shape))
        print("Probability sums:", sample_probabilities[:4].sum(dim=1).cpu().numpy())
        print("Untrained batch loss:", round(starting_loss, 4), "| log(10):", round(np.log(10), 4))
        print("Same argmax:", torch.equal(sample_logits.argmax(1), sample_probabilities.argmax(1)))
        ''') ,
        md('''
        ## Step 4 — Declare the experiment before fitting
        Adam adjusts step sizes for individual parameters. One **epoch** visits the training dataset once;
        one **step** updates weights from one mini-batch. We select the checkpoint with the smallest
        validation loss over the declared eight epochs. Test data plays no part in this decision.
        ''') ,
        code('''
        epochs = 1 if SMOKE_TEST else 8
        learning_rate = 0.003
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        history = []
        best_val_loss = float("inf")
        best_state = None
        best_epoch = None
        print("Epochs:", epochs, "| training steps per epoch:", len(train_loader))
        ''') ,
        md('''
        ## Step 5 — Read and run the complete training loop
        Follow the numbered statements: move the batch → clear old gradients → forward pass →
        calculate loss → backward pass → update weights. `model.train()` enables training behavior.
        Validation uses `model.eval()` and `torch.inference_mode()` to avoid gradient recording.

        We multiply each batch's mean loss by its batch size before averaging, so a smaller final
        batch receives the correct weight. Training metrics average batches while parameters are
        changing; validation measures the model at the end of the epoch.
        ''') ,
        code('''
        for epoch in range(1, epochs + 1):
            model.train()
            train_loss_sum, train_correct, train_count = 0.0, 0, 0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)  # 1. Device
                optimizer.zero_grad(set_to_none=True)                # 2. Clear
                logits = model(images)                              # 3. Forward
                loss = criterion(logits, labels)                     # 4. Loss
                loss.backward()                                      # 5. Backward
                optimizer.step()                                     # 6. Update

                batch_size = labels.size(0)
                train_loss_sum += loss.item() * batch_size
                train_correct += (logits.argmax(1) == labels).sum().item()
                train_count += batch_size

            model.eval()
            val_loss_sum, val_correct, val_count = 0.0, 0, 0
            with torch.inference_mode():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    logits = model(images)
                    loss = criterion(logits, labels)
                    batch_size = labels.size(0)
                    val_loss_sum += loss.item() * batch_size
                    val_correct += (logits.argmax(1) == labels).sum().item()
                    val_count += batch_size

            row = {"epoch": epoch, "train_loss": train_loss_sum / train_count,
                   "val_loss": val_loss_sum / val_count,
                   "train_accuracy": train_correct / train_count,
                   "val_accuracy": val_correct / val_count}
            history.append(row)
            if row["val_loss"] < best_val_loss:
                best_val_loss = row["val_loss"]
                best_epoch = epoch
                # A deep copy preserves the selected weights while later epochs keep training.
                best_state = copy.deepcopy(model.state_dict())
            print(f"Epoch {epoch:02d}: train loss={row['train_loss']:.4f}, "
                  f"val loss={row['val_loss']:.4f}, val accuracy={row['val_accuracy']:.3f}")

        model.load_state_dict(best_state)
        model.eval()
        print("Restored validation-selected epoch:", best_epoch)
        ''') ,
        md('''
        ## Step 6 — Read learning curves
        Expect loss to fall and accuracy to rise in normal mode. If training improves while
        validation deteriorates, the model may be overfitting. Both staying poor can indicate an
        optimization problem, unsuitable representation or a data bug. These are clues, not guarantees.
        A single smoke-test epoch verifies execution; it cannot reveal a meaningful curve.
        ''') ,
        code('''
        history_table = pd.DataFrame(history)
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
        for name in ["train", "val"]:
            axes[0].plot(history_table["epoch"], history_table[f"{name}_loss"], marker="o", label=name)
            axes[1].plot(history_table["epoch"], history_table[f"{name}_accuracy"], marker="o", label=name)
        axes[0].set(xlabel="Epoch", ylabel="Cross-entropy loss")
        axes[1].set(xlabel="Epoch", ylabel="Accuracy", ylim=(0, 1.05))
        for ax in axes:
            ax.legend()
            ax.grid(alpha=0.2)
        plt.tight_layout()
        plt.show()
        history_table.round(4)
        ''') ,
        md('''
        ## Step 7 — Freeze the choice and evaluate once on held-out test data
        Collect logits, labels and images in one pass. The following metrics and plots all reuse
        those predictions. Macro F1 gives each class equal weight. Do not change training settings
        because of this score; choose experiment changes using validation data.
        ''') ,
        code('''
        all_logits, all_targets, all_images = [], [], []
        with torch.inference_mode():
            for images, labels in test_loader:
                all_logits.append(model(images.to(device)).cpu())
                all_targets.append(labels)
                all_images.append(images)
        test_logits = torch.cat(all_logits)
        test_targets = torch.cat(all_targets).numpy()
        test_images = torch.cat(all_images).numpy()
        test_predictions = test_logits.argmax(1).numpy()
        test_probabilities = test_logits.softmax(1).numpy()
        test_metrics = {
            "accuracy": float(accuracy_score(test_targets, test_predictions)),
            "macro_f1": float(f1_score(test_targets, test_predictions, average="macro", zero_division=0))}
        print("Final test metrics:", test_metrics)
        ConfusionMatrixDisplay.from_predictions(
            test_targets, test_predictions, labels=np.arange(10), cmap="Blues", colorbar=False)
        plt.title("Final test confusion matrix")
        plt.show()
        ''') ,
        md('''
        ## Step 8 — Inspect predictions and save the experiment
        A high softmax value is the model's confidence, not proof of correctness. We will later
        study calibration. Saving a `state_dict` stores learned parameter tensors; architecture and
        preprocessing must also be known when loading. Save portable CPU weights plus useful metadata.
        ''') ,
        code('''
        fig, axes = plt.subplots(2, 5, figsize=(11, 4))
        for index, ax in enumerate(axes.flat):
            prediction = test_predictions[index]
            confidence = test_probabilities[index, prediction]
            ax.imshow(test_images[index, 0], cmap="gray", vmin=0, vmax=1)
            ax.set_title(f"True {test_targets[index]}, pred {prediction}\\np={confidence:.2f}",
                         color="green" if prediction == test_targets[index] else "red")
            ax.axis("off")
        plt.tight_layout()
        plt.show()

        output_dir = ARTIFACTS / "04_mlp"
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = output_dir / "digit_mlp.pt"
        torch.save({"model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                    "hidden_size": 64, "classes": list(range(10)), "input_shape": [1, 8, 8],
                    "preprocessing": "sklearn digits pixel values divided by 16",
                    "selected_epoch": best_epoch, "seed": 42,
                    "smoke_test": SMOKE_TEST}, checkpoint_path)
        history_table.to_csv(output_dir / "history.csv", index=False)
        (output_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")
        print("Saved:", checkpoint_path)
        ''') ,
        md('''
        ## Practice 1 — Verify that saving and loading preserves predictions
        Instantiate a fresh `DigitMLP`, load the saved state and compare its logits to the trained
        model on a validation batch. Use `weights_only=True` for this tensor-and-primitive checkpoint.
        Only load checkpoint files you trust.

        <details><summary>Sample answer</summary>

        ```python
        saved = torch.load(checkpoint_path, map_location=device, weights_only=True)
        restored = DigitMLP(saved["hidden_size"]).to(device)
        restored.load_state_dict(saved["model_state"])
        restored.eval()
        probe_images, _ = next(iter(val_loader))
        with torch.inference_mode():
            print(torch.allclose(model(probe_images.to(device)), restored(probe_images.to(device))))
        ```

        Identical architecture, weights, mode and preprocessing should give matching predictions.
        </details>
        ''') ,
        code('''
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        print("Checkpoint architecture:", saved["input_shape"], "hidden units:", saved["hidden_size"])
        # Create a fresh model, load its state, and compare validation logits.
        ''') ,
        md('''
        ## Practice 2 — Explain one manual cross-entropy calculation
        Take the first validation example, convert logits to log probabilities with `log_softmax`,
        and negate the log probability assigned to its true class. Compare with `criterion` on that example.

        <details><summary>Sample answer</summary>

        ```python
        images, labels = next(iter(val_loader))
        with torch.inference_mode():
            logits = model(images[:1].to(device))
            manual_loss = -logits.log_softmax(1)[0, int(labels[0])]
            library_loss = criterion(logits, labels[:1].to(device))
        print(manual_loss.item(), library_loss.item())
        ```

        Assigning a higher probability to the correct class lowers this loss.
        </details>
        ''') ,
        code('''
        probe_images, probe_labels = next(iter(val_loader))
        print("True class of the probe:", int(probe_labels[0]))
        # Compare manual and library cross entropy below.
        ''') ,
        md('''
        ## Practice 3 — Plan a controlled architecture comparison
        Count parameters for hidden widths 16, 64 and 128. Predict the effect on capacity and
        overfitting. As a further experiment, restart from the same seed, train each width using
        the same split and learning rate, and compare **validation** loss. Skip the test cell while tuning.

        <details><summary>Sample answer</summary>

        ```python
        for width in [16, 64, 128]:
            trial_model = DigitMLP(hidden_size=width)
            print(width, sum(p.numel() for p in trial_model.parameters()))
        ```

        Counts are 1,210; 4,810; and 9,610. More parameters increase capacity, but do not guarantee
        better held-out performance. Record validation results before selecting a model.
        </details>
        ''') ,
        code('''
        widths_to_compare = [16, 64, 128]
        print("Declared comparison:", widths_to_compare)
        # Count parameters now; use a fresh kernel for each later training trial.
        ''') ,
        md('''
        ## What you learned
        You built a neural network, distinguished logits from probabilities, trained in mini-batches,
        restored the validation-selected checkpoint, examined held-out predictions and saved artifacts.
        You can now read the core code behind most supervised image-classification training loops.

        **Checkpoint:** explain why validation has no `backward()` or `optimizer.step()`, and why
        `model.eval()` and disabling gradient recording solve different problems.

        **Next:** [05 · Convolutional neural networks](05_convolutional_neural_networks.ipynb).

        Primary references: [building a model](https://docs.pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html),
        [training loops](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html),
        [cross entropy](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html),
        [saving and loading](https://docs.pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html).
        ''') ,
    ]


if __name__ == "__main__":
    for filename, build in [
        ("01_image_pixels_numpy.ipynb", lesson01),
        ("02_classical_ml_baselines.ipynb", lesson02),
        ("03_pytorch_tensors_autograd.ipynb", lesson03),
        ("04_first_neural_classifier.ipynb", lesson04),
    ]:
        write_notebook(filename, build())
