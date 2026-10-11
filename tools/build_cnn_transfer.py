"""Build the CNN, regularization, transfer, and custom-data course notebooks."""
import inspect
from notebook_builder import md, code, bootstrap, write_notebook


def exercises(items):
    cells = []
    for title, task, starter, answer in items:
        cells += [md(f"### Exercise: {title}\n\n{task}"), code(inspect.cleandoc(starter)),
                  md(f"<details><summary>Sample answer — open after trying</summary>\n\n{answer}\n\n</details>")]
    return cells


cells = [
    md('''
    # 05 — Build and understand a convolutional neural network

    **Prerequisites:** notebooks 01–04: tensors, data splits, cross-entropy, and gradients.
    **Goals:** trace image shapes, build a CNN, write its training loop, and inspect learned filters.
    **Time:** about 20–35 minutes including exercises. The built-in 8×8 digits need no download.

    Run cells in order. Read each prediction before running its code. The numbers you obtain depend
    on hardware and random seeds; learning curves matter more than a particular score.
    ''')
    , bootstrap()
    , code('''
    import copy
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import torch
    from torch import nn
    from course_utils import get_digits_loaders, evaluate

    train_loader, val_loader, test_loader = get_digits_loaders(batch_size=64)
    xb, yb = next(iter(train_loader))
    print("Images N,C,H,W:", tuple(xb.shape), "| labels:", tuple(yb.shape))
    fig, axes = plt.subplots(1, 8, figsize=(12, 2))
    for ax, x, y in zip(axes, xb[:8], yb[:8]):
        ax.imshow(x[0], cmap="gray", vmin=0, vmax=1)
        ax.set_title(str(y.item())); ax.axis("off")
    plt.show()
    ''')
    , md('''
    ## 1. Follow the dimensions before training

    A convolution slides the same small weight grid over the image: it learns a local pattern
    at many positions. Channels represent different learned patterns, not separate images.
    For dilation 1, one spatial output dimension is `floor((input + 2×padding − kernel)/stride) + 1`.
    With a 3×3 kernel, stride 1, and padding 1, height and width stay unchanged.

    Predict: `N×1×8×8 → N×16×8×8 → N×16×4×4 → N×32×4×4 → N×32×2×2 → N×128 → N×10`.
    Pooling reduces spatial detail. ReLU introduces nonlinearity. The final layer returns **logits**;
    `CrossEntropyLoss` expects logits and integer labels, so do not insert softmax before the loss.
    ''')
    , code('''
    class DigitCNN(nn.Module):
        def __init__(self, width=16):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, width, kernel_size=3, padding=1), nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(width, width * 2, kernel_size=3, padding=1), nn.ReLU(),
                nn.AdaptiveAvgPool2d((2, 2)),
            )
            self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(width * 2 * 2 * 2, 10))

        def forward(self, x):
            return self.classifier(self.features(x))

    model = DigitCNN().to(device)
    z = xb[:2].to(device)
    for layer in [*model.features, *model.classifier]:
        z = layer(z)
        print(f"{layer.__class__.__name__:20s} {tuple(z.shape)}")
    print("Trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    ''')
    , md('''
    ## 2. Write the training loop yourself

    Every batch performs: move data → predict → compute loss → clear old gradients → backpropagate
    → update parameters. `train()` enables training behavior; `eval()` selects inference behavior.
    `inference_mode()` avoids constructing a gradient graph during validation.

    We weight batch losses by batch size because the final batch may be smaller. Validation selects
    the best epoch; the test set stays untouched until this model is selected.
    ''')
    , code('''
    def run_epoch(model, loader, optimizer=None):
        training = optimizer is not None
        model.train(training)
        loss_total, correct, count = 0.0, 0, 0
        context = torch.enable_grad() if training else torch.inference_mode()
        with context:
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                loss = nn.functional.cross_entropy(logits, labels)
                if training:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                loss_total += loss.item() * len(labels)
                correct += (logits.argmax(1) == labels).sum().item()
                count += len(labels)
        return loss_total / count, correct / count

    seed_everything(42)
    model = DigitCNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4)
    epochs = 2 if SMOKE_TEST else 12
    history, best_loss, best_state = [], float("inf"), None
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader)
        history.append(dict(epoch=epoch, train_loss=train_loss, val_loss=val_loss,
                            train_acc=train_acc, val_acc=val_acc))
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
        print(f"Epoch {epoch:02d} | train {train_loss:.3f}/{train_acc:.1%} | val {val_loss:.3f}/{val_acc:.1%}")
    model.load_state_dict(best_state)
    ''')
    , md('''
    ## 3. Interpret training and validation together

    Expect loss to decrease from roughly `log(10) ≈ 2.30`, although the initial value varies.
    If training improves but validation worsens, suspect overfitting. If neither improves, check
    image ranges, labels, shapes, learning rate, and whether `optimizer.step()` runs.
    Smoke mode checks wiring only and is too short to judge accuracy.
    ''')
    , code('''
    history_df = pd.DataFrame(history).set_index("epoch")
    fig, axes = plt.subplots(1, 2, figsize=(11, 3))
    history_df[["train_loss", "val_loss"]].plot(ax=axes[0], title="Cross-entropy")
    history_df[["train_acc", "val_acc"]].plot(ax=axes[1], title="Accuracy")
    axes[1].set_ylim(0, 1); plt.tight_layout(); plt.show()
    history_df.tail()
    ''')
    , code('''
    # One final test evaluation after choosing the epoch from validation loss.
    test_result = evaluate(model, test_loader, device=device)
    print(f"Selected model test accuracy: {test_result['accuracy']:.1%}")
    predictions = test_result["logits"].argmax(1)
    mistakes = (predictions != test_result["targets"]).nonzero().flatten()
    print("Test errors:", len(mistakes), "of", len(predictions))
    ''')
    , md('''
    ## 4. Inspect filters and activations

    The first-layer kernels are only 3×3, so do not expect each one to have an obvious human meaning.
    Activations show where a learned filter responds in one image. These plots describe internal
    calculations; they do not prove which visual evidence caused a decision.
    ''')
    , code('''
    model.eval()
    with torch.inference_mode():
        feature_maps = model.features[:2](xb[:1].to(device))[0].cpu()
    kernels = model.features[0].weight.detach().cpu()
    fig, axes = plt.subplots(2, 8, figsize=(12, 4))
    for j in range(8):
        axes[0, j].imshow(kernels[j, 0], cmap="coolwarm")
        axes[1, j].imshow(feature_maps[j], cmap="viridis")
        axes[0, j].set_title(f"Filter {j}")
        axes[0, j].axis("off"); axes[1, j].axis("off")
    plt.tight_layout(); plt.show()
    ''')
    , md('''
    ## 5. Save architecture information with weights

    A `state_dict` stores parameters, not the Python class definition. Reload into the same
    architecture. This checkpoint also records the expected image range, shape, and class order.
    ''')
    , code('''
    checkpoint_path = ARTIFACTS / "05_digit_cnn.pt"
    torch.save({"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "architecture": "DigitCNN(width=16)", "classes": [str(i) for i in range(10)],
                "input_shape": [1, 8, 8], "input_range": [0.0, 1.0]}, checkpoint_path)
    print(checkpoint_path)
    ''')
]
cells += exercises([
    ("Change channel width", "Predict how doubling width changes parameter count. Build both models; do not retrain yet.",
     '''widths = [8, 16, 32]
     parameter_counts = {w: sum(p.numel() for p in DigitCNN(w).parameters()) for w in widths}
     print(parameter_counts)
     # TODO: Explain why parameter count grows faster than width.''',
     "The first convolution grows linearly with width, but the second connects `width` input channels to `2×width` outputs, so its weights grow quadratically. More parameters do not guarantee better validation accuracy."),
    ("Check softmax", "Turn the logits for one batch into probabilities. Verify that each row sums to one, and compare predicted classes.",
     '''with torch.inference_mode():
         example_logits = model(xb[:4].to(device)).cpu()
     probabilities = example_logits.softmax(dim=1)
     print("Probability sums:", probabilities.sum(1))
     # TODO: Compare example_logits.argmax(1) and probabilities.argmax(1).''',
     "```python\nassert torch.equal(example_logits.argmax(1), probabilities.argmax(1))\n```\nSoftmax preserves the largest element's index. Use probabilities for display and logits for cross-entropy."),
    ("Try a pooling alternative", "Replace max pooling with average pooling in a fresh model. Optionally train it with the same seed and epoch budget; compare validation loss, without repeatedly consulting the test set.",
     '''alternative = DigitCNN().to(device)
     alternative.features[2] = nn.AvgPool2d(2)
     print("Output shape:", alternative(xb[:2].to(device)).shape)
     RUN_POOLING_EXPERIMENT = False
     if RUN_POOLING_EXPERIMENT:
         from course_utils import fit
         seed_everything(42)
         alternative = DigitCNN().to(device)
         alternative.features[2] = nn.AvgPool2d(2)
         alternative_history = fit(alternative, train_loader, val_loader, epochs=epochs,
                                   lr=0.003, weight_decay=1e-4, device=device)
         print(pd.DataFrame(alternative_history).tail())''',
     "Max pooling retains the strongest response in each window; average pooling smooths responses. Reinitialize both comparisons with the same seed, reset data order, and compare several seeds before claiming one is better."),
])
cells += [md('''
    **Checkpoint:** explain `N,C,H,W`, predict each layer's shape, and describe one complete optimizer step.

    **Optional next dataset:** after mastering the loop, try `torchvision.datasets.FashionMNIST`
    (download required) with `ToTensor()`, a fixed stratified validation split from its training partition,
    and a separate official test set. Adaptive pooling keeps this architecture compatible with 28×28 inputs;
    update class names and saved preprocessing metadata. Do not mix its test images into training.

    Next: [06 — Augmentation and regularization](06_augmentation_regularization.ipynb).
    References: [Conv2d](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html),
    [cross-entropy](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html),
    [FashionMNIST](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.FashionMNIST.html).
    ''')]
write_notebook("05_convolutional_neural_networks.ipynb", cells)


cells = [
    md('''
    # 06 — Augmentation, dropout, and weight decay

    **Prerequisites:** notebook 05's CNN and training/validation distinction.
    **Goals:** design a controlled experiment, apply augmentation only to training images,
    and compare dropout and weight decay using validation curves.
    **Time:** 25–40 minutes. All data are local bundled digits; no downloads.

    A transformation must preserve the correct label. A horizontal flip can change the meaning
    of a digit, so our default uses only a small rotation and translation. Even these can erase
    detail at 8×8, so we resize to 16×16 and inspect the results first.
    ''')
    , bootstrap()
    , code('''
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import torch
    from torch import nn
    from torchvision import transforms as T
    from course_utils import get_digits_loaders, fit, evaluate

    augmentation = T.RandomAffine(degrees=8, translate=(0.06, 0.06))
    plain_loaders = get_digits_loaders(batch_size=64, image_size=16, seed=42)
    xb, yb = next(iter(plain_loaders[0]))
    fig, axes = plt.subplots(1, 7, figsize=(11, 2))
    for i, ax in enumerate(axes):
        picture = xb[0] if i == 0 else augmentation(xb[0])
        ax.imshow(picture[0], cmap="gray", vmin=0, vmax=1)
        ax.set_title("Original" if i == 0 else "Augmented"); ax.axis("off")
    plt.show()
    ''')
    , md('''
    ## 1. Decide what changes and what stays fixed

    We use the same split seed, model initialization seed, batch size, learning rate, and epoch budget.
    Four runs separate augmentation from dropout plus weight decay, then combine them.
    Dropout and weight decay are grouped in one run for speed; the exercises separate them further.

    This is one controlled pilot, not strong statistical evidence. Random transforms consume RNG
    draws, so stochastic training trajectories still differ. Repeat with several seeds to measure stability.
    ''')
    , code('''
    class RegularizedCNN(nn.Module):
        def __init__(self, dropout=0.0):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 24, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(24, 48, 3, padding=1), nn.ReLU(),
                nn.AdaptiveAvgPool2d((2, 2)))
            self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(192, 10))
        def forward(self, x):
            return self.head(self.features(x))

    configs = {
        "baseline": dict(augment=False, dropout=0.0, decay=0.0),
        "augmentation": dict(augment=True, dropout=0.0, decay=0.0),
        "dropout_and_decay": dict(augment=False, dropout=0.25, decay=1e-3),
        "combined": dict(augment=True, dropout=0.25, decay=1e-3),
    }
    pd.DataFrame(configs).T
    ''')
    , md('''
    ## 2. Run the comparisons

    `get_digits_loaders` always uses the same disjoint examples for a given split seed. Its
    `train_transform` runs only on training examples; validation and test have deterministic resizing.
    Inspect the helper source in `course_utils.py` if you want to see that separation.

    Dropout randomly removes some activations during training and disables itself in evaluation.
    AdamW's weight decay gently shrinks weights at each optimizer step. Excessive regularization
    can cause underfitting. There is no promise that these settings outperform the baseline.
    ''')
    , code('''
    epochs = 2 if SMOKE_TEST else 12
    runs = {}
    for name, cfg in configs.items():
        seed_everything(42)
        loaders = get_digits_loaders(batch_size=64, image_size=16, seed=42,
                                    train_transform=augmentation if cfg["augment"] else None)
        candidate = RegularizedCNN(cfg["dropout"]).to(device)
        history = fit(candidate, loaders[0], loaders[1], epochs=epochs, lr=0.003,
                      weight_decay=cfg["decay"], device=device)
        runs[name] = {"model": candidate, "history": pd.DataFrame(history),
                      "val": evaluate(candidate, loaders[1], device=device)}
        print(name, "best-epoch validation accuracy:", f"{runs[name]['val']['accuracy']:.1%}")
    ''')
    , code('''
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for name, run in runs.items():
        h = run["history"]
        axes[0].plot(h.epoch, h.val_loss, label=name)
        axes[1].plot(h.epoch, h.val_acc, label=name)
    axes[0].set_title("Validation loss"); axes[1].set_title("Validation accuracy")
    for ax in axes:
        ax.set_xlabel("Epoch"); ax.legend(fontsize=8)
    plt.tight_layout(); plt.show()
    ''')
    , md('''
    ## 3. Look for a gap, not only a winning score

    Training metrics are measured while the model changes, with dropout and augmentation active.
    Validation metrics use a fixed model and clean images, so validation can occasionally look easier.
    A large training/validation gap can suggest overfitting but does not prove its cause.
    ''')
    , code('''
    fig, axes = plt.subplots(1, 2, figsize=(11, 3))
    for name in ["baseline", "combined"]:
        h = runs[name]["history"]
        axes[0].plot(h.epoch, h.train_loss, label=f"{name}: train")
        axes[0].plot(h.epoch, h.val_loss, "--", label=f"{name}: val")
        axes[1].plot(h.epoch, h.train_acc - h.val_acc, label=name)
    axes[0].set_title("Loss: solid train, dashed validation")
    axes[1].set_title("Train accuracy − validation accuracy")
    for ax in axes:
        ax.set_xlabel("Epoch"); ax.legend(fontsize=8)
    plt.tight_layout(); plt.show()
    ''')
    , code('''
    summary = pd.DataFrame([
        {"run": name, "val_loss": run["val"]["loss"], "val_accuracy": run["val"]["accuracy"]}
        for name, run in runs.items()
    ]).sort_values("val_loss")
    selected_name = summary.iloc[0]["run"]
    print("Choose by validation loss:", selected_name)
    summary
    ''')
    , md('''
    ## 4. Evaluate the selected model once

    We used validation to compare configurations. Now estimate generalization with the held-out
    test partition. If you keep adjusting settings after reading the test score, that score becomes
    part of the tuning process and stops being an independent final estimate.
    ''')
    , code('''
    selected_model = runs[selected_name]["model"]
    test_result = evaluate(selected_model, plain_loaders[2], device=device)
    print(f"Selected: {selected_name} | test accuracy: {test_result['accuracy']:.1%}")
    summary.to_csv(ARTIFACTS / "06_regularization_comparison.csv", index=False)
    ''')
    , code('''
    # Demonstrate train/eval behavior without changing the trained model.
    dropout_demo = nn.Dropout(0.5)
    ones = torch.ones(12)
    dropout_demo.train(); print("Training:", dropout_demo(ones))
    dropout_demo.eval(); print("Evaluation:", dropout_demo(ones))
    ''')
]
cells += exercises([
    ("Separate dropout from weight decay", "Train a dropout-only model, then modify the starter to train a decay-only model. Use validation loss for comparison.",
     '''RUN_EXTRA_EXPERIMENTS = False
     if RUN_EXTRA_EXPERIMENTS:
         seed_everything(42)
         loaders = get_digits_loaders(batch_size=64, image_size=16, seed=42)
         dropout_only = RegularizedCNN(dropout=0.25).to(device)
         extra_history = fit(dropout_only, loaders[0], loaders[1], epochs=epochs,
                             lr=0.003, weight_decay=0.0, device=device)
         print(pd.DataFrame(extra_history).tail())''',
     "For decay only, use `RegularizedCNN(dropout=0.0)` with `weight_decay=1e-3`. Reseed, recreate loaders, and keep the epoch budget fixed. This separates the two interventions."),
    ("Challenge a label-preserving assumption", "Show a stronger rotation next to the original. Decide whether a person could still assign the original label reliably.",
     '''strong_transform = T.RandomAffine(degrees=50, translate=(0.25, 0.25))
     fig, axes = plt.subplots(1, 2, figsize=(5, 2))
     axes[0].imshow(xb[0, 0], cmap="gray"); axes[0].set_title(f"Original: {yb[0].item()}")
     axes[1].imshow(strong_transform(xb[0])[0], cmap="gray"); axes[1].set_title("Strong transform")
     for ax in axes: ax.axis("off")
     plt.show()
     # TODO: Repeat with several digits before judging the transform.''',
     "Large rotations or translations can change digit meaning or remove strokes. Inspect multiple images from every class. Stronger augmentation is not automatically better."),
    ("Plan a repeatable experiment", "Create a table of three random seeds. Enable the optional loop to estimate variability using validation accuracy.",
     '''seed_results = []
     RUN_SEED_STUDY = False
     if RUN_SEED_STUDY:
         for model_seed in [7, 42, 123]:
             seed_everything(model_seed)
             loaders = get_digits_loaders(batch_size=64, image_size=16, seed=42)
             candidate = RegularizedCNN().to(device)
             fit(candidate, loaders[0], loaders[1], epochs=epochs, lr=0.003, device=device)
             seed_results.append({"seed": model_seed,
                                  "val_accuracy": evaluate(candidate, loaders[1], device=device)["accuracy"]})
     print(pd.DataFrame(seed_results))''',
     "Keep the split seed fixed to study model initialization variability. Report the individual scores, mean, and sample standard deviation; three runs are a small pilot. A different study can vary split seeds as well."),
])
cells += [md('''
    **Checkpoint:** identify leakage from validation augmentation, explain dropout in `eval()` mode,
    and describe how to compare two regularizers fairly.

    Next: [07 — Transfer learning with ResNet](07_transfer_learning_resnet.ipynb).
    References: [transforms](https://docs.pytorch.org/vision/stable/transforms.html),
    [Dropout](https://docs.pytorch.org/docs/stable/generated/torch.nn.Dropout.html),
    [AdamW](https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html).
    ''')]
write_notebook("06_augmentation_regularization.ipynb", cells)


cells = [
    md('''
    # 07 — Transfer learning: a frozen ResNet, then careful fine-tuning

    **Prerequisites:** notebooks 05–06; understand a classifier head, validation, and learning rate.
    **Goals:** adapt genuine ImageNet-pretrained ResNet-18 features to CIFAR-10, freeze and unfreeze
    layers correctly, handle BatchNorm state, and choose checkpoints using validation only.
    **Time:** 30–60 minutes including downloads and experiments. Normal mode downloads CIFAR-10
    (about 170 MB archive) and pretrained weights (about 45 MB). Allow roughly 1 GB free disk space.

    The quick default trains on **2,000** class-balanced images and uses **500** each for validation
    and test. Batch 32 and two short stages fit a typical 8 GB GPU. This is a learning experiment,
    not a benchmark result. Set `COURSE_SMOKE_TEST=1` before launching the kernel for a tiny offline
    **FakeData + randomly initialized ResNet** wiring check; that mode does **not** demonstrate learned
    transfer or meaningful accuracy. Normal download failures raise errors; no synthetic fallback occurs.
    ''')
    , bootstrap()
    , code('''
    import copy
    import json
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Subset
    from torchvision import datasets, models, transforms as T
    from course_utils import DATA_DIR, evaluate

    BATCH_SIZE = 16 if SMOKE_TEST else 32
    weights = models.ResNet18_Weights.DEFAULT
    evaluation_transform = weights.transforms()
    training_transform = T.Compose([
        T.Resize(256), T.RandomCrop(224), T.RandomHorizontalFlip(),
        T.ToTensor(), T.Normalize(mean=evaluation_transform.mean, std=evaluation_transform.std),
    ])
    print("Evaluation preprocessing:", evaluation_transform)
    print("SMOKE: random weights and synthetic data" if SMOKE_TEST else "REAL: ImageNet weights + CIFAR-10")
    ''')
    , md('''
    ## 1. Match preprocessing and keep partitions disjoint

    The pretrained weights expect three-channel images, a 224×224 crop, and their ImageNet
    normalization. We obtain validation preprocessing directly from the weights object and use the
    same normalization during training. Resizing CIFAR-10's 32×32 images does not add new detail.

    Validation comes from the official training partition; final test images come from the official
    test partition. We explicitly select equal counts per class for this quick experiment.
    ''')
    , code('''
    def balanced_indices(targets, per_class, seed=42, exclude=()):
        rng = np.random.default_rng(seed)
        targets = np.asarray(targets)
        allowed = np.ones(len(targets), dtype=bool)
        allowed[np.asarray(list(exclude), dtype=int)] = False
        result = []
        for label in np.unique(targets):
            pool = np.flatnonzero((targets == label) & allowed)
            if len(pool) < per_class:
                raise ValueError(f"Insufficient examples of class {label}")
            result.extend(rng.permutation(pool)[:per_class].tolist())
        rng.shuffle(result)
        return result

    if SMOKE_TEST:
        # Explicit offline verification only: no learned pretrained features are loaded.
        train_data = datasets.FakeData(32, (3, 32, 32), 10, transform=training_transform, random_offset=0)
        val_data = datasets.FakeData(16, (3, 32, 32), 10, transform=evaluation_transform, random_offset=1000)
        test_data = datasets.FakeData(16, (3, 32, 32), 10, transform=evaluation_transform, random_offset=2000)
        class_names = [f"synthetic_{i}" for i in range(10)]
        split_indices = {"mode": "synthetic wiring verification"}
    else:
        base_train = datasets.CIFAR10(DATA_DIR, train=True, download=True, transform=training_transform)
        base_val = datasets.CIFAR10(DATA_DIR, train=True, download=False, transform=evaluation_transform)
        base_test = datasets.CIFAR10(DATA_DIR, train=False, download=True, transform=evaluation_transform)
        train_ids = balanced_indices(base_train.targets, 200, seed=42)
        val_ids = balanced_indices(base_train.targets, 50, seed=43, exclude=train_ids)
        test_ids = balanced_indices(base_test.targets, 50, seed=44)
        assert not set(train_ids) & set(val_ids)
        train_data, val_data, test_data = (Subset(base_train, train_ids),
                                          Subset(base_val, val_ids), Subset(base_test, test_ids))
        class_names = base_train.classes
        split_indices = {"train": train_ids, "val": val_ids, "test": test_ids}

    generator = torch.Generator().manual_seed(42)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, generator=generator)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print("Train/val/test:", len(train_data), len(val_data), len(test_data))
    ''')
    , code('''
    xb, yb = next(iter(train_loader))
    mean = torch.tensor(evaluation_transform.mean).view(3, 1, 1)
    std = torch.tensor(evaluation_transform.std).view(3, 1, 1)
    fig, axes = plt.subplots(1, 6, figsize=(12, 2))
    for ax, image, label in zip(axes, xb[:6], yb[:6]):
        ax.imshow((image * std + mean).clamp(0, 1).permute(1, 2, 0))
        ax.set_title(class_names[label.item()], fontsize=9); ax.axis("off")
    plt.show()
    ''')
    , md('''
    ## 2. Replace the classifier and freeze the backbone

    The original ImageNet head predicts 1,000 categories. Our new randomly initialized head predicts
    ten categories. First train only this head so its early gradients cannot disturb useful features.

    **Subtle point:** `requires_grad=False` freezes parameters, but does not stop BatchNorm running
    averages from changing in `train()` mode. During head training, keep the entire model in `eval()`
    and put just the head in `train()`. During fine-tuning, freeze BatchNorm statistics explicitly.
    This is a useful small-dataset strategy; with abundant data, updating statistics can be reasonable.
    ''')
    , code('''
    seed_everything(42)
    model = models.resnet18(weights=None if SMOKE_TEST else weights)
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model = model.to(device)

    def show_trainable(model):
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable: {trainable:,} / {total:,}")

    show_trainable(model)
    ''')
    , code('''
    def set_stage_mode(model, stage):
        if stage == "head":
            model.eval()
            model.fc.train()
        else:
            model.train()
            for module in model.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.eval()  # Keep the pretrained running means/variances fixed.

    def train_stage(model, optimizer, epochs, stage):
        history, best_loss, best_state = [], float("inf"), None
        for epoch in range(1, epochs + 1):
            set_stage_mode(model, stage)
            total_loss, correct, count = 0.0, 0, 0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(images)
                loss = nn.functional.cross_entropy(logits, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(labels)
                correct += (logits.argmax(1) == labels).sum().item()
                count += len(labels)
            validation = evaluate(model, val_loader, device=device)
            row = dict(stage=stage, epoch=epoch, train_loss=total_loss/count,
                       train_acc=correct/count, val_loss=validation["loss"], val_acc=validation["accuracy"])
            history.append(row)
            if row["val_loss"] < best_loss:
                best_loss = row["val_loss"]
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(row)
        model.load_state_dict(best_state)
        return history

    head_optimizer = torch.optim.AdamW(model.fc.parameters(), lr=1e-3, weight_decay=1e-4)
    head_history = train_stage(model, head_optimizer, 1 if SMOKE_TEST else 3, "head")
    head_validation = evaluate(model, val_loader, device=device)
    head_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    ''')
    , md('''
    ## 3. Fine-tune the final residual block gently

    Unfreeze `layer4`, then create a **new optimizer** containing those newly trainable parameters.
    Its learning rate is ten times smaller than the head's: pretrained features usually need smaller
    updates. Earlier blocks remain frozen. BatchNorm affine parameters remain frozen too, while
    `set_stage_mode` also freezes their running statistics.

    A lower training loss after unfreezing does not guarantee a better validation result. We retain
    the head-only checkpoint in case fine-tuning makes validation worse.
    ''')
    , code('''
    for parameter in model.layer4.parameters():
        parameter.requires_grad = True
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            for parameter in module.parameters():
                parameter.requires_grad = False
    fine_optimizer = torch.optim.AdamW([
        {"params": [p for p in model.layer4.parameters() if p.requires_grad], "lr": 1e-4},
        {"params": model.fc.parameters(), "lr": 1e-3},
    ], weight_decay=1e-4)
    show_trainable(model)
    bn_before = model.bn1.running_mean.detach().clone()
    fine_history = train_stage(model, fine_optimizer, 1 if SMOKE_TEST else 2, "fine_tune")
    assert torch.equal(bn_before, model.bn1.running_mean), "Frozen BatchNorm statistics changed"
    fine_validation = evaluate(model, val_loader, device=device)
    ''')
    , code('''
    history_df = pd.DataFrame(head_history + fine_history)
    history_df["global_epoch"] = np.arange(1, len(history_df) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3))
    axes[0].plot(history_df.global_epoch, history_df.train_loss, label="train")
    axes[0].plot(history_df.global_epoch, history_df.val_loss, label="validation")
    axes[1].plot(history_df.global_epoch, history_df.val_acc, label="validation accuracy")
    for ax in axes:
        ax.axvline(len(head_history) + 0.5, color="gray", linestyle="--")
        ax.set_xlabel("Epoch (dashed line: start fine-tuning)"); ax.legend()
    plt.tight_layout(); plt.show()
    print("Head validation:", head_validation["accuracy"], "Fine-tuned:", fine_validation["accuracy"])
    ''')
    , md('''
    ## 4. Select on validation, evaluate once on test, and save

    The curve contains all epochs; the current model contains the best fine-tuning epoch. Compare
    that checkpoint with the best head-only checkpoint using validation loss, then evaluate the winner
    on the test set. Synthetic smoke-test scores have no learning interpretation.
    ''')
    , code('''
    selected_stage = "fine_tune"
    if head_validation["loss"] <= fine_validation["loss"]:
        model.load_state_dict(head_state)
        selected_stage = "head"
    test_result = evaluate(model, test_loader, device=device)
    print("Selected stage:", selected_stage, "| test accuracy:", f"{test_result['accuracy']:.1%}")
    checkpoint = {
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "architecture": "resnet18", "classes": class_names, "selected_stage": selected_stage,
        "pretrained_source": None if SMOKE_TEST else str(weights),
        "smoke_test": SMOKE_TEST, "image_size": 224,
        "mean": evaluation_transform.mean, "std": evaluation_transform.std,
    }
    suffix = "smoke" if SMOKE_TEST else "cifar10"
    torch.save(checkpoint, ARTIFACTS / f"07_resnet18_{suffix}.pt")
    (ARTIFACTS / f"07_splits_{suffix}.json").write_text(json.dumps(split_indices), encoding="utf-8")
    ''')
]
cells += exercises([
    ("Audit frozen parameters", "List trainable parameter names. Confirm earlier blocks and BatchNorm parameters are frozen.",
     '''trainable_names = [name for name, p in model.named_parameters() if p.requires_grad]
     print("\\n".join(trainable_names))
     # TODO: Assert that every name starts with layer4. or fc.''',
     "```python\nassert all(name.startswith(('layer4.', 'fc.')) for name in trainable_names)\nassert all(not p.requires_grad for m in model.modules()\n           if isinstance(m, nn.modules.batchnorm._BatchNorm) for p in m.parameters())\n```"),
    ("Design a fair transfer comparison", "Create a randomly initialized comparison model. Write down what must stay fixed before optionally training it.",
     '''seed_everything(42)
     scratch_model = models.resnet18(weights=None, num_classes=10)
     print("Scratch parameters:", sum(p.numel() for p in scratch_model.parameters()))
     # TODO: Plan a comparison with identical data splits and compute budget.
     del scratch_model''',
     "Keep splits, preprocessing, seed policy, batch size, selection metric, and training budget fixed. A short budget favors the pretrained model; report that limitation. A fair best-performance comparison requires separate validation tuning and often more scratch-training compute."),
    ("Read optimizer groups", "Inspect both learning rates and the number of parameters they control. Explain why changing requires_grad alone would not update an already-created head-only optimizer.",
     '''for i, group in enumerate(fine_optimizer.param_groups):
         print("Group", i, "lr", group["lr"], "parameters", sum(p.numel() for p in group["params"]))
     # TODO: Predict what changes if the backbone LR is raised to 0.01.''',
     "An optimizer holds the parameter objects supplied at creation. Newly unfrozen objects must be added or included in a new optimizer. A very large backbone LR can rapidly overwrite pretrained features; validation must test the effect."),
])
cells += [md('''
    **Checkpoint:** explain why frozen BatchNorm needs explicit mode handling, when to recreate an
    optimizer, and why normalization must match the pretrained weights.

    Next: [08 — Your own image dataset](08_your_own_image_dataset.ipynb).
    References: [ResNet-18 weights and preprocessing](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html),
    [official transfer tutorial](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html),
    [CIFAR-10](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.CIFAR10.html).
    ''')]
write_notebook("07_transfer_learning_resnet.ipynb", cells)


cells = [
    md('''
    # 08 — Build a reliable classifier for your own image folders

    **Prerequisites:** notebooks 05–07; know training, validation, test, and class indices.
    **Goals:** validate an ImageFolder dataset, check class mappings and duplicate leakage, train
    a model, and save the exact label/preprocessing contract required for prediction.
    **Time:** 25–40 minutes. The default generates a tiny demonstration dataset from bundled digits,
    with no network download. You can then point the same pipeline at your own prepared folders.

    This notebook **does not move, split, delete, repair, or overwrite your own images**. It creates
    demo files only inside `data/demo_imagefolder`. User images are opened read-only. For a first real
    project use two or three clearly defined classes, then inspect examples and mistakes carefully.
    ''')
    , bootstrap()
    , code('''
    import hashlib
    import json
    import warnings
    from collections import Counter
    from pathlib import Path
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from PIL import Image
    import torch
    from torch.utils.data import DataLoader, Subset
    from torchvision import datasets, transforms as T
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    from course_utils import DATA_DIR, TinyCNN, fit, evaluate

    USE_MY_DATA = False
    MY_DATA_ROOT = ROOT / "my_image_dataset"  # Change this only when USE_MY_DATA = True.
    DEMO_ROOT = DATA_DIR / "demo_imagefolder"
    IMAGE_SIZE = 32
    ''')
    , md('''
    ## 1. Prepare separate split folders

    `ImageFolder` assigns alphabetical class names to integer indices. All three partitions must
    contain exactly the same class folders and at least one image per class.

    ```text
    my_image_dataset/
      train/cat/image001.jpg     train/dog/image101.jpg
      val/cat/image201.jpg       val/dog/image301.jpg
      test/cat/image401.jpg      test/dog/image501.jpg
    ```

    Split by the **independent source** before copying files: photos of the same person, object,
    video, or burst should stay in one partition. Randomly splitting nearly identical frames can
    greatly inflate test accuracy. The demo starts from the fixed disjoint digits partitions.
    ''')
    , code('''
    if not USE_MY_DATA:
        # Guard the sole write destination, and never adopt a pre-existing unrelated folder.
        demo_resolved = DEMO_ROOT.resolve()
        assert demo_resolved.parent == DATA_DIR.resolve() and demo_resolved.name == "demo_imagefolder"
        marker = DEMO_ROOT / "generated_by_course.json"
        if DEMO_ROOT.exists() and not marker.exists():
            if any(DEMO_ROOT.iterdir()):
                raise RuntimeError(f"Refusing to use an unmarked nonempty demo folder: {DEMO_ROOT}")
        DEMO_ROOT.mkdir(parents=True, exist_ok=True)
        if not marker.exists():
            marker.write_text(json.dumps({"source": "sklearn digits", "seed": 42,
                                          "purpose": "generated course demo", "version": 1}), encoding="utf-8")
        # Always export the full deterministic demo, even during smoke tests. Original row IDs
        # keep filenames stable when the notebook is rerun with a different compute budget.
        digits = load_digits()
        all_ids = np.arange(len(digits.target))
        train_ids, held_ids = train_test_split(all_ids, test_size=0.30,
                                               stratify=digits.target, random_state=42)
        val_ids, test_ids = train_test_split(held_ids, test_size=0.50,
                                             stratify=digits.target[held_ids], random_state=42)
        for split_name, ids in {"train": train_ids, "val": val_ids, "test": test_ids}.items():
            for index in ids:
                pixels, label = digits.images[index] / 16.0, digits.target[index]
                destination = DEMO_ROOT / split_name / str(int(label)) / f"digit_{index:04d}.png"
                if not destination.resolve().is_relative_to(demo_resolved):
                    raise RuntimeError(f"Demo destination escapes its directory: {destination}")
                destination.parent.mkdir(parents=True, exist_ok=True)
                # Exclusive creation makes reruns safe: existing files are never overwritten.
                if not destination.exists():
                    picture = Image.fromarray(np.rint(pixels * 255).astype(np.uint8))
                    with destination.open("xb") as stream:
                        picture.save(stream, format="PNG")
        dataset_root = DEMO_ROOT
    else:
        dataset_root = MY_DATA_ROOT.expanduser().resolve()
        if not dataset_root.is_dir():
            raise FileNotFoundError(f"Prepare your train/val/test folders first: {dataset_root}")
    print("Reading images from:", dataset_root)
    ''')
    , md('''
    ## 2. Validate before training

    We verify images and decode them in a second open, preserving Pillow's decompression-bomb checks.
    Failures are reported instead of silently dropping files, because silent drops can alter class
    balance or remove a class entirely. This scan can take time on a large dataset.

    The demo uses PNG; your images can use the formats accepted by ImageFolder. Fix or remove corrupt
    images yourself after reviewing the report, then rerun. This notebook makes no such changes.
    ''')
    , code('''
    from torchvision.datasets.folder import IMG_EXTENSIONS

    def inspect_image(path):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(path) as image:
                    image.verify()
                with Image.open(path) as image:
                    image.load()
                    if image.width < 2 or image.height < 2:
                        raise ValueError("Image dimensions are too small")
            return None
        except (OSError, ValueError, SyntaxError, Image.DecompressionBombWarning,
                Image.DecompressionBombError) as error:
            return f"{type(error).__name__}: {error}"

    bad_files, image_files = [], {}
    for split_name in ["train", "val", "test"]:
        split_path = dataset_root / split_name
        if not split_path.is_dir():
            raise FileNotFoundError(f"Missing split directory: {split_path}")
        image_files[split_name] = sorted(p for p in split_path.rglob("*")
                                        if p.is_file() and p.suffix.lower() in IMG_EXTENSIONS)
        if not image_files[split_name]:
            raise ValueError(f"No supported image files in {split_path}")
        for path in image_files[split_name]:
            problem = inspect_image(path)
            if problem is not None:
                bad_files.append({"path": str(path), "problem": problem})
    if bad_files:
        display(pd.DataFrame(bad_files))
        raise ValueError("Image validation failed. Review the files; nothing was changed.")
    print("Validated file counts:", {s: len(paths) for s, paths in image_files.items()})
    ''')
    , md('''
    ## 3. Verify class maps and exact duplicate leakage

    A label index must mean the same class everywhere. Hashing detects byte-identical files across
    partitions; it cannot detect resized, re-encoded, cropped, or visually near-identical copies.
    Dataset collection records and group-aware splitting are still essential.

    This lesson resizes every image to 32×32 RGB to keep training light. That can distort aspect ratio
    and lose detail; choose a deliberate crop or letterbox policy for a real project.
    ''')
    , code('''
    transform = T.Compose([T.Resize((IMAGE_SIZE, IMAGE_SIZE)), T.ToTensor()])
    image_sets = {split: datasets.ImageFolder(dataset_root / split, transform=transform)
                  for split in ["train", "val", "test"]}
    class_to_idx = image_sets["train"].class_to_idx
    assert len(class_to_idx) >= 2, "Classification needs at least two classes in this lesson"
    for split, dataset in image_sets.items():
        assert dataset.class_to_idx == class_to_idx, f"Class mapping mismatch in {split}"
    class_names = image_sets["train"].classes
    class_counts = pd.DataFrame({s: Counter(ds.targets) for s, ds in image_sets.items()}).fillna(0).astype(int)
    class_counts.index = [class_names[i] for i in class_counts.index]
    print("Class map:", class_to_idx)
    display(class_counts.sort_index())

    def file_digest(path):
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    seen_hashes, duplicate_pairs = {}, []
    for split, paths in image_files.items():
        for path in paths:
            digest = file_digest(path)
            if digest in seen_hashes and seen_hashes[digest][0] != split:
                duplicate_pairs.append((str(seen_hashes[digest][1]), str(path)))
            seen_hashes.setdefault(digest, (split, path))
    if duplicate_pairs:
        display(pd.DataFrame(duplicate_pairs, columns=["first_path", "duplicate_path"]))
        raise ValueError("Exact cross-split duplicates found. Review your split before training.")
    print("No byte-identical cross-split duplicates found.")
    ''')
    , code('''
    generator = torch.Generator().manual_seed(42)
    training_sets = image_sets.copy()
    if SMOKE_TEST:
        # Limit compute, while retaining examples from every class where available.
        for split, dataset in image_sets.items():
            per_class = 8 if split == "train" else 3
            labels = np.asarray(dataset.targets)
            ids = np.concatenate([np.flatnonzero(labels == label)[:per_class]
                                  for label in range(len(class_names))]).tolist()
            training_sets[split] = Subset(dataset, ids)
    loaders = {split: DataLoader(dataset, batch_size=32, shuffle=(split == "train"),
                                  num_workers=0, generator=generator if split == "train" else None)
               for split, dataset in training_sets.items()}
    xb, yb = next(iter(loaders["train"]))
    fig, axes = plt.subplots(1, min(8, len(xb)), figsize=(12, 2), squeeze=False)
    for ax, image, label in zip(axes.flat, xb[:8], yb[:8]):
        ax.imshow(image.permute(1, 2, 0)); ax.set_title(class_names[label.item()]); ax.axis("off")
    plt.show()
    print("Batch:", tuple(xb.shape))
    ''')
    , md('''
    ## 4. Establish a simple baseline first

    The demo digits can be learned with a tiny CNN. For real natural images, this scratch model may
    be weak; reuse notebook 07's pretrained backbone and its **matching preprocessing** as your next
    experiment. Keep your current splits fixed when comparing models.

    Check whether classes are imbalanced. Accuracy can hide failure on a rare class; notebook 09
    develops per-class metrics and error analysis.
    ''')
    , code('''
    seed_everything(42)
    model = TinyCNN(num_classes=len(class_names), in_channels=3).to(device)
    history = fit(model, loaders["train"], loaders["val"], epochs=2 if SMOKE_TEST else 10,
                  lr=0.003, weight_decay=1e-4, device=device)
    history_df = pd.DataFrame(history).set_index("epoch")
    history_df[["train_loss", "val_loss"]].plot(figsize=(7, 3), title="Your dataset: training and validation")
    plt.show()
    test_result = evaluate(model, loaders["test"], device=device)
    print(f"Selected model test accuracy: {test_result['accuracy']:.1%}")
    ''')
    , md('''
    ## 5. Save the prediction contract and run one-file inference

    Checkpoint metadata must preserve class order, channel conversion, resize dimensions, and
    normalization. An accurate model can produce meaningless predictions if its labels are decoded
    with a different alphabetical order. The prediction below reuses the exact evaluation transform.
    Only load checkpoints you trust.
    ''')
    , code('''
    source_name = "user" if USE_MY_DATA else "demo"
    checkpoint_path = ARTIFACTS / f"08_imagefolder_{source_name}.pt"
    checkpoint = {"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                  "architecture": "TinyCNN", "in_channels": 3,
                  "classes": class_names, "class_to_idx": class_to_idx,
                  "image_size": IMAGE_SIZE, "input_range": [0.0, 1.0], "color_mode": "RGB"}
    torch.save(checkpoint, checkpoint_path)

    def predict_file(path):
        problem = inspect_image(Path(path))
        if problem is not None:
            raise ValueError(problem)
        with Image.open(path) as image:
            tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)
        model.eval()
        with torch.inference_mode():
            probabilities = model(tensor).softmax(dim=1)[0].cpu()
        return pd.Series(probabilities.numpy(), index=class_names).sort_values(ascending=False)

    example_path, expected_index = image_sets["test"].samples[0]
    print("Expected:", class_names[expected_index], "| file:", example_path)
    display(predict_file(example_path).head())
    print("Saved:", checkpoint_path)
    ''')
]
cells += exercises([
    ("Inspect class imbalance", "Compute each training class's fraction and the majority-class baseline. This baseline always predicts the most common training class.",
     '''counts = np.bincount(image_sets["train"].targets, minlength=len(class_names))
     print(pd.Series(counts / counts.sum(), index=class_names))
     majority_class = int(counts.argmax())
     print("Majority class:", class_names[majority_class])
     # TODO: Measure this fixed predictor on the validation labels.''',
     "```python\nval_labels = np.asarray(image_sets['val'].targets)\nprint('Majority baseline validation accuracy:', (val_labels == majority_class).mean())\n```\nChoose the class from training data, then measure on validation. Do not select it using test labels."),
    ("Round-trip the checkpoint", "Load the saved metadata into a new model. Check that predictions agree for the same image tensor.",
     '''saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
     restored = TinyCNN(num_classes=len(saved["classes"]), in_channels=saved["in_channels"])
     restored.load_state_dict(saved["state_dict"])
     restored.eval()
     print("Restored classes:", saved["classes"])
     # TODO: Compare logits from restored and model on xb[:2].''',
     "```python\nmodel.eval()\nwith torch.inference_mode():\n    original_logits = model(xb[:2].to(device)).cpu()\n    restored_logits = restored(xb[:2])\nassert torch.allclose(original_logits, restored_logits, atol=1e-4, rtol=1e-4)\n```\nSmall CPU/GPU floating-point differences are expected."),
    ("Plan a real dataset", "Fill in this collection plan before switching USE_MY_DATA to True. Decide the independent group and a meaningful failure case.",
     '''dataset_plan = {
         "classes": ["class_a", "class_b"],
         "independent_group": "one photographed physical object",
         "split_rule": "all photos of the same object stay together",
         "deployment_conditions": "describe lighting, camera, and background",
         "important_failure": "describe a costly or confusing mistake",
     }
     print(json.dumps(dataset_plan, indent=2))
     # TODO: Replace these placeholders with your own project decisions.''',
     "For leaf species, group by individual plant, collect multiple lighting conditions, and reserve different plants for test. Check that backgrounds or collection folders do not reveal labels. More copies of the same photograph do not add independent evidence."),
])
cells += [md('''
    **Checkpoint:** prepare train/val/test folders with consistent labels, detect corrupt files,
    explain why hashes miss near-duplicates, and restore a saved model with its class order.

    Next: [09 — Evaluation, imbalance, and calibration](09_evaluation_imbalance_calibration.ipynb).
    References: [ImageFolder](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.ImageFolder.html),
    [Pillow image validation](https://pillow.readthedocs.io/en/stable/reference/Image.html),
    [saving and loading models](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html).
    ''')]
write_notebook("08_your_own_image_dataset.ipynb", cells)
