"""Build lessons 09–12. Run from the project root with the course environment."""
from notebook_builder import md, code, bootstrap, write_notebook


def lesson09():
    return [
        md('''
        # 09 · Evaluate honestly: imbalance, errors, and confidence

        **Prerequisites:** train a CNN and distinguish train, validation, and test data.
        **Time:** about 35–55 minutes including experiments. Everything uses offline digits.

        By the end you will compare macro and aggregate metrics, create an imbalanced
        training set without changing evaluation data, and fit temperature scaling on
        a separate calibration split. A confidence of 0.9 is a claim about reliability,
        not proof that a prediction is correct.

        Run each cell in order. Keep the test set closed until the final evaluation.
        ''') ,
        bootstrap(),
        code('''
        import copy
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import torch
        from torch import nn
        from torch.utils.data import TensorDataset, DataLoader
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                                     f1_score, classification_report, ConfusionMatrixDisplay)
        from course_utils import digit_splits, TinyCNN, evaluate
        epochs = 1 if SMOKE_TEST else 8
        ''') ,
        md('''
        ## 1. Give every split one job

        Training updates network weights. Validation selects a model checkpoint and
        weighting strategy. Calibration fits one temperature after those decisions.
        Test estimates final performance once. We divide the original validation split
        into model-validation and calibration subsets with stratification.

        This deliberately small calibration set makes uncertainty visible. In a real
        project, plan split sizes before collecting results, and split by patient,
        subject, product, or recording session when images share a source.
        ''') ,
        code('''
        splits = digit_splits(seed=42)
        x_train, y_train = splits['train']
        x_hold, y_hold = splits['val']
        x_val, x_cal, y_val, y_cal = train_test_split(
            x_hold, y_hold, test_size=0.5, stratify=y_hold, random_state=42)
        x_test, y_test = splits['test']
        if SMOKE_TEST and len(y_train) > 128:
            x_train, _, y_train, _ = train_test_split(
                x_train, y_train, train_size=128, stratify=y_train, random_state=42)
            x_test, y_test = x_test[:64], y_test[:64]

        def make_loader(x, y, shuffle=False):
            ds = TensorDataset(torch.from_numpy(x).unsqueeze(1).float(),
                               torch.from_numpy(y).long())
            return DataLoader(ds, batch_size=64, shuffle=shuffle, num_workers=0,
                              generator=torch.Generator().manual_seed(42))

        val_loader = make_loader(x_val, y_val)
        cal_loader = make_loader(x_cal, y_cal)
        test_loader = make_loader(x_test, y_test)
        print({'train': len(y_train), 'validation': len(y_val),
               'calibration': len(y_cal), 'test': len(y_test)})
        ''') ,
        md('''
        ## 2. Create a controlled training imbalance

        Keep all examples of digits 0–4 and only one quarter of digits 5–9 in training.
        Evaluation retains its natural class distribution. Inverse-frequency weights
        are computed **only from retained training labels**. Weighted cross-entropy
        increases the contribution of rare-class examples; it does not create new data.
        ''') ,
        code('''
        rng = np.random.default_rng(42)
        kept = []
        for cls in range(10):
            indices = np.flatnonzero(y_train == cls)
            n = len(indices) if cls < 5 else max(2, len(indices) // 4)
            kept.extend(rng.choice(indices, size=n, replace=False).tolist())
        kept = np.array(sorted(kept))
        x_imb, y_imb = x_train[kept], y_train[kept]
        counts = np.bincount(y_imb, minlength=10)
        weights = torch.tensor(len(y_imb) / (10 * counts), dtype=torch.float32)
        display(pd.DataFrame({'digit': range(10), 'training_count': counts,
                              'loss_weight': weights.numpy()}))
        plt.bar(range(10), counts)
        plt.xticks(range(10)); plt.xlabel('Digit'); plt.ylabel('Training images')
        plt.show()
        ''') ,
        code('''
        def train_candidate(weighted):
            # Reset initialization and batch order for a controlled comparison.
            seed_everything(42)
            model = TinyCNN().to(device)
            loader = make_loader(x_imb, y_imb, shuffle=True)
            criterion = nn.CrossEntropyLoss(weight=weights.to(device) if weighted else None)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
            best_score, best_state, history = -1.0, None, []
            for epoch in range(epochs):
                model.train()
                for images, labels in loader:
                    images, labels = images.to(device), labels.to(device)
                    optimizer.zero_grad(set_to_none=True)
                    loss = criterion(model(images), labels)
                    loss.backward()
                    optimizer.step()
                result = evaluate(model, val_loader, device=device)
                score = f1_score(result['targets'], result['logits'].argmax(1),
                                 labels=list(range(10)), average='macro', zero_division=0)
                history.append({'epoch': epoch + 1, 'val_macro_f1': score})
                if score > best_score:
                    best_score, best_state = score, copy.deepcopy(model.state_dict())
            model.load_state_dict(best_state)
            return model, history

        plain_model, plain_history = train_candidate(False)
        weighted_model, weighted_history = train_candidate(True)
        print('Training complete; only validation labels selected checkpoints.')
        ''') ,
        md('''
        ## 3. Read metrics in context

        Accuracy gives every example equal weight. Balanced accuracy is mean recall
        across classes. Macro F1 computes each class's F1 and averages them equally;
        weighted F1 weights class F1 by support. A strong common class can conceal
        poor rare-class performance in aggregate scores. F1 and balanced accuracy
        answer different questions; neither replaces a per-class review.
        ''') ,
        code('''
        def metric_row(result):
            truth = result['targets'].numpy()
            pred = result['logits'].argmax(1).numpy()
            return {'accuracy': accuracy_score(truth, pred),
                    'balanced_accuracy': balanced_accuracy_score(truth, pred),
                    'macro_f1': f1_score(truth, pred, labels=list(range(10)),
                                         average='macro', zero_division=0),
                    'weighted_f1': f1_score(truth, pred, average='weighted', zero_division=0)}

        candidates = {'plain': plain_model, 'weighted': weighted_model}
        val_results = {name: evaluate(model, val_loader, device=device)
                       for name, model in candidates.items()}
        comparison = pd.DataFrame({name: metric_row(result)
                                   for name, result in val_results.items()}).T
        display(comparison)
        chosen_name = comparison['macro_f1'].idxmax()
        chosen_model = candidates[chosen_name]
        chosen_val = val_results[chosen_name]
        print('Predeclared selection metric: validation macro F1. Selected:', chosen_name)
        ''') ,
        code('''
        truth = chosen_val['targets'].numpy()
        pred = chosen_val['logits'].argmax(1).numpy()
        print(classification_report(truth, pred, labels=list(range(10)), zero_division=0))
        ConfusionMatrixDisplay.from_predictions(truth, pred, labels=list(range(10)),
                                                normalize='true', cmap='Blues')
        plt.title('Validation: row-normalized confusion matrix')
        plt.show()
        ''') ,
        code('''
        wrong = np.flatnonzero(pred != truth)
        fig, axes = plt.subplots(1, 6, figsize=(12, 2.5))
        for ax in axes:
            ax.axis('off')
        for ax, idx in zip(axes, wrong[:6]):
            ax.imshow(x_val[idx], cmap='gray')
            ax.set_title(f'true {truth[idx]}, predicted {pred[idx]}')
        fig.suptitle('Validation errors: investigate patterns before changing a model')
        plt.tight_layout(); plt.show()
        print('Validation errors:', len(wrong))
        ''') ,
        md('''
        ## 4. Calibrate the frozen model

        Softmax converts logits to probabilities, but training does not guarantee
        calibrated confidence. Temperature scaling uses `softmax(logits / T)` with
        positive `T`. It changes confidence while preserving the predicted class.

        Freeze the network. Fit `T` by minimizing cross-entropy on calibration data
        only. We constrain `T` to [0.05, 20] for numerical stability in this tiny
        experiment. A boundary optimum is a reason to investigate sample size and
        model quality, not to claim successful calibration.
        ''') ,
        code('''
        import torch.nn.functional as F
        cal_result = evaluate(chosen_model, cal_loader, device=device)
        # evaluate uses inference_mode; clone outside it before autograd fits T.
        cal_logits = cal_result['logits'].detach().clone().float()
        cal_targets = cal_result['targets'].clone()
        log_temperature = nn.Parameter(torch.zeros(()))
        temperature_optimizer = torch.optim.LBFGS([log_temperature], lr=0.2, max_iter=40)

        def temperature_value():
            return log_temperature.exp().clamp(0.05, 20.0)

        def closure():
            temperature_optimizer.zero_grad()
            loss = F.cross_entropy(cal_logits / temperature_value(), cal_targets)
            loss.backward()
            return loss

        temperature_optimizer.step(closure)
        temperature = temperature_value().detach().item()
        print(f'Temperature: {temperature:.3f}')
        print('Calibration NLL before:', F.cross_entropy(cal_logits, cal_targets).item())
        print('Calibration NLL after: ', F.cross_entropy(cal_logits / temperature, cal_targets).item())
        ''') ,
        md('''
        ## 5. Understand a reliability diagram

        We group examples by their **top predicted probability**. For each bin, compare
        mean confidence with the fraction correctly classified. This is top-label
        calibration, not a complete classwise calibration test.

        Expected calibration error (ECE) is a bin-count-weighted average absolute gap.
        It depends on bin boundaries, sample size, and the confidence distribution.
        A tiny ECE can coexist with a weak classifier. Always inspect bin counts and
        negative log-likelihood (NLL); calibration cannot repair incorrect class rankings.
        ''') ,
        code('''
        def reliability(logits, labels, bins=8):
            probabilities = logits.softmax(1)
            confidence, prediction = probabilities.max(1)
            correct = prediction.eq(labels).float()
            bin_id = torch.clamp((confidence * bins).long(), max=bins - 1)
            rows = []
            for b in range(bins):
                mask = bin_id == b
                count = int(mask.sum())
                rows.append({'bin': b, 'count': count,
                             'confidence': confidence[mask].mean().item() if count else np.nan,
                             'accuracy': correct[mask].mean().item() if count else np.nan})
            table = pd.DataFrame(rows)
            ece = ((table['count'] / len(labels)) *
                   (table['confidence'] - table['accuracy']).abs()).sum()
            return table, float(ece)

        # Calibration-set graphs are diagnostics, not an unbiased final score.
        before, ece_before = reliability(cal_logits, cal_targets)
        after, ece_after = reliability(cal_logits / temperature, cal_targets)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for table, label in [(before, 'Original'), (after, 'Temperature-scaled')]:
            used = table['count'] > 0
            axes[0].plot(table.loc[used, 'confidence'], table.loc[used, 'accuracy'], 'o-', label=label)
        axes[0].plot([0, 1], [0, 1], 'k--')
        axes[0].set(xlabel='Mean confidence', ylabel='Observed accuracy', title='Calibration split')
        axes[0].legend()
        axes[1].bar(after['bin'], after['count'])
        axes[1].set(xlabel='Confidence bin', ylabel='Images', title='Scaled confidence support')
        plt.tight_layout(); plt.show()
        print({'calibration_ece_before': ece_before, 'calibration_ece_after': ece_after})
        ''') ,
        md('''
        ## 6. Open the test set once

        The model, checkpoint, weighting choice, temperature method, and bins are now
        fixed. One forward pass provides logits for both original and scaled
        probabilities. Report both; do not choose the method after seeing test results.
        Calibration may help NLL while ECE gets worse, especially with few examples.
        ''') ,
        code('''
        test_result = evaluate(chosen_model, test_loader, device=device)
        test_logits, test_targets = test_result['logits'], test_result['targets']
        rows = []
        for name, scale in [('original', 1.0), ('temperature_scaled', temperature)]:
            logits = test_logits / scale
            _, ece = reliability(logits, test_targets)
            rows.append({'method': name, **metric_row({'logits': logits, 'targets': test_targets}),
                         'nll': F.cross_entropy(logits, test_targets).item(), 'ece_8_bins': ece})
        final_report = pd.DataFrame(rows)
        display(final_report)
        assert torch.equal(test_logits.argmax(1), (test_logits / temperature).argmax(1))
        out_dir = ARTIFACTS / '09_evaluation'; out_dir.mkdir(parents=True, exist_ok=True)
        final_report.to_csv(out_dir / 'test_metrics.csv', index=False)
        print('Saved:', out_dir / 'test_metrics.csv')
        ''') ,
        md('''
        ## Practice 1 · Why can accuracy be misleading?

        Change the number of majority-class examples below. Predict accuracy and
        balanced accuracy before running. This uses synthetic labels, so it leaves
        your test protocol intact.
        ''') ,
        code('''
        majority_count = 90  # Try 50 and 990.
        synthetic_truth = np.array([0] * majority_count + [1] * 10)
        always_zero = np.zeros_like(synthetic_truth)
        print('Accuracy:', accuracy_score(synthetic_truth, always_zero))
        print('Balanced accuracy:', balanced_accuracy_score(synthetic_truth, always_zero))
        ''') ,
        md('''
        <details><summary>Solution idea</summary>
        Accuracy is majority_count / (majority_count + 10), while balanced accuracy
        stays at 0.5 because class 0 recall is 1 and class 1 recall is 0.
        </details>

        ## Practice 2 · Change ECE bins

        Recompute ECE using calibration logits with 3, 5, 10, and 20 bins. Keep the
        temperature fixed. Explain why choosing whichever bin count looks best would
        misrepresent calibration quality.
        ''') ,
        code('''
        bin_experiment = []
        for bins in [3, 5, 10, 20]:
            table, score = reliability(cal_logits / temperature, cal_targets, bins=bins)
            bin_experiment.append({'bins': bins, 'ece': score,
                                   'nonempty_bins': int((table['count'] > 0).sum())})
        display(pd.DataFrame(bin_experiment))
        ''') ,
        md('''
        <details><summary>Solution idea</summary>
        Bin boundaries regroup examples, altering both gaps and estimation noise.
        Report a predefined scheme with support counts and NLL, and use enough data.
        </details>

        ## Practice 3 · Explore abstention on validation data

        A classifier can abstain when confidence is low. Change `threshold` and
        measure coverage (fraction answered) and accuracy among answers. Use
        validation to design a rule; evaluating a new final rule needs a fresh
        held-out test because this notebook has already revealed its test results.
        ''') ,
        code('''
        threshold = 0.6  # Try 0.3 and 0.8.
        confidence, prediction = (chosen_val['logits'] / temperature).softmax(1).max(1)
        accepted = confidence >= threshold
        selective_accuracy = (prediction[accepted] == chosen_val['targets'][accepted]).float().mean().item() if accepted.any() else float('nan')
        print({'threshold': threshold, 'coverage': accepted.float().mean().item(),
               'accuracy_when_accepted': selective_accuracy, 'accepted_images': int(accepted.sum())})
        ''') ,
        md('''
        <details><summary>Solution idea</summary>
        A higher threshold often reduces coverage; accuracy need not improve
        monotonically on a small sample. No accepted examples means undefined
        selective accuracy, not perfect accuracy. Inspect classwise coverage too.
        </details>

        **Completion check:** explain the four split roles, read a confusion-matrix
        row, compute inverse-frequency weights, and explain why temperature preserves
        argmax. Weighted training and calibration are useful tools, not guaranteed wins.

        **Primary sources:** [scikit-learn classification metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics),
        [probability calibration](https://scikit-learn.org/stable/modules/calibration.html),
        [Guo et al., temperature scaling](https://proceedings.mlr.press/v70/guo17a.html).

        **Next:** [10 · Experiments and checkpoints](10_experiments_checkpoints.ipynb).
        ''') ,
    ]


def lesson10():
    return [
        md('''
        # 10 · Run controlled experiments and resume training

        **Prerequisites:** CNN training, validation, and Python dictionaries.
        **Time:** 35–50 minutes. No downloads or earlier notebook artifacts required.

        You will compare two learning rates with the same initialization and batch
        order, select checkpoints by validation loss, use a scheduler and early
        stopping, save experiment records, and actually resume optimizer and random
        state. Keep the final test result separate from experiment selection.
        ''') ,
        bootstrap(),
        code('''
        import copy
        import json
        import random
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import torch
        from torch import nn
        from torch.utils.data import DataLoader
        from course_utils import get_digits_loaders, TinyCNN, evaluate
        train_source, val_loader, test_loader = get_digits_loaders(batch_size=64, seed=42)
        out_dir = ARTIFACTS / '10_experiments'; out_dir.mkdir(parents=True, exist_ok=True)
        max_epochs = 1 if SMOKE_TEST else 8
        ''') ,
        md('''
        ## 1. Write down the comparison before training

        Compare Adam learning rates 0.001 and 0.003; hold all other settings fixed.
        We use the same model initialization, split, shuffling seed, batch size,
        optimizer, epoch cap, and checkpoint criterion. The chosen result has the
        lowest validation cross-entropy. A single seed teaches control but does not
        establish statistical superiority; a serious claim needs repeated seeds.
        ''') ,
        code('''
        configs = [dict(name='lr_0p001', lr=0.001, seed=42),
                   dict(name='lr_0p003', lr=0.003, seed=42)]
        display(pd.DataFrame(configs))

        def make_run(config):
            seed_everything(config['seed'])
            model = TinyCNN().to(device)
            generator = torch.Generator().manual_seed(config['seed'])
            loader = DataLoader(train_source.dataset, batch_size=64, shuffle=True,
                                generator=generator, num_workers=0)
            optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=1)
            return model, optimizer, scheduler, loader, generator
        ''') ,
        md('''
        ## 2. Make one training epoch explicit

        `model.train()` enables training behavior; `evaluate()` switches to evaluation
        behavior. The scheduler receives validation loss **after** each epoch.
        Early stopping ends a run after three epochs without meaningful improvement.
        With eight epochs it may never trigger; the mechanism is still present.
        ''') ,
        code('''
        def train_epoch(model, optimizer, loader):
            model.train()
            total_loss, total = 0.0, 0
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = nn.functional.cross_entropy(model(images), labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * labels.numel()
                total += labels.numel()
            return total_loss / total

        def advance_epoch(model, optimizer, scheduler, loader, epoch):
            lr_used = optimizer.param_groups[0]['lr']
            train_loss = train_epoch(model, optimizer, loader)
            result = evaluate(model, val_loader, device=device)
            scheduler.step(result['loss'])
            return {'epoch': epoch, 'lr': lr_used, 'train_loss': train_loss,
                    'val_loss': result['loss'], 'val_accuracy': result['accuracy']}
        ''') ,
        md('''
        ## 3. A resumable checkpoint contains more than weights

        We save the model, optimizer momentum, scheduler state, completed epoch,
        best validation checkpoint, patience counter, configuration, and history.
        We also save Python, NumPy, PyTorch CPU/CUDA, and DataLoader-generator RNG
        states. This demo saves **at epoch boundaries** with `num_workers=0`.

        Mid-epoch resume additionally requires sampler progress and worker state.
        Different devices, PyTorch releases, and nondeterministic GPU kernels can
        change numerics; this notebook does not promise bitwise equivalence across
        machines. Only load checkpoint files you trust.
        ''') ,
        code('''
        def capture_rng(generator):
            state = np.random.get_state()
            # Built-in values and tensors keep this compatible with weights_only=True.
            return {'python': random.getstate(),
                    'numpy': {'name': state[0], 'keys': state[1].tolist(),
                              'pos': state[2], 'has_gauss': state[3], 'cached': state[4]},
                    'torch': torch.get_rng_state(),
                    'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                    'loader': generator.get_state()}

        def restore_rng(state, generator):
            random.setstate(state['python'])
            n = state['numpy']
            np.random.set_state((n['name'], np.array(n['keys'], dtype=np.uint32),
                                 n['pos'], n['has_gauss'], n['cached']))
            torch.set_rng_state(state['torch'].cpu())
            if state['cuda'] and torch.cuda.is_available():
                torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])
            generator.set_state(state['loader'].cpu())
        ''') ,
        code('''
        def run_experiment(config):
            model, optimizer, scheduler, loader, generator = make_run(config)
            history, best_loss, best_epoch, bad_epochs = [], float('inf'), 0, 0
            best_state = None
            checkpoint_path = out_dir / (config['name'] + '_last.pt')
            for epoch in range(1, max_epochs + 1):
                row = advance_epoch(model, optimizer, scheduler, loader, epoch)
                history.append(row)
                if row['val_loss'] < best_loss - 1e-4:
                    best_loss, best_epoch, bad_epochs = row['val_loss'], epoch, 0
                    best_state = copy.deepcopy(model.state_dict())
                else:
                    bad_epochs += 1
                torch.save({'config': config, 'epoch': epoch,
                            'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                            'scheduler': scheduler.state_dict(), 'best_model': best_state,
                            'best_loss': best_loss, 'best_epoch': best_epoch,
                            'bad_epochs': bad_epochs, 'history': history,
                            'rng': capture_rng(generator)}, checkpoint_path)
                if bad_epochs >= 3:
                    break
            model.load_state_dict(best_state)
            torch.save(best_state, out_dir / (config['name'] + '_best_weights.pt'))
            return {'config': config, 'model': model, 'history': history,
                    'best_loss': best_loss, 'best_epoch': best_epoch,
                    'checkpoint_path': checkpoint_path}

        experiments = [run_experiment(config) for config in configs]
        print('Saved both last-state and best-weight checkpoints.')
        ''') ,
        md('''
        ## 4. Compare and select using validation only

        A best-weight checkpoint is convenient for inference. A last-state checkpoint
        is convenient for continuing a interrupted run. They answer different needs:
        pairing best weights with a last-epoch optimizer would silently mismatch state.
        ''') ,
        code('''
        summary = pd.DataFrame([{'name': e['config']['name'], 'lr': e['config']['lr'],
                                 'seed': e['config']['seed'], 'best_epoch': e['best_epoch'],
                                 'best_val_loss': e['best_loss'],
                                 'epochs_run': len(e['history'])} for e in experiments])
        display(summary.sort_values('best_val_loss'))
        chosen = min(experiments, key=lambda e: e['best_loss'])
        for e in experiments:
            h = pd.DataFrame(e['history'])
            plt.plot(h['epoch'], h['val_loss'], 'o-', label=e['config']['name'])
        plt.xlabel('Epoch'); plt.ylabel('Validation cross-entropy'); plt.legend(); plt.show()
        print('Selected before opening test:', chosen['config']['name'])
        ''') ,
        code('''
        summary.to_csv(out_dir / 'experiment_summary.csv', index=False)
        details = {'selection_metric': 'lowest validation cross-entropy',
                   'torch_version': str(torch.__version__), 'device': str(device),
                   'smoke_test': SMOKE_TEST,
                   'runs': [{'config': e['config'], 'history': e['history']} for e in experiments]}
        (out_dir / 'experiment_details.json').write_text(json.dumps(details, indent=2), encoding='utf-8')
        print('Records:', out_dir)
        ''') ,
        md('''
        ## 5. Restore the last training state and take another epoch

        Construct objects first, load their states, then restore RNG state last so
        initialization cannot consume the restored random stream. `map_location='cpu'`
        loads RNG tensors safely; optimizer loading maps its tensors to parameters.

        This demonstration continues the chosen configuration for one additional
        epoch. It is a separate validation-only experiment, so it does not silently
        replace the already selected best model.
        ''') ,
        code('''
        checkpoint = torch.load(chosen['checkpoint_path'], map_location='cpu', weights_only=True)

        def restore_run(checkpoint):
            objects = make_run(checkpoint['config'])
            model, optimizer, scheduler, loader, generator = objects
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            restore_rng(checkpoint['rng'], generator)
            return objects

        resumed, resumed_opt, resumed_sched, resumed_loader, resumed_gen = restore_run(checkpoint)
        resumed_row = advance_epoch(resumed, resumed_opt, resumed_sched, resumed_loader,
                                   checkpoint['epoch'] + 1)
        print('Resumed epoch:', resumed_row)
        print('Restored scheduler epochs:', resumed_sched.last_epoch)
        ''') ,
        code('''
        # Restore the same snapshot again and compare one-epoch continuation.
        replay, replay_opt, replay_sched, replay_loader, replay_gen = restore_run(checkpoint)
        replay_row = advance_epoch(replay, replay_opt, replay_sched, replay_loader,
                                  checkpoint['epoch'] + 1)
        max_difference = max((a.detach() - b.detach()).abs().max().item()
                             for a, b in zip(resumed.parameters(), replay.parameters()))
        print('Largest parameter difference after restored continuation:', max_difference)
        print('Validation-loss difference:', abs(resumed_row['val_loss'] - replay_row['val_loss']))
        print('Zero or tiny differences are expected on a consistent setup; GPU determinism varies.')
        ''') ,
        md('''
        ## 6. Evaluate the preselected model once

        Use the original best checkpoint selected in step 4. The resume demonstration
        is not part of test-based model selection. Write the result next to its
        configuration so a future reader knows exactly what was measured.
        ''') ,
        code('''
        final = evaluate(chosen['model'], test_loader, device=device)
        final_record = {'chosen_config': chosen['config'], 'best_epoch': chosen['best_epoch'],
                        'test_loss': final['loss'], 'test_accuracy': final['accuracy']}
        (out_dir / 'final_test.json').write_text(json.dumps(final_record, indent=2), encoding='utf-8')
        print(final_record)
        ''') ,
        md('''
        ## Practice 1 · Inspect Adam's hidden state

        Compare parameter count with optimizer-state keys. Find the accumulated
        first and second moments. Explain why restoring only weights is not the
        same as continuing training with Adam.
        ''') ,
        code('''
        first_state = next(iter(checkpoint['optimizer']['state'].values()))
        print('One parameter optimizer state:', {key: tuple(value.shape) if torch.is_tensor(value) else value
                                                 for key, value in first_state.items()})
        print('Model tensors:', len(checkpoint['model']))
        print('Optimizer parameter entries:', len(checkpoint['optimizer']['state']))
        ''') ,
        md('''
        <details><summary>Solution idea</summary>
        Adam maintains a step count, exp_avg (first moment), and exp_avg_sq (second
        moment). Resetting them changes the update even when the model weights match.
        </details>

        ## Practice 2 · Observe patience and scheduling without another model

        Feed a synthetic validation-loss sequence to a fresh scheduler. Change its
        patience and predict when the learning rate will drop. The tiny optimizer
        steps below preserve the normal optimizer-before-scheduler order.
        ''') ,
        code('''
        toy_parameter = nn.Parameter(torch.tensor(1.0))
        toy_optimizer = torch.optim.SGD([toy_parameter], lr=0.1)
        toy_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(toy_optimizer, patience=1, factor=0.5)
        for observed_loss in [1.0, 0.8, 0.81, 0.82, 0.83, 0.7]:
            toy_optimizer.zero_grad(); (toy_parameter ** 2).backward(); toy_optimizer.step()
            toy_scheduler.step(observed_loss)
            print('Loss:', observed_loss, '| next learning rate:', toy_optimizer.param_groups[0]['lr'])
        ''') ,
        md('''
        <details><summary>Solution idea</summary>
        With patience 1, the scheduler tolerates one bad epoch and reduces the rate
        after the next one. Early stopping has its own counter; choose its patience
        long enough to allow a learning-rate reduction a chance to help.
        </details>

        ## Practice 3 · Design a repeated-seed comparison

        Create the experiment plan below. Set `RUN_REPEATED_SEEDS=True` to run it
        and summarize validation losses. It never reads the test set. The optional
        runs use unique filenames, preserving the initial experiment records.
        ''') ,
        code('''
        repeated_plan = [dict(name=f'repeat_lr{lr}_seed{seed}', lr=lr, seed=seed)
                         for seed in [10, 20, 30] for lr in [0.001, 0.003]]
        display(pd.DataFrame(repeated_plan))
        RUN_REPEATED_SEEDS = False  # Change to True when ready for six short runs.
        if RUN_REPEATED_SEEDS and not SMOKE_TEST:
            repeated = [run_experiment(config) for config in repeated_plan]
            table = pd.DataFrame([{'lr': e['config']['lr'], 'seed': e['config']['seed'],
                                   'val_loss': e['best_loss']} for e in repeated])
            display(table.groupby('lr')['val_loss'].agg(['mean', 'std', 'count']))
        else:
            print('Plan ready. Enable the switch to estimate variation across seeds.')
        ''') ,
        md('''
        <details><summary>Solution idea</summary>
        Pair learning rates within each seed and compare per-seed differences.
        Report variation and compute cost; a tiny mean difference may not be robust.
        Repeated searches can overfit validation, so a final fresh holdout matters.
        </details>

        **Completion check:** identify your selection metric, distinguish last versus
        best checkpoints, restore optimizer/scheduler/RNG state, and trace a result
        from the JSON/CSV files back to its configuration.

        **Primary sources:** [PyTorch saving and loading](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html),
        [reproducibility notes](https://docs.pytorch.org/docs/2.11/notes/randomness.html),
        [ReduceLROnPlateau](https://docs.pytorch.org/docs/2.11/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html).

        **Next:** [11 · Mixed precision, mixup, and distillation](11_mixed_precision_mixup_distillation.ipynb).
        ''') ,
    ]


def lesson11():
    return [
        md('''
        # 11 · Mixed precision, mixup, and knowledge distillation

        **Prerequisites:** training loops, cross-entropy, and controlled experiments.
        **Time:** 40–60 minutes including exercises. Offline data; GPU is optional.

        You will use modern `torch.amp`, mix images with correctly paired labels,
        and teach a small student from a frozen teacher. We run controlled short
        comparisons and inspect validation results. These techniques solve different
        problems and are not automatic accuracy improvements.
        ''') ,
        bootstrap(),
        code('''
        import copy
        import time
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import torch
        from torch import nn
        import torch.nn.functional as F
        from course_utils import get_digits_loaders, TinyCNN, fit, evaluate
        epochs = 1 if SMOKE_TEST else 6
        train_loader, val_loader, test_loader = get_digits_loaders(batch_size=64, seed=42)
        use_amp = device.type == 'cuda'
        print('CUDA float16 AMP enabled:', use_amp, '| CPU runs float32.')
        ''') ,
        md('''
        ## 1. Understand autocasting and loss scaling

        Autocast selects lower precision for eligible operations; model parameters
        remain float32. `GradScaler` scales float16 losses to reduce gradient underflow.
        It unscales before optimizer updates and can skip updates with nonfinite
        gradients. If clipping, unscale first. Our CPU fallback disables autocast
        and scaling; it does not depend on CPU bfloat16 support.

        AMP helps many larger CUDA workloads, but tiny 8×8 models may be dominated
        by overhead. Measure after warm-up with CUDA synchronization; do not infer a
        GPU speedup from a single noisy notebook run.
        ''') ,
        code('''
        demo_model = TinyCNN().to(device)
        demo_optimizer = torch.optim.Adam(demo_model.parameters(), lr=0.003)
        demo_scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
        images, labels = next(iter(train_loader))
        images, labels = images.to(device), labels.to(device)
        demo_optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            demo_logits = demo_model(images)
            demo_loss = F.cross_entropy(demo_logits, labels)
        demo_scaler.scale(demo_loss).backward()
        demo_scaler.unscale_(demo_optimizer)
        gradient_norm = nn.utils.clip_grad_norm_(demo_model.parameters(), max_norm=5.0)
        demo_scaler.step(demo_optimizer)
        demo_scaler.update()
        print('Parameter dtype:', next(demo_model.parameters()).dtype)
        print('Logit dtype:', demo_logits.dtype, '| loss:', demo_loss.item(), '| gradient norm:', float(gradient_norm))
        ''') ,
        md('''
        ## 2. Mix two images and both targets

        Draw λ from a Beta distribution and form `mixed_x = λ*x + (1-λ)*x[perm]`.
        The supervised loss is `λ*CE(logits,y) + (1-λ)*CE(logits,y[perm])`.
        Keeping only the first label would teach contradictory targets. We use a
        single λ per batch for clarity; mixed-image accuracy is not ordinary accuracy.
        ''') ,
        code('''
        def mixup_batch(x, y, alpha=0.4):
            if alpha <= 0:
                return x, y, y, 1.0
            lam = float(np.random.beta(alpha, alpha))
            permutation = torch.randperm(x.size(0), device=x.device)
            return lam * x + (1.0 - lam) * x[permutation], y, y[permutation], lam

        mixed, y_a, y_b, lam = mixup_batch(images, labels)
        fig, axes = plt.subplots(1, 4, figsize=(9, 2.5))
        for i, ax in enumerate(axes):
            ax.imshow(mixed[i, 0].cpu(), cmap='gray')
            ax.set_title(f'{lam:.2f}×{y_a[i].item()} + {1-lam:.2f}×{y_b[i].item()}')
            ax.axis('off')
        plt.tight_layout(); plt.show()
        ''') ,
        md('''
        ## 3. Train with a switch for AMP and mixup

        Validation always uses clean images and ordinary labels. Save the lowest
        validation-loss state so every configuration uses the same selection rule.
        Synchronization brackets timing because CUDA work is asynchronous. These
        short timings include validation and checkpoint copies and are not benchmarks.
        ''') ,
        code('''
        def train_mixup(alpha=0.0, amp=False):
            seed_everything(42)
            train, val, _ = get_digits_loaders(batch_size=64, seed=42)
            model = TinyCNN().to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
            enabled = amp and device.type == 'cuda'
            scaler = torch.amp.GradScaler('cuda', enabled=enabled)
            best_loss, best_state, history = float('inf'), None, []
            if device.type == 'cuda':
                torch.cuda.synchronize()
            started = time.perf_counter()
            for epoch in range(epochs):
                model.train()
                for x, y in train:
                    x, y = x.to(device), y.to(device)
                    x, first, second, mix = mixup_batch(x, y, alpha)
                    optimizer.zero_grad(set_to_none=True)
                    with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=enabled):
                        logits = model(x)
                        loss = mix * F.cross_entropy(logits, first) + (1-mix) * F.cross_entropy(logits, second)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                result = evaluate(model, val, device=device)
                history.append({'epoch': epoch + 1, 'val_loss': result['loss'], 'val_accuracy': result['accuracy']})
                if result['loss'] < best_loss:
                    best_loss, best_state = result['loss'], copy.deepcopy(model.state_dict())
            if device.type == 'cuda':
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            model.load_state_dict(best_state)
            return model, history, elapsed
        ''') ,
        code('''
        fp32_model, fp32_history, fp32_seconds = train_mixup(alpha=0.0, amp=False)
        amp_model, amp_history, amp_seconds = train_mixup(alpha=0.0, amp=True)
        mixup_model, mixup_history, mixup_seconds = train_mixup(alpha=0.4, amp=True)
        cnn_models = {'CNN float32': fp32_model, 'CNN AMP': amp_model, 'CNN AMP + mixup': mixup_model}
        cnn_table = []
        for (name, model), seconds in zip(cnn_models.items(), [fp32_seconds, amp_seconds, mixup_seconds]):
            result = evaluate(model, val_loader, device=device)
            cnn_table.append({'name': name, 'val_loss': result['loss'],
                              'val_accuracy': result['accuracy'], 'seconds_including_validation': seconds})
        display(pd.DataFrame(cnn_table))
        print('On CPU, the first two runs use the same float32 path.')
        ''') ,
        md('''
        ## 4. Define a student and freeze a teacher

        We reuse the trained float32 CNN as teacher. The student is a smaller MLP.
        First train a hard-label-only student baseline; then train the same student
        initialization with distillation. Teacher quality matters: an undertrained
        teacher can transfer bad predictions, and the smoke-test teacher is intentionally
        weak. Check parameter counts rather than assuming one model is smaller.
        ''') ,
        code('''
        teacher = fp32_model
        teacher.eval()
        teacher.requires_grad_(False)
        teacher.zero_grad(set_to_none=True)

        class Student(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(nn.Flatten(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 10))
            def forward(self, x):
                return self.net(x)

        def parameter_count(model):
            return sum(p.numel() for p in model.parameters())

        print('Teacher parameters:', parameter_count(teacher))
        print('Student parameters:', parameter_count(Student()))
        print('Teacher validation accuracy:', evaluate(teacher, val_loader, device=device)['accuracy'])
        ''') ,
        md('''
        ## 5. Build the distillation loss

        Hard labels use ordinary cross-entropy. Soft targets use temperature-softened
        teacher probabilities and student log-probabilities. PyTorch KL divergence
        expects log-probabilities as its input; `reduction='batchmean'` divides by
        batch size. Multiply the soft loss by `T²` to compensate for temperature's
        gradient scaling. Teacher logits are detached and never backpropagated.

        `alpha` below is the soft-target contribution, so alpha=0 is the supervised
        baseline. Keep temperature and alpha fixed before evaluating final test data.
        ''') ,
        code('''
        def distillation_loss(student_logits, teacher_logits, labels, temperature=3.0, alpha=0.5):
            hard = F.cross_entropy(student_logits, labels)
            soft = F.kl_div(F.log_softmax(student_logits / temperature, dim=1),
                            F.softmax(teacher_logits.detach() / temperature, dim=1),
                            reduction='batchmean') * temperature ** 2
            return (1-alpha) * hard + alpha * soft, hard.detach(), soft.detach()

        with torch.no_grad():
            teacher_logits = teacher(images)
        demo_student = Student().to(device)
        loss, hard, soft = distillation_loss(demo_student(images), teacher_logits, labels)
        print({'combined': loss.item(), 'hard_ce': hard.item(), 'soft_kl_scaled': soft.item()})
        ''') ,
        code('''
        def train_student(alpha):
            seed_everything(42)
            train, val, _ = get_digits_loaders(batch_size=64, seed=42)
            student = Student().to(device)
            optimizer = torch.optim.Adam(student.parameters(), lr=0.003)
            best_loss, best_state = float('inf'), None
            for epoch in range(epochs):
                student.train()
                for x, y in train:
                    x, y = x.to(device), y.to(device)
                    with torch.no_grad():
                        teacher_output = teacher(x)
                    optimizer.zero_grad(set_to_none=True)
                    loss, _, _ = distillation_loss(student(x), teacher_output, y,
                                                   temperature=3.0, alpha=alpha)
                    loss.backward()
                    optimizer.step()
                result = evaluate(student, val, device=device)
                if result['loss'] < best_loss:
                    best_loss, best_state = result['loss'], copy.deepcopy(student.state_dict())
            student.load_state_dict(best_state)
            return student

        hard_student = train_student(alpha=0.0)
        distilled_student = train_student(alpha=0.5)
        # Freeze the teacher's behavior and verify no teacher gradient was created.
        assert all(p.grad is None for p in teacher.parameters())
        ''') ,
        md('''
        ## 6. Compare students fairly, then evaluate the selected student

        Both students see identical splits, initialization, batch order, epochs,
        and optimizer settings. Choose by validation cross-entropy, then perform
        one final test evaluation. This is a small educational comparison, not a
        general claim that distillation beats supervised learning. It also required
        a teacher's training and inference cost, which should be reported in practice.
        ''') ,
        code('''
        students = {'hard labels': hard_student, 'distilled': distilled_student}
        student_rows = []
        for name, model in students.items():
            result = evaluate(model, val_loader, device=device)
            student_rows.append({'student': name, 'val_loss': result['loss'],
                                 'val_accuracy': result['accuracy'], 'parameters': parameter_count(model)})
        student_table = pd.DataFrame(student_rows)
        display(student_table)
        chosen_name = student_table.loc[student_table['val_loss'].idxmin(), 'student']
        final = evaluate(students[chosen_name], test_loader, device=device)
        print('Selected:', chosen_name, '| test loss:', final['loss'], '| test accuracy:', final['accuracy'])
        out_dir = ARTIFACTS / '11_training_techniques'; out_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(cnn_table).to_csv(out_dir / 'cnn_validation.csv', index=False)
        student_table.to_csv(out_dir / 'student_validation.csv', index=False)
        ''') ,
        md('''
        ## Practice 1 · Verify the mixup endpoints

        Show that λ=1 equals the first-label loss and λ=0 equals the second-label
        loss. Change the labels and observe that the identities still hold. This
        is an algebra check, not a training run.
        ''') ,
        code('''
        toy_logits = torch.tensor([[2.0, -1.0], [-0.5, 1.5]])
        toy_a, toy_b = torch.tensor([0, 1]), torch.tensor([1, 0])
        ce_a, ce_b = F.cross_entropy(toy_logits, toy_a), F.cross_entropy(toy_logits, toy_b)
        for coefficient in [0.0, 0.5, 1.0]:
            print(coefficient, (coefficient * ce_a + (1-coefficient) * ce_b).item())
        assert torch.allclose(1.0 * ce_a + 0.0 * ce_b, ce_a)
        assert torch.allclose(0.0 * ce_a + 1.0 * ce_b, ce_b)
        ''') ,
        md('''
        <details><summary>Solution idea</summary>
        The mixed loss interpolates between two valid cross-entropies. Feeding a
        single integer label for every mixed image loses the second target.
        </details>

        ## Practice 2 · See what temperature does

        Plot teacher probabilities for one validation image at different temperatures.
        Higher temperatures soften the distribution. Argmax remains unchanged for
        any positive temperature; soft relative preferences become more visible.
        ''') ,
        code('''
        sample_x, sample_y = next(iter(val_loader))
        with torch.no_grad():
            sample_logits = teacher(sample_x[:1].to(device)).float().cpu()[0]
        for t in [1.0, 3.0, 8.0]:
            plt.plot(range(10), (sample_logits / t).softmax(0), 'o-', label=f'T={t}')
        plt.xticks(range(10)); plt.xlabel('Digit'); plt.ylabel('Probability')
        plt.title(f'Teacher probabilities; true digit {sample_y[0].item()}')
        plt.legend(); plt.show()
        ''') ,
        md('''
        <details><summary>Solution idea</summary>
        As temperature increases, softmax approaches a uniform distribution. Very
        high temperature can dilute useful information. Distillation temperature
        and post-training calibration temperature serve different objectives.
        </details>

        ## Practice 3 · Calculate memory and plan a real AMP benchmark

        Compute activation storage for a hypothetical batch at float32 and float16.
        Then use a larger real workload to benchmark after warm-up. Arithmetic below
        estimates only the named tensor; parameters, gradients, optimizer state,
        saved activations, and allocator overhead add memory.
        ''') ,
        code('''
        batch, channels, height, width = 32, 64, 56, 56
        elements = batch * channels * height * width
        for dtype in [torch.float32, torch.float16]:
            megabytes = elements * torch.tensor([], dtype=dtype).element_size() / 1024**2
            print(dtype, f'{megabytes:.1f} MiB for this activation tensor')
        if device.type == 'cuda':
            print('Current allocated GPU memory (MiB):', torch.cuda.memory_allocated() / 1024**2)
        ''') ,
        md('''
        <details><summary>Solution idea</summary>
        Float16 uses half the storage per element. AMP does not halve all training
        memory because float32 weights, optimizer states, and some operations remain.
        Benchmark the same model and batch shape with warm-up, synchronized timings,
        multiple repetitions, and peak memory reset immediately before measurement.
        </details>

        **Completion check:** describe autocast versus scaling, implement paired-label
        mixup, identify KL argument order, and explain detached teacher logits and T².

        **Primary sources:** [torch.amp](https://docs.pytorch.org/docs/2.11/amp.html),
        [mixup paper](https://arxiv.org/abs/1710.09412),
        [PyTorch distillation tutorial](https://docs.pytorch.org/tutorials/beginner/knowledge_distillation_tutorial.html),
        [Hinton et al., distillation](https://arxiv.org/abs/1503.02531).

        **Next:** [12 · Build a small vision transformer](12_vision_transformer.ipynb).
        ''') ,
    ]


def lesson12():
    return [
        md('''
        # 12 · Build a vision transformer from image patches

        **Prerequisites:** CNNs, tensor shapes, training loops, and validation.
        **Time:** 40–60 minutes with exercises. This notebook runs fully offline.

        You will turn images into patch tokens, add a learned class token and
        positional embeddings, pass them through a TransformerEncoder, and train a
        small classifier. Compare against a CNN with a fixed equal-epoch budget.

        This is an architectural learning exercise. The original ViT results rely
        on large-scale training; a tiny transformer trained from scratch on digits
        is not a substitute for pretrained models or large-dataset evidence.
        ''') ,
        bootstrap(),
        code('''
        import math
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import torch
        from torch import nn
        import torch.nn.functional as F
        from course_utils import get_digits_loaders, TinyCNN, fit, evaluate
        train_loader, val_loader, test_loader = get_digits_loaders(batch_size=64, image_size=16, seed=42)
        images, labels = next(iter(train_loader))
        epochs = 1 if SMOKE_TEST else 8
        print('Input batch:', tuple(images.shape), '| labels:', tuple(labels.shape))
        ''') ,
        md('''
        ## 1. Divide an image into patches

        We resize 8×8 digits to 16×16 solely to make a 4×4 patch grid easy to inspect;
        resizing adds pixels but no new visual information. Patch size 4 produces
        `(16/4)² = 16` tokens. Each grayscale patch contains 16 raw values.
        `unfold` below exposes those raw patches for learning and visualization.
        ''') ,
        code('''
        patch_size = 4
        raw_patches = images.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        # [B,C,grid_h,grid_w,patch_h,patch_w] -> [B,num_patches,patch_values]
        flat_patches = raw_patches.permute(0, 2, 3, 1, 4, 5).reshape(images.size(0), 16, 16)
        print('Raw patch axes:', tuple(raw_patches.shape))
        print('Flattened tokens:', tuple(flat_patches.shape))
        fig, axes = plt.subplots(1, 2, figsize=(8, 3))
        axes[0].imshow(images[0, 0], cmap='gray')
        axes[0].set_xticks(np.arange(-0.5, 16, 4), minor=True)
        axes[0].set_yticks(np.arange(-0.5, 16, 4), minor=True)
        axes[0].grid(which='minor', color='tab:red', linewidth=1)
        axes[0].set_title(f'Digit {labels[0].item()} with patch boundaries')
        axes[1].imshow(flat_patches[0], cmap='gray', aspect='auto')
        axes[1].set(xlabel='Value within patch', ylabel='Patch index', title='16 raw patch tokens')
        plt.tight_layout(); plt.show()
        ''') ,
        md('''
        ## 2. Learn an embedding for each patch

        A convolution with kernel size and stride equal to the patch size applies
        the same learned linear projection to every nonoverlapping patch. Its
        `embed_dim` output channels become token features. This is an efficient
        equivalent to flattening each patch and applying one Linear layer.
        ''') ,
        code('''
        embed_dim = 48
        patch_projection = nn.Conv2d(1, embed_dim, kernel_size=4, stride=4)
        patch_map = patch_projection(images)
        patch_tokens = patch_map.flatten(2).transpose(1, 2)
        print('Patch map [B,D,grid_h,grid_w]:', tuple(patch_map.shape))
        print('Tokens [B,N,D]:', tuple(patch_tokens.shape))
        assert patch_tokens.shape[1:] == (16, 48)
        ''') ,
        md('''
        ## 3. Add a class token and position information

        Attention compares tokens without inherently knowing their image locations.
        Learned positional embeddings identify locations in this fixed-size grid.
        A learned class token is prepended; its final representation summarizes
        information useful for classification. The sequence now has 17 tokens.
        ''') ,
        code('''
        class_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        positional_embedding = nn.Parameter(torch.zeros(1, 17, embed_dim))
        sequence = torch.cat([class_token.expand(images.size(0), -1, -1), patch_tokens], dim=1)
        sequence = sequence + positional_embedding
        print('Sequence [batch, 1 + patches, features]:', tuple(sequence.shape))
        ''') ,
        md('''
        ## 4. Inspect the attention calculation

        For one attention head, learned projections produce queries Q, keys K, and
        values V. `softmax(Q @ K.T / sqrt(head_dim)) @ V` mixes value vectors according
        to query-key compatibility. Each attention row sums to one. Multiple heads
        learn separate comparisons, followed by an output projection.

        This next matrix uses random projections for the concept only. It is not
        the trained model's attention and is not an explanation of its predictions.
        ''') ,
        code('''
        head_dim = 12
        token_example = sequence[:1].detach()
        qkv = nn.Linear(embed_dim, 3 * head_dim)(token_example)
        q, k, v = qkv.chunk(3, dim=-1)
        attention = (q @ k.transpose(-2, -1) / math.sqrt(head_dim)).softmax(dim=-1)
        attended_values = attention @ v
        print('Attention:', tuple(attention.shape), '| output:', tuple(attended_values.shape))
        print('First three row sums:', attention[0].sum(-1)[:3].detach())
        plt.imshow(attention[0].detach(), cmap='viridis')
        plt.xlabel('Key token'); plt.ylabel('Query token')
        plt.title('Illustrative random single-head attention'); plt.colorbar(); plt.show()
        ''') ,
        md('''
        ## 5. Assemble the model

        Each encoder block combines multi-head attention, residual connections,
        normalization, and a feed-forward network. `batch_first=True` keeps shapes
        `[B,N,D]`. Two encoder layers with four heads are enough to inspect the
        machinery. The class token goes through normalization and a ten-class head.

        TransformerEncoder clones the initial layer. We explicitly reinitialize
        cloned layers so they do not start with identical projection weights.
        ''') ,
        code('''
        class TinyViT(nn.Module):
            def __init__(self, image_size=16, patch_size=4, dim=48, heads=4, depth=2):
                super().__init__()
                assert image_size % patch_size == 0 and dim % heads == 0
                self.image_size = image_size
                self.patch_size = patch_size
                self.num_patches = (image_size // patch_size) ** 2
                self.patch_embed = nn.Conv2d(1, dim, kernel_size=patch_size, stride=patch_size)
                self.cls = nn.Parameter(torch.zeros(1, 1, dim))
                self.position = nn.Parameter(torch.zeros(1, self.num_patches + 1, dim))
                layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads,
                    dim_feedforward=dim * 2, dropout=0.1, activation='gelu', batch_first=True)
                self.encoder = nn.TransformerEncoder(layer, num_layers=depth, enable_nested_tensor=False)
                self.norm = nn.LayerNorm(dim)
                self.head = nn.Linear(dim, 10)
                self.apply(self.initialize)
                # in_proj_weight is a parameter of MultiheadAttention, not a Linear child.
                for block in self.encoder.layers:
                    nn.init.xavier_uniform_(block.self_attn.in_proj_weight)
                    if block.self_attn.in_proj_bias is not None:
                        nn.init.zeros_(block.self_attn.in_proj_bias)
                nn.init.trunc_normal_(self.cls, std=0.02)
                nn.init.trunc_normal_(self.position, std=0.02)

            @staticmethod
            def initialize(module):
                if isinstance(module, (nn.Linear, nn.Conv2d)):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif isinstance(module, nn.LayerNorm):
                    nn.init.ones_(module.weight); nn.init.zeros_(module.bias)

            def tokenize(self, x):
                if x.shape[-2:] != (self.image_size, self.image_size):
                    raise ValueError('Input size must match the configured positional grid.')
                tokens = self.patch_embed(x).flatten(2).transpose(1, 2)
                return torch.cat([self.cls.expand(x.size(0), -1, -1), tokens], dim=1) + self.position

            def forward(self, x):
                encoded = self.encoder(self.tokenize(x))
                return self.head(self.norm(encoded[:, 0]))

        shape_model = TinyViT()
        with torch.no_grad():
            print('Embedded sequence:', tuple(shape_model.tokenize(images[:2]).shape))
            print('Classification logits:', tuple(shape_model(images[:2]).shape))
        ''') ,
        md('''
        ## 6. Compare with a CNN under a declared budget

        Both models get the same input resolution, data split, batch size, seed,
        optimizer family, learning rate, and epoch budget. Each uses its best
        validation-loss checkpoint. Architectures have different parameter counts
        and different optimization needs, so equal epochs are an instructional
        comparison, not equal compute or equally tuned performance.

        A CNN's local structure often helps on small datasets. Expect the transformer
        to need more careful tuning and data; it is fine if the CNN wins here.
        ''') ,
        code('''
        models, histories = {}, {}
        for name, factory in [('CNN', TinyCNN), ('ViT', TinyViT)]:
            seed_everything(42)
            train, val, _ = get_digits_loaders(batch_size=64, image_size=16, seed=42)
            model = factory().to(device)
            history = fit(model, train, val, epochs=epochs, lr=0.003,
                          weight_decay=1e-4, device=device)
            models[name], histories[name] = model, history
        print('Both models trained; test data remain unopened.')
        ''') ,
        code('''
        rows = []
        for name, model in models.items():
            result = evaluate(model, val_loader, device=device)
            rows.append({'model': name, 'parameters': sum(p.numel() for p in model.parameters()),
                         'val_loss': result['loss'], 'val_accuracy': result['accuracy']})
        validation_comparison = pd.DataFrame(rows)
        display(validation_comparison)
        for name, history in histories.items():
            table = pd.DataFrame(history)
            plt.plot(table['epoch'], table['val_loss'], 'o-', label=name)
        plt.xlabel('Epoch'); plt.ylabel('Validation cross-entropy'); plt.legend(); plt.show()
        ''') ,
        md('''
        ## 7. Evaluate one preselected model and inspect predictions

        Select by validation loss, then open test once. Show fixed test examples
        rather than searching for the most impressive successes. Continue exploring
        architecture on validation data; reserve new final data for new claims.
        ''') ,
        code('''
        chosen_name = validation_comparison.loc[validation_comparison['val_loss'].idxmin(), 'model']
        chosen_model = models[chosen_name]
        final = evaluate(chosen_model, test_loader, device=device)
        print('Selected:', chosen_name, '| test loss:', final['loss'], '| test accuracy:', final['accuracy'])
        test_images, _ = next(iter(test_loader))
        probabilities = final['logits'].softmax(1)
        confidence, predictions = probabilities.max(1)
        fig, axes = plt.subplots(1, 6, figsize=(12, 2.5))
        for i, ax in enumerate(axes):
            ax.imshow(test_images[i, 0], cmap='gray')
            ax.set_title(f'true {final["targets"][i].item()} → {predictions[i].item()}\\np={confidence[i]:.2f}')
            ax.axis('off')
        plt.tight_layout(); plt.show()
        out_dir = ARTIFACTS / '12_vit'; out_dir.mkdir(parents=True, exist_ok=True)
        validation_comparison.to_csv(out_dir / 'validation_comparison.csv', index=False)
        torch.save({'model_state': models['ViT'].state_dict(),
                    'config': {'image_size': 16, 'patch_size': 4, 'dim': 48, 'heads': 4, 'depth': 2}},
                   out_dir / 'tiny_vit.pt')
        print('Saved learning artifact:', out_dir / 'tiny_vit.pt')
        ''') ,
        md('''
        ## Practice 1 · Count attention cost

        Change patch size and calculate the number of tokens and attention-score
        entries per head. Dense attention's score matrix grows with the square of
        sequence length. Smaller patches preserve detail but consume more memory.
        ''') ,
        code('''
        cost_rows = []
        for patch in [2, 4, 8]:
            tokens = (16 // patch) ** 2 + 1
            cost_rows.append({'patch_size': patch, 'tokens_including_cls': tokens,
                              'attention_entries_per_head': tokens ** 2})
        display(pd.DataFrame(cost_rows))
        ''') ,
        md('''
        <details><summary>Solution idea</summary>
        Patch sizes 2, 4, and 8 yield 65, 17, and 5 tokens. Attention-score matrices
        contain 4225, 289, and 25 entries per head per image, respectively. Actual
        memory depends on the attention implementation and other activations too.
        </details>

        ## Practice 2 · Remove positional information for one diagnostic

        Temporarily zero the learned position tensor and measure **validation** loss.
        Restore it immediately. This measures reliance on learned positions in this
        model; it is not equivalent to training an architecture without positions.
        ''') ,
        code('''
        vit = models['ViT']
        original_position = vit.position.detach().clone()
        baseline_val = evaluate(vit, val_loader, device=device)
        try:
            with torch.no_grad():
                vit.position.zero_()
            no_position_val = evaluate(vit, val_loader, device=device)
        finally:
            with torch.no_grad():
                vit.position.copy_(original_position)
        print({'validation_loss_original': baseline_val['loss'],
               'validation_loss_zero_position': no_position_val['loss']})
        ''') ,
        md('''
        <details><summary>Solution idea</summary>
        Zeroing positions creates an intervention the model was not trained for.
        A proper architecture ablation retrains both variants across multiple seeds
        with controlled budgets. One validation difference cannot prove causality
        about transformers in general.
        </details>

        ## Practice 3 · Change patch size safely

        Build a patch-size-2 model, inspect shapes, and verify output dimensions.
        Its positional embedding length changes, so it requires newly initialized
        parameters (or an explicit positional interpolation method). This starter
        performs a forward pass only; optionally train it using validation data.
        ''') ,
        code('''
        alternative = TinyViT(patch_size=2).to(device)
        alternative.eval()
        with torch.no_grad():
            small_batch = images[:2].to(device)
            print('Alternative sequence:', tuple(alternative.tokenize(small_batch).shape))
            print('Alternative logits:', tuple(alternative(small_batch).shape))
        print('Alternative parameters:', sum(p.numel() for p in alternative.parameters()))
        ''') ,
        md('''
        <details><summary>Solution idea</summary>
        Expect [2,65,48] tokens and [2,10] logits. Patch projection parameters decrease
        while positional parameters increase. Training and attention compute can
        increase even when total parameter count changes little.
        </details>

        **Completion check:** trace `[B,1,16,16] → [B,16,48] → [B,17,48] → [B,10]`,
        explain class and position tokens, compute attention shapes, and state the
        limits of the CNN/ViT comparison. Finishing a small example builds foundations;
        expert practice also requires larger datasets, robust evaluation, and iteration.

        **Primary sources:** [Dosovitskiy et al., ViT](https://arxiv.org/abs/2010.11929),
        [Vaswani et al., attention](https://arxiv.org/abs/1706.03762),
        [PyTorch TransformerEncoder](https://docs.pytorch.org/docs/2.11/generated/torch.nn.TransformerEncoder.html).

        **Next:** [13 · Explainability and inference](13_explainability_and_inference.ipynb).
        ''') ,
    ]


if __name__ == '__main__':
    lessons = {
        '09_evaluation_imbalance_calibration.ipynb': lesson09(),
        '10_experiments_checkpoints.ipynb': lesson10(),
        '11_mixed_precision_mixup_distillation.ipynb': lesson11(),
        '12_vision_transformer.ipynb': lesson12(),
    }
    for name, cells in lessons.items():
        for index, cell in enumerate(cells):
            if cell['cell_type'] == 'code':
                compile(cell['source'], f'{name}:cell{index}', 'exec')
        write_notebook(name, cells)
        print(f'  {len(cells)} cells; all Python cells compile')
