# Linear evaluation repair and reproduction status

This patch repairs the public linear-probe entry point. It does not implement
Table 4's equivariance evaluator or establish reproduction of paper metrics.
The audit baseline is commit `dbe5e9361953a28258a7f24d92f742a162376c77`.

## What was wrong

- The [original script][original] imports unavailable `utils`, `moco.vision_transformer_eval`,
  and `moco.resnet` modules. The public modules are `builder.utils`,
  `vision_transformer`, and `resnet`.
- The issue's proposed first two import changes are correct but insufficient.
  The old call supplies `patch_size`, `cls_layer`, and `if_baseline`, whereas
  the [public factory][factory] `vit_small(args, ssl_type, drop_path_rate, **kwargs)` requires an args
  object with scalar `equiv_layer` and `equiv_mode`.
- The original config has `equiv_layer`, not the script's `cls_layer` field.
  The script also indexes `args.seed` although neither its parser nor the
  supplied config defines it.
- The original loader always selects `student`, ignoring `--checkpoint_key`.
  Barlow Twins saves `model`; DINO wraps encoder keys with `backbone.`.
  Permissive loading also fails to reject missing encoder weights reliably.
- The supplied replacement `eval_parallel.py` uses the correct factory call
  form, but still reads absent `args.cls_layer`, passes the config's entire
  `equiv_layer` list to a factory expecting a scalar, and lacks a default seed.
  Its printed `if_baseline` flag does not configure the model. It also always
  loads `student` and is not a drop-in repair for the public config.
- `config/eval.yaml` **already exists**. The original loader prepends `config/`,
  so `--config eval.yaml` is valid when run from the repository root. The
  configured checkpoint paths refer to local files, not downloadable weights.

The repair adapts each config entry to the existing model interface, validates
config lengths and seeds, checks all backbone weights, and uses the same
feature-extraction path in fitting and evaluation. Worker exceptions propagate
to the caller instead of remaining in background threads. The backbone's
training architecture is unchanged.

## Configure and run a linear probe

Use the dependencies documented in the [README](../README.md), plus a compatible
torchvision installation. Edit [config/eval.example.yaml](../config/eval.example.yaml)
to point to a checkpoint you possess. The example is a runnable configuration
template after supplying weights and data; its values are not certified paper
hyperparameters. Run from the repository root:

```bash
CUDA_VISIBLE_DEVICES=0 python lincls_parallel.py \
  --config eval.example.yaml --epochs 50 --val_epoch 40 --seed 1 \
  --data_path /path/to/imagenet_eval \
  --test_path /path/to/held_out_imagenet
```

Each run needs one aligned entry in each of `ckpts`, `lrs`, `gpus`, `arch`,
`equiv_layer`, and `if_baseline`. Their list lengths must match. GPU indices are
relative to `CUDA_VISIBLE_DEVICES`; use `gpus: [0]` for the example above.
`--seed` supplies one shared seed; YAML can instead supply a seed list of the
same length. YAML fields override CLI fields with the same name.
These seeds initialize classifiers; parallel runs still share an augmented
data stream and global RNG state, and cuDNN benchmarking remains enabled.
This patch does not establish exact multi-run determinism.

- `if_baseline: false` selects `equiv_mode='erl'` and that entry's `equiv_layer`.
  `if_baseline: true` selects `equiv_mode='inv'` and layer 12. Boolean strings
  from the original config remain accepted. Match these settings to training.
- Public ViT-S supports patch size 16. By default, concatenate CLS tokens from
  the last four blocks: `4 * 384 = 1536` classifier inputs. Optional final patch
  averaging adds 384 inputs. ResNet-50 uses its 2048-dimensional pooled output.
- For SER, require `1 <= equiv_layer <= 12 - n_last_blocks` so every selected
  token sequence already contains CLS. With four blocks, layers 9 and 12 are
  rejected. This patch does not claim to restore Table 4's layer-9/layer-12
  linear-evaluation setup; silently treating a patch token as CLS is incorrect.
- `--checkpoint_key auto` selects `student` when present, otherwise `model`.
  Use `--checkpoint_key teacher` explicitly to select a teacher checkpoint.
  This choice applies to every run in that invocation.
  Supported wrapper prefixes and projection heads are removed before strict
  backbone loading. Missing, unexpected, or shape-incompatible backbone weights
  fail rather than leaving parts of the encoder randomly initialized.
- `--evaluate` was not implemented and is now rejected explicitly. This entry
  point fits a classifier on a frozen encoder and requires CUDA for that loop.

## Data and optimization settings

`data_path/train` fits the classifier, `data_path/val` selects its best epoch,
and `test_path` evaluates the selected classifier. All three use ImageFolder
class subdirectories and must have consistent class-to-index mappings. Keep
selection data separate from final test data; do not reuse the final test set
to select epochs or learning rates. The original split manifest, split creation
seed, checkpoint files, and run-to-paper-result mapping are not provided.

The original parser defaults are 100 epochs, validation starting at zero-based
epoch 60, batch size 2048, 24 loader workers, and 1000 labels. The example above
uses 50 epochs and starts selection at zero-based epoch 40. Require
`0 <= val_epoch < epochs`; otherwise no selected classifier can be tested.

Fitting uses `RandomResizedCrop(224)`, horizontal flip, and tensor conversion.
Selection/test use bicubic `Resize(256)`, `CenterCrop(224)`, and tensor conversion.
All use ImageNet mean `(0.485, 0.456, 0.406)` and standard deviation
`(0.229, 0.224, 0.225)`. Only the linear classifier is optimized with SGD,
momentum 0.9, zero weight decay, cross-entropy, and cosine LR decay to zero.
The effective starting LR is `lrs[i] * batch_size_per_gpu * world_size / 256`;
these independent parallel probe runs normally have `world_size=1`.

## Table 4: confirmed facts and remaining information

The [paper, Section 4.3 and Appendix A.2][paper] explicitly measures Table 4
equivariance at the **final layer**. Layer 3 is the main training regularization
location, not the measurement location. Its score is cosine similarity between
`rho_g(f(x))` and `f(rho_g(x))`, with relative transform `g = g2 g1^-1`.
It reports rotation, horizontal-flip, and scale scores separately.

The following are verified **training-code facts**, not a recovered Table 4
evaluation protocol ([augmentation helper][aug], [defaults][defaults]):

| Component | Public training implementation |
| --- | --- |
| Scale | Uniform area multiplier in `[0.7, 1.3]`; log-uniform aspect ratio in `[3/4, 4/3]`. These are not independent side-length scales. |
| Resolution | Round the derived token-grid dimensions, then multiply by patch stride 16 for ViT-S. |
| Resampling | Bicubic `torch.nn.functional.interpolate` for images and feature maps; PyTorch 2.6 defaults are `align_corners=False`, `antialias=False`. |
| Rotation | `torch.rot90(..., dims=[2,3])`, with quarter-turn count in `{0,1,2,3}` and a per-sample 180-degree flag. Positive counts rotate counterclockwise. |
| Pairing | Scale/aspect and quarter-turn count are shared within each view's batch. MoCo's two views use complementary per-sample flip/180-degree flags. |

The exact Table 4 evaluation scale range/distribution, rotation direction and
angle averaging, preprocessing, sample selection, checkpoints/seeds, token or
pooled representation, normalization/projection point, and cosine aggregation
are not recoverable from these files. In particular, a pooled embedding has no
spatial grid to rotate; the historical tensor/action choice needs confirmation.
The training helper combines transforms, so using it unchanged does not establish
the separate Table 4 evaluation conditions. Neither linear-evaluation script
computes these scores. No speculative evaluator is added by this patch.

## Verification scope

On 2026-10-11, all **28 tests passed** with Python 3.12.14, PyTorch
2.6.0+cpu, torchvision 0.21.0+cpu, Kornia 0.8.0, and ruamel.yaml 0.17.36.
The original launcher arguments now parse (including `--seed 1`), and the
entry point reaches the explicit CUDA requirement on a CPU-only machine.
Python compilation, `bash -n run_eval.sh`, and `git diff --check` also passed.

`python -m pytest -q tests/test_linear_eval.py` exercises CPU synthetic inputs/checkpoints,
including model construction, feature shapes, strict loading, config validation,
and worker-error propagation. This checks code behavior without GPU training.
It does not validate real checkpoint provenance, the missing data split, CUDA
parallel execution, ImageNet accuracy, or Table 4 metric reproduction.

## Issue #1 reply draft (not posted)

> Thanks for reporting this. Yes, `from builder import utils` and
> `import vision_transformer as vits` are the correct public import paths.
> They are necessary but not sufficient: the ResNet import also needs updating,
> and the evaluator must adapt each run's configuration to the public ViT
> constructor and checkpoint format. `config/eval.yaml` already exists and is
> resolved by the original config loader. The repair and CPU checks are
> documented here. Table 4 evaluates equivariance at the final layer, whereas
> layer 3 is the main regularization location. The linear-probe script does not
> compute Table 4 scores, and the available training transformations alone do
> not fully specify that evaluation protocol.

[original]: https://github.com/aitrics-chris/SER/blob/dbe5e9361953a28258a7f24d92f742a162376c77/lincls_parallel.py
[factory]: https://github.com/aitrics-chris/SER/blob/dbe5e9361953a28258a7f24d92f742a162376c77/vision_transformer.py#L414-L421
[aug]: https://github.com/aitrics-chris/SER/blob/dbe5e9361953a28258a7f24d92f742a162376c77/builder/utils.py#L903-L949
[defaults]: https://github.com/aitrics-chris/SER/blob/dbe5e9361953a28258a7f24d92f742a162376c77/main_moco.py#L102-L107
[paper]: https://arxiv.org/html/2603.06693v1
