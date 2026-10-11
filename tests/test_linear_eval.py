"""CPU regression checks; no dataset, pretrained download, or GPU is required."""

import warnings

import pytest
import torch

import lincls_parallel as evaluation


@pytest.fixture(scope="module", autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def arguments(baseline=False, arch="vit_small"):
    args = evaluation.get_args_parser().parse_args([])
    args.ckpts = ["unused.pth"]
    args.lrs = [0.1]
    args.gpus = [0]
    args.arch = [arch]
    args.equiv_layer = [12 if baseline else 3]
    args.if_baseline = ["True" if baseline else "False"]
    args.seed = 7
    args.epochs = 2
    args.val_epoch = 0
    evaluation.normalize_eval_args(args)
    return args


@pytest.fixture(scope="module", params=[False, True], ids=["ser", "baseline"])
def backbone(request):
    args = arguments(baseline=request.param)
    model, width = evaluation.build_backbone(args, 0)
    return model.eval(), width


@torch.no_grad()
def test_vit_features_match_encoder_and_pooling(backbone):
    model, width = backbone
    image = torch.randn(2, 3, 32, 48)
    features = evaluation.extract_features(model, image, 4, False)
    assert features.shape == (2, width) == (2, 1536)
    assert torch.isfinite(features).all()
    # The final CLS used by linear evaluation must equal the encoder's own CLS.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="User provided device_type of 'cuda'.*")
        expected_cls = model.forward_inv(image)
    torch.testing.assert_close(features[:, -384:], expected_cls)
    pooled = evaluation.extract_features(model, image, 4, True)
    assert pooled.shape == (2, 1920)
    torch.testing.assert_close(pooled[:, :1536], features)
    expected_patch_mean = model.get_intermediate_layers(image, 1)[0][:, 1:].mean(1)
    torch.testing.assert_close(pooled[:, -384:], expected_patch_mean)


@torch.no_grad()
def test_vit_image_size_224(backbone):
    model, width = backbone
    features = evaluation.extract_features(model, torch.randn(1, 3, 224, 224), 4, False)
    assert features.shape == (1, width)
    assert torch.isfinite(features).all()


def test_cpu_linear_probe_updates_only_classifier(backbone):
    model, width = backbone
    model.requires_grad_(False)
    classifier = evaluation.LinearClassifier(width, num_labels=7)
    optimizer = torch.optim.SGD(classifier.parameters(), lr=0.1)
    with torch.no_grad():
        features = evaluation.extract_features(model, torch.randn(2, 3, 32, 48), 4, False)
    loss = torch.nn.functional.cross_entropy(classifier(features), torch.tensor([1, 4]))
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(classifier.linear.weight.grad).all()
    assert torch.count_nonzero(classifier.linear.weight.grad) > 0
    assert all(parameter.grad is None for parameter in model.parameters())
    previous_weight = classifier.linear.weight.detach().clone()
    optimizer.step()
    assert not torch.equal(classifier.linear.weight, previous_weight)


@torch.no_grad()
def test_resnet_features_are_batch_vectors():
    args = arguments(arch="resnet50")
    model, width = evaluation.build_backbone(args, 0)
    model.eval()
    image = torch.randn(2, 3, 64, 64)
    features = evaluation.extract_features(model, image, 4, False)
    assert features.shape == (2, width) == (2, 2048)
    torch.testing.assert_close(features, model.forward_inv(image))


def test_cli_seed_and_explicit_config_path(tmp_path):
    config = tmp_path / "evaluation.yaml"
    config.write_text(
        "ckpts: [one.pth, two.pth]\n"
        "lrs: [0.1, 0.2]\ngpus: [0, 1]\n"
        "arch: [vit_small, vit_small]\n"
        "equiv_layer: [3, 12]\nif_baseline: [false, true]\n"
    )
    args = evaluation.get_args_parser().parse_args(
        ["--config", str(config), "--seed", "11", "--epochs", "2", "--val_epoch", "0"]
    )
    evaluation.over_write_args_from_file(args, args.config)
    evaluation.normalize_eval_args(args)
    assert args.seed == [11, 11]
    assert args.equiv_layer == [3, 12]
    assert len(args.ckpts) == len(args.gpus) == 2


def test_config_name_resolves_in_repository():
    args = evaluation.get_args_parser().parse_args([])
    evaluation.over_write_args_from_file(args, "eval.yaml")
    assert args.ckpts and len(args.ckpts) == len(args.gpus)


def test_mismatched_run_lists_fail():
    args = arguments()
    args.lrs = [0.1, 0.2]
    with pytest.raises(ValueError):
        evaluation.normalize_eval_args(args)


def test_parallel_worker_failure_reaches_caller():
    def worker(index):
        if index == 1:
            raise RuntimeError("failed evaluation worker")

    with pytest.raises(RuntimeError, match="failed evaluation worker"):
        evaluation.parallel_apply(worker, [(0,), (1,)])


@pytest.mark.parametrize("setting,value", [("patch_size", 8), ("n_last_blocks", 13)])
def test_unsupported_vit_settings_fail(setting, value):
    args = arguments()
    setattr(args, setting, value)
    with pytest.raises(ValueError):
        evaluation.build_backbone(args, 0)


def test_ser_rejects_blocks_without_cls():
    args = arguments()
    args.equiv_layer = [9]
    with pytest.raises(ValueError):
        evaluation.build_backbone(args, 0)


@pytest.mark.parametrize("layout", ["moco", "dino", "barlowtwins", "teacher"])
def test_checkpoint_layouts_restore_backbone(backbone, tmp_path, layout):
    model, _ = backbone
    state = model.state_dict()
    expected = state["cls_token"].clone()
    checkpoint_key = "auto"
    if layout == "moco":
        payload = {"student": state}
    elif layout == "barlowtwins":
        payload = {"model": state}
    elif layout == "dino":
        wrapped = {"module.backbone." + key: value for key, value in state.items()}
        wrapped["module.head.last_layer.weight"] = torch.ones(2, 2)
        wrapped["module.projector_equiv.weight"] = torch.ones(2, 2)
        payload = {"student": wrapped}
    else:
        payload = {
            "student": {"deliberately_invalid": torch.ones(1)},
            "teacher": {"backbone." + key: value for key, value in state.items()},
        }
        checkpoint_key = "teacher"
    checkpoint = tmp_path / "checkpoint.pth"
    torch.save(payload, checkpoint)
    with torch.no_grad():
        model.cls_token.add_(1)
    evaluation.load_backbone_checkpoint(model, checkpoint, checkpoint_key=checkpoint_key)
    torch.testing.assert_close(model.cls_token, expected)


@pytest.mark.parametrize("damage", ["missing", "unexpected", "shape"])
def test_bad_backbone_checkpoints_fail(backbone, tmp_path, damage):
    model, _ = backbone
    state = dict(model.state_dict())
    if damage == "missing":
        del state["blocks.0.attn.qkv.weight"]
    elif damage == "unexpected":
        state["unknown_backbone_parameter"] = torch.ones(1)
    else:
        state["pos_embed"] = torch.zeros(1, 1, 384)
    checkpoint = tmp_path / "broken.pth"
    torch.save({"student": state}, checkpoint)
    with pytest.raises((ValueError, RuntimeError)):
        evaluation.load_backbone_checkpoint(model, checkpoint)
