"""Tests for allsky.embeddings.backbone: model selection and embedding width.

No test here loads hub weights: construction is lazy, so the identity and the
declared width can be checked offline.
"""

import pytest

from allsky.embeddings.backbone import (
    AVAILABLE_BACKBONES,
    DinoV2Backbone,
    Pooling,
    TorchvisionBackbone,
    build_backbone,
    embedding_dim,
)


class TestEmbeddingDim:
    @pytest.mark.parametrize(
        ("model", "expected"),
        [
            ("dinov2_vits14", 384),
            ("dinov2_vitb14", 768),
            ("dinov2_vitl14", 1024),
            ("dinov2_vitg14", 1536),
        ],
    )
    def test_cls_width_is_the_models_token_width(self, model: str, expected: int):
        assert embedding_dim(model, "cls") == expected
        assert embedding_dim(model, "mean") == expected

    @pytest.mark.parametrize(
        "model", ["dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14", "dinov2_vitg14"]
    )
    def test_cls_plus_mean_concatenates_two_tokens(self, model: str):
        assert embedding_dim(model, "cls+mean") == 2 * embedding_dim(model, "cls")

    def test_an_unknown_model_names_the_ones_it_accepts(self):
        with pytest.raises(ValueError, match="dinov2_vitb14"):
            embedding_dim("dinov2_vith14", "cls")

    def test_an_unknown_pooling_is_rejected(self):
        with pytest.raises(ValueError, match="pooling"):
            embedding_dim("dinov2_vits14", "cls+max")


class TestBackboneSelection:
    def test_the_larger_variants_are_selectable(self):
        assert "dinov2_vitb14" in AVAILABLE_BACKBONES
        assert "dinov2_vitl14" in AVAILABLE_BACKBONES

    @pytest.mark.parametrize("model", ["dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14"])
    def test_build_backbone_carries_the_requested_identity_and_width(self, model: str):
        backbone = build_backbone(model)

        assert backbone.name == model
        assert backbone.dim == embedding_dim(model, "cls")

    @pytest.mark.parametrize("model", [n for n in AVAILABLE_BACKBONES if n != "fake"])
    def test_every_advertised_backbone_builds_with_the_pooling_its_family_offers(self, model: str):
        """Only the four DINOv2 names were ever constructed, while
        ``configs/allsky/experiments/resnet50/`` and ``effnet/`` are shipped
        arms. ``weights='none'`` builds the architecture without fetching
        anything, so the whole advertised list is reachable here.
        """
        # A convolutional trunk emits no tokens, so it offers 'mean' alone; the
        # ViTs default to 'cls'.
        pooling: Pooling = "mean" if model in TorchvisionBackbone.HEADS else "cls"

        backbone = build_backbone(model, pooling=pooling, weights="none")

        assert backbone.name == model
        assert isinstance(backbone.dim, int)
        assert backbone.dim > 0

    @pytest.mark.parametrize("model", sorted(TorchvisionBackbone.HEADS))
    def test_a_convolutional_trunk_refuses_the_token_pooling_it_has_no_tokens_for(self, model: str):
        with pytest.raises(ValueError, match="emits no tokens"):
            build_backbone(model, pooling="cls", weights="none")

    def test_the_identity_is_per_instance_not_shared_by_the_class(self):
        """``extract.py`` records ``backbone.name`` into the store meta, so a
        class-level name would label a ViT-B store as ViT-S."""
        small = build_backbone("dinov2_vits14")
        base = build_backbone("dinov2_vitb14")

        assert small.name == "dinov2_vits14"
        assert base.name == "dinov2_vitb14"

    def test_pooling_widens_the_declared_dim(self):
        assert build_backbone("dinov2_vitb14", pooling="cls+mean").dim == 1536

    def test_image_size_must_be_a_multiple_of_the_patch_size(self):
        with pytest.raises(ValueError, match="patch size"):
            DinoV2Backbone(model="dinov2_vitb14", image_size=225)


def test_a_convnet_backbone_refuses_a_token_pooling_before_the_store_is_stamped() -> None:
    """``build_backbone("resnet50", pooling="cls")`` constructed a backbone whose first
    encode died, after ``extract_embeddings`` had already written the store's meta."""
    with pytest.raises(ValueError, match="pooling"):
        build_backbone("resnet50", pooling="cls")
