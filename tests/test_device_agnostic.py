"""Device selection tests for the inference path. The mocked tests need no accelerator."""

import unittest
from unittest import mock

import torch

from uninavid import device as dev
from uninavid.model import builder
from uninavid.model.multimodal_encoder.eva_vit import VisionTransformer

HAS_XPU = hasattr(torch, "xpu") and torch.xpu.is_available()


def fake_host(xpu=False, cuda=False):
    available = {"xpu": xpu, "cuda": cuda, "cpu": True}
    return mock.patch.object(dev, "is_available", side_effect=lambda name: available[name])


class DeviceModuleTest(unittest.TestCase):
    def test_detection_order(self):
        with fake_host():
            self.assertEqual(dev.default_device_type(), "cpu")
            self.assertEqual(dev.resolve_device(), torch.device("cpu"))
        with fake_host(cuda=True):
            self.assertEqual(dev.resolve_device(), torch.device("cuda", 0))
        with fake_host(xpu=True, cuda=True):
            self.assertEqual(dev.resolve_device(), torch.device("xpu", 0))

    def test_explicit_device_wins(self):
        with fake_host(xpu=True):
            self.assertEqual(dev.resolve_device("cpu"), torch.device("cpu"))
            self.assertEqual(dev.resolve_device("cuda:1"), torch.device("cuda", 1))

    def test_seed_and_cache_only_touch_matching_backend(self):
        with fake_host(), mock.patch("torch.cuda.empty_cache") as cuda_cache:
            dev.empty_cache("cpu")
            dev.empty_cache("cuda")
            cuda_cache.assert_not_called()
        with fake_host(cuda=True), mock.patch("torch.cuda.manual_seed_all") as seed, mock.patch(
            "torch.cuda.empty_cache"
        ) as cache:
            dev.manual_seed_all(3)
            dev.empty_cache("cuda")
            seed.assert_called_once_with(3)
            cache.assert_called_once_with()


class LoaderDispatchTest(unittest.TestCase):
    def load(self, **kwargs):
        model = mock.MagicMock()
        model.config.mm_use_im_start_end = False
        model.config.mm_use_im_patch_token = False
        model.get_vision_tower.return_value.is_loaded = True
        with mock.patch.object(builder.LlavaLlamaAttForCausalLM, "from_pretrained", return_value=model) as load, mock.patch.object(
            builder.AutoTokenizer, "from_pretrained"
        ), mock.patch.object(builder, "resolve_device", wraps=dev.resolve_device):
            builder.load_pretrained_model("ckpt", None, "uninavid-vid", **kwargs)
        return load.call_args.kwargs, model

    def test_non_cuda_host_pins_resolved_device(self):
        with fake_host(xpu=True):
            kwargs, model = self.load()
        self.assertEqual(kwargs["device_map"], {"": torch.device("xpu", 0)})
        self.assertEqual(kwargs["torch_dtype"], torch.float16)
        model.get_vision_tower.return_value.to.assert_called_with(device=torch.device("xpu", 0), dtype=torch.float16)

    def test_cuda_host_keeps_auto_placement(self):
        with fake_host(cuda=True):
            kwargs, _ = self.load()
        self.assertEqual(kwargs["device_map"], "auto")

    def test_explicit_device_and_dtype_win(self):
        with fake_host(xpu=True):
            kwargs, model = self.load(device="cpu", dtype=torch.bfloat16)
        self.assertEqual(kwargs["device_map"], {"": torch.device("cpu")})
        self.assertEqual(kwargs["torch_dtype"], torch.bfloat16)
        model.get_vision_tower.return_value.to.assert_called_with(device=torch.device("cpu"), dtype=torch.bfloat16)


@unittest.skipUnless(HAS_XPU, "needs an XPU")
class XpuParityTest(unittest.TestCase):
    def test_tiny_eva_tower_matches_cpu(self):
        torch.manual_seed(0)
        model = VisionTransformer(
            img_size=28, patch_size=14, use_mean_pooling=False, embed_dim=64, depth=2, num_heads=4,
            mlp_ratio=4.0, qkv_bias=True,
        ).eval()
        for p in model.parameters():
            p.data = torch.randn_like(p) * 0.1
        images = torch.randn(2, 3, 28, 28)
        with torch.no_grad():
            ref = model(images)
            out = model.to("xpu")(images.to("xpu")).cpu()
        self.assertGreater(ref.std().item(), 1e-3)
        torch.testing.assert_close(out, ref, atol=2e-3, rtol=1e-3)


if __name__ == "__main__":
    unittest.main()
