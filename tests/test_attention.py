import unittest
import torch
import butane

class TestAttention(unittest.TestCase):

    def test_self_attention(self):
        self_att = butane.nn.SelfAttention(256, n_heads=1)
        self.assertIsNotNone(self_att(torch.randn(2, 3, 256)))

        spatial_self_att = butane.nn.SpatialSelfAttention(3, n_heads=1)
        self.assertIsNotNone(spatial_self_att(torch.randn(2, 3, 256, 256)))

    def test_cross_attention(self):
        cross_att = butane.nn.CrossAttention(256, n_heads=1)
        self.assertIsNotNone(cross_att(torch.randn(2, 3, 256), torch.randn(2, 3, 256)))

        spatial_cross_att = butane.nn.SpatialCrossAttention(3, n_heads=1)
        self.assertIsNotNone(spatial_cross_att(torch.randn(2, 3, 256, 256), torch.randn(2, 3, 256, 256)))

    def test_cross_attention_mismatched_q_kv_length(self):
        for n_heads in (1, 4):
            cross_att = butane.nn.CrossAttention(16, n_heads=n_heads)
            out = cross_att(torch.randn(2, 5, 16), torch.randn(2, 9, 16))
            self.assertEqual(out.shape, (2, 5, 16))

    def test_cross_attention_padding_mask_matches_truncation(self):
        torch.manual_seed(0)
        B, L_Q, L_KV, D = 2, 5, 9, 16
        x1 = torch.randn(B, L_Q, D)
        x2 = torch.randn(B, L_KV, D)
        mask = torch.ones(B, L_KV, dtype=torch.bool)
        mask[:, -3:] = False

        for n_heads in (1, 4):
            cross_att = butane.nn.CrossAttention(D, n_heads=n_heads).eval()
            with torch.no_grad():
                out_masked = cross_att(x1, x2, mask=mask)
                out_truncated = cross_att(x1, x2[:, : L_KV - 3])
            self.assertTrue(torch.allclose(out_masked, out_truncated, atol=1e-5))

    def test_self_attention_padding_mask(self):
        self_att = butane.nn.SelfAttention(16, n_heads=4)
        mask = torch.ones(2, 7, dtype=torch.bool)
        mask[:, -2:] = False
        out = self_att(torch.randn(2, 7, 16), mask=mask)
        self.assertEqual(out.shape, (2, 7, 16))

    def test_spatial_cross_attention_mismatched_shapes_with_mask(self):
        spatial_cross_att = butane.nn.SpatialCrossAttention(16, n_heads=2)
        mask = torch.ones(2, 9, dtype=torch.bool)
        mask[:, -2:] = False
        out = spatial_cross_att(torch.randn(2, 16, 4, 4), torch.randn(2, 16, 3, 3), mask=mask)
        self.assertEqual(out.shape, (2, 16, 4, 4))

if __name__ == '__main__':
    unittest.main()
