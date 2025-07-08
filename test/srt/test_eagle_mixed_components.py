# Add to test/srt/test_eagle_mixed_components.py (new file)

import unittest
from python.sglang.srt.managers.schedule_batch import ScheduleBatch
from python.sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
from python.sglang.srt.speculative.eagle_utils import MixedBatchVerifyInput
from python.sglang.srt.layers.logits_processor import LogitsProcessorOutput
import torch

class TestMixedBatchComponents(unittest.TestCase):
    """Focused unit tests for mixed batch components."""
    
    def test_logits_splitting(self):
        """Test _split_mixed_logits functionality."""
        mixed_input = self._create_test_mixed_input()
        
        # Create combined logits (2 prefill + 1 decode = 3 total)
        combined_logits = LogitsProcessorOutput(
            next_token_logits=torch.randn(3, 1000),
            hidden_states=torch.randn(3, 512),
        )
        
        prefill_logits, decode_logits = mixed_input._split_mixed_logits(combined_logits)
        
        # Check prefill portion (first 2)
        self.assertEqual(prefill_logits.next_token_logits.shape[0], 2)
        self.assertEqual(prefill_logits.hidden_states.shape[0], 2)
        
        # Check decode portion (last 1)
        self.assertEqual(decode_logits.next_token_logits.shape[0], 1)
        self.assertEqual(decode_logits.hidden_states.shape[0], 1)
    
    def test_prefill_processing(self):
        """Test _process_prefill_results functionality."""
        mixed_input = self._create_test_mixed_input()
        
        prefill_logits = LogitsProcessorOutput(
            next_token_logits=torch.randn(2, 1000),
            hidden_states=torch.randn(2, 512),
        )
        
        # Mock batch
        batch = self._create_mock_batch()
        
        results = mixed_input._process_prefill_results(prefill_logits, batch)
        
        self.assertIn("next_tokens", results)
        self.assertIn("logits", results)
        self.assertEqual(len(results["next_tokens"]), 2)
    
    def test_tensor_device_consistency(self):
        """Test that all tensors stay on the correct device."""
        mixed_input = self._create_test_mixed_input()
        batch = self._create_mock_batch()
        
        # Verify all tensors are on the same device
        device = batch.device
        
        unified_ids = mixed_input._create_unified_input_ids(batch)
        self.assertEqual(unified_ids.device.type, device.split(':')[0])
        
        cache_loc = mixed_input._allocate_unified_cache(batch, page_size=1)
        self.assertEqual(cache_loc.device.type, device.split(':')[0])
    
    def _create_test_mixed_input(self):
        """Helper to create test MixedBatchVerifyInput."""
        return MixedBatchVerifyInput(
            # Standard EAGLE fields
            draft_token=torch.tensor([101, 102, 103]),
            custom_mask=torch.tensor([True, True, True]),
            positions=torch.tensor([0, 1, 2]),
            retrive_index=torch.tensor([[-1, 0, 1]]),
            retrive_next_token=torch.tensor([[-1, 1, 2]]),
            retrive_next_sibling=torch.tensor([[-1, -1, -1]]),
            retrive_cum_len=None,
            spec_steps=2,
            topk=4,
            draft_token_num=3,
            capture_hidden_mode=CaptureHiddenMode.FULL,
            seq_lens_sum=10,
            seq_lens_cpu=torch.tensor([5, 5]),
            # Mixed batch fields
            prefill_indices=[0, 1],
            decode_indices=[2],
            original_input_ids=torch.tensor([100, 101, 102, 103, 104]),
            prefill_token_count=4,
            decode_token_count=3,
        )
    
    def _create_mock_batch(self):
        """Helper to create mock batch."""
        
        batch = ScheduleBatch()
        batch.device = "cuda"
        batch.input_ids = torch.tensor([100, 101, 102, 103, 104])
        batch.reqs = [None, None, None]  # 3 mock requests
        return batch

if __name__ == "__main__":
    unittest.main()