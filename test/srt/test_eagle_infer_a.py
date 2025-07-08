import unittest

import requests
from python.sglang.srt.layers.logits_processor import LogitsProcessorOutput
from python.sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from python.sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardMode
from python.sglang.srt.sampling.sampling_params import SamplingParams
from python.sglang.srt.speculative.eagle_utils import EagleVerifyInput, MixedBatchVerifyInput
import torch

import sglang as sgl
from sglang.srt.hf_transformers_utils import get_tokenizer
from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_EAGLE_DRAFT_MODEL_FOR_TEST,
    DEFAULT_EAGLE_TARGET_MODEL_FOR_TEST,
    DEFAULT_MODEL_NAME_FOR_TEST_MLA,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
)

torch_dtype = torch.float16
prefill_tolerance = 5e-2
decode_tolerance: float = 5e-2


class TestEAGLEEngine(CustomTestCase):
    BASE_CONFIG = {
        "model_path": DEFAULT_EAGLE_TARGET_MODEL_FOR_TEST,
        "speculative_draft_model_path": DEFAULT_EAGLE_DRAFT_MODEL_FOR_TEST,
        "speculative_algorithm": "EAGLE",
        "speculative_num_steps": 5,
        "speculative_eagle_topk": 4,
        "speculative_num_draft_tokens": 8,
        "mem_fraction_static": 0.7,
        "cuda_graph_max_bs": 5,
    }
    NUM_CONFIGS = 2

    def setUp(self):
        self.prompt = "Today is a sunny day and I like"
        self.sampling_params = {"temperature": 0, "max_new_tokens": 8}

        ref_engine = sgl.Engine(
            model_path=self.BASE_CONFIG["model_path"], cuda_graph_max_bs=1
        )
        self.ref_output = ref_engine.generate(self.prompt, self.sampling_params)["text"]
        ref_engine.shutdown()

    def test_correctness(self):
        configs = [
            # Basic config
            self.BASE_CONFIG,
            # Chunked prefill
            {**self.BASE_CONFIG, "chunked_prefill_size": 4},
        ]

        for i, config in enumerate(configs[: self.NUM_CONFIGS]):
            with self.subTest(i=i):
                print(f"{config=}")
                engine = sgl.Engine(**config, log_level="info", decode_log_interval=10)
                try:
                    self._test_single_generation(engine)
                    self._test_batch_generation(engine)
                    self._test_eos_token(engine)
                    self._test_acc_length(engine)
                finally:
                    engine.shutdown()
                print("=" * 100)

    def _test_single_generation(self, engine):
        output = engine.generate(self.prompt, self.sampling_params)["text"]
        print(f"{output=}, {self.ref_output=}")
        self.assertEqual(output, self.ref_output)

    def _test_batch_generation(self, engine):
        prompts = [
            "Hello, my name is",
            "The president of the United States is",
            "The capital of France is",
            "The future of AI is",
        ]
        params = {"temperature": 0, "max_new_tokens": 50}

        outputs = engine.generate(prompts, params)
        for prompt, output in zip(prompts, outputs):
            print(f"Prompt: {prompt}")
            print(f"Generated: {output['text']}")
            print("-" * 40)

        print(f"{engine.get_server_info()=}")

        avg_spec_accept_length = engine.get_server_info()["internal_states"][0][
            "avg_spec_accept_length"
        ]
        print(f"{avg_spec_accept_length=}")
        self.assertGreater(avg_spec_accept_length, 1.9)

    def _test_eos_token(self, engine):
        prompt = "[INST] <<SYS>>\nYou are a helpful assistant.\n<</SYS>>\nToday is a sunny day and I like [/INST]"
        params = {
            "temperature": 0.1,
            "max_new_tokens": 1024,
            "skip_special_tokens": False,
        }

        tokenizer = get_tokenizer(DEFAULT_EAGLE_TARGET_MODEL_FOR_TEST)
        output = engine.generate(prompt, params)["text"]
        print(f"{output=}")

        tokens = tokenizer.encode(output, truncation=False)
        self.assertNotIn(tokenizer.eos_token_id, tokens)

    def _test_acc_length(self, engine):
        prompt = [
            "Human: Give me a fully functional FastAPI server. Show the python code.\n\nAssistant:",
        ] * 5  # test batched generation
        sampling_params = {"temperature": 0, "max_new_tokens": 512}
        output = engine.generate(prompt, sampling_params)
        output = output[0]

        if "spec_verify_ct" in output["meta_info"]:
            acc_length = (
                output["meta_info"]["completion_tokens"]
                / output["meta_info"]["spec_verify_ct"]
            )
        else:
            acc_length = 1.0

        speed = (
            output["meta_info"]["completion_tokens"]
            / output["meta_info"]["e2e_latency"]
        )
        print(f"{acc_length=:.4f}, {speed=}")

        if engine.server_args.model_path == DEFAULT_EAGLE_TARGET_MODEL_FOR_TEST:
            self.assertGreater(acc_length, 3.6)
        else:
            self.assertGreater(acc_length, 2.5)


class TestEAGLEEngineTokenMap(TestEAGLEEngine):
    BASE_CONFIG = {
        "model_path": "meta-llama/Meta-Llama-3-8B-Instruct",
        "speculative_draft_model_path": "lmsys/sglang-EAGLE-LLaMA3-Instruct-8B",
        "speculative_algorithm": "EAGLE",
        "speculative_num_steps": 5,
        "speculative_eagle_topk": 4,
        "speculative_num_draft_tokens": 8,
        "speculative_token_map": "thunlp/LLaMA3-Instruct-8B-FR-Spec/freq_32768.pt",
        "mem_fraction_static": 0.7,
        "cuda_graph_max_bs": 5,
        "dtype": "float16",
    }
    NUM_CONFIGS = 1


class TestEAGLE3Engine(TestEAGLEEngine):
    BASE_CONFIG = {
        "model_path": "meta-llama/Llama-3.1-8B-Instruct",
        "speculative_draft_model_path": "jamesliu1/sglang-EAGLE3-Llama-3.1-Instruct-8B",
        "speculative_algorithm": "EAGLE3",
        "speculative_num_steps": 5,
        "speculative_eagle_topk": 16,
        "speculative_num_draft_tokens": 64,
        "mem_fraction_static": 0.7,
        "cuda_graph_max_bs": 5,
        "dtype": "float16",
    }
    NUM_CONFIGS = 1


@unittest.skipIf(is_in_ci(), "To reduce the CI execution time.")
class TestEAGLEDraftExtend(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            DEFAULT_EAGLE_TARGET_MODEL_FOR_TEST,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--speculative-algorithm",
                "EAGLE",
                "--speculative-draft-model-path",
                DEFAULT_EAGLE_DRAFT_MODEL_FOR_TEST,
                "--speculative-num-steps",
                1,
                "--speculative-eagle-topk",
                1,
                "--speculative-num-draft-tokens",
                2,
                "--max-running-requests",
                4,
                "--attention-backend",
                "fa3",
            ],
        )
        cls.accept_len_threshold = 1.50

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_one_batch_accept_length(self):
        resp = requests.get(self.base_url + "/flush_cache")
        self.assertEqual(resp.status_code, 200)

        prompts = [
            "Hello, my name is",
            "The president of the United States is",
            "The capital of France is",
            "The future of AI is",
        ]
        url = self.base_url + "/generate"
        data = {
            "text": prompts,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": 512,
            },
        }
        response = requests.post(url, json=data)
        self.assertEqual(response.status_code, 200)
        outputs = response.json()
        for i in range(len(prompts)):
            output = outputs[i]
            if "spec_verify_ct" in output["meta_info"]:
                acc_length = (
                    output["meta_info"]["completion_tokens"]
                    / output["meta_info"]["spec_verify_ct"]
                )
            else:
                acc_length = 1.0

            print(f"{acc_length=}")
            self.assertGreater(acc_length, self.accept_len_threshold)


class TestEAGLEDraftExtendFlashinfer(TestEAGLEDraftExtend):
    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            DEFAULT_EAGLE_TARGET_MODEL_FOR_TEST,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--speculative-algorithm",
                "EAGLE",
                "--speculative-draft-model-path",
                DEFAULT_EAGLE_DRAFT_MODEL_FOR_TEST,
                "--speculative-num-steps",
                1,
                "--speculative-eagle-topk",
                1,
                "--speculative-num-draft-tokens",
                2,
                "--max-running-requests",
                4,
                "--attention-backend",
                "flashinfer",
            ],
        )
        cls.accept_len_threshold = 1.50


@unittest.skipIf(is_in_ci(), "To reduce the CI execution time.")
class TestEAGLEDraftExtendTriton(TestEAGLEDraftExtend):
    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            DEFAULT_EAGLE_TARGET_MODEL_FOR_TEST,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--speculative-algorithm",
                "EAGLE",
                "--speculative-draft-model-path",
                DEFAULT_EAGLE_DRAFT_MODEL_FOR_TEST,
                "--speculative-num-steps",
                1,
                "--speculative-eagle-topk",
                1,
                "--speculative-num-draft-tokens",
                2,
                "--max-running-requests",
                4,
                "--attention-backend",
                "triton",
            ],
        )
        cls.accept_len_threshold = 1.50


@unittest.skipIf(is_in_ci(), "To reduce the CI execution time.")
class TestEAGLEDraftExtendFlashinferMLA(TestEAGLEDraftExtend):
    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            DEFAULT_MODEL_NAME_FOR_TEST_MLA,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--speculative-algorithm",
                "EAGLE",
                "--speculative-num-steps",
                1,
                "--speculative-eagle-topk",
                1,
                "--speculative-num-draft-tokens",
                2,
                "--max-running-requests",
                4,
                "--attention-backend",
                "flashinfer",
            ],
        )
        cls.accept_len_threshold = 1.85


class TestEAGLEMixedChunkEngine(CustomTestCase):
    """Test EAGLE with mixed chunked prefill enabled."""
    
    BASE_CONFIG = {
        "model_path": DEFAULT_EAGLE_TARGET_MODEL_FOR_TEST,
        "speculative_draft_model_path": DEFAULT_EAGLE_DRAFT_MODEL_FOR_TEST,
        "speculative_algorithm": "EAGLE",
        "speculative_num_steps": 3,
        "speculative_eagle_topk": 4,
        "speculative_num_draft_tokens": 16,
        "enable_mixed_chunk": True,
        "chunked_prefill_size": 512,
        "max_running_requests": 8,
    }

    def test_mixed_batch_creation(self):
        """Test creating mixed batches with both prefill and decode requests."""
        with self.create_engine() as engine:
            # Create mixed requests: some prefill, some decode
            requests = []
            
            # Prefill requests (longer input)
            for i in range(2):
                requests.append({
                    "text": "Write a story about space exploration. " * 20,  # Long input
                    "sampling_params": {"max_new_tokens": 10, "temperature": 0},
                })
            
            # Simulate decode requests by sending shorter follow-ups
            for i in range(2):
                requests.append({
                    "text": "Continue: The astronaut",  # Short input (decode-like)
                    "sampling_params": {"max_new_tokens": 5, "temperature": 0},
                })
            
            outputs = []
            for req in requests:
                outputs.append(engine.generate(**req))
            
            # Verify all requests completed successfully
            for output in outputs:
                self.assertIsNotNone(output.text)
                self.assertGreater(len(output.text), 0)

    def test_mixed_batch_verify_input_creation(self):
        """Test MixedBatchVerifyInput creation and methods."""
        
        # Create mock batch
        mock_reqs = self._create_mock_requests(prefill_count=2, decode_count=2)
        batch = self._create_mock_batch(mock_reqs)
        
        # Create mock EAGLE input for decode requests
        decode_eagle_input = self._create_mock_eagle_input()
        
        prefill_indices = [0, 1]
        decode_indices = [2, 3]
        
        # Test creation
        mixed_input = MixedBatchVerifyInput.create_from_mixed_batch(
            batch, decode_eagle_input, prefill_indices, decode_indices
        )
        
        self.assertEqual(mixed_input.prefill_indices, prefill_indices)
        self.assertEqual(mixed_input.decode_indices, decode_indices)
        self.assertIsNotNone(mixed_input.draft_token)

    def test_mixed_batch_tensor_creation(self):
        """Test unified input_ids and cache allocation."""
        
        mock_reqs = self._create_mock_requests(prefill_count=1, decode_count=1)
        batch = self._create_mock_batch(mock_reqs)
        
        mixed_input = MixedBatchVerifyInput.create_from_mixed_batch(
            batch, self._create_mock_eagle_input(), [0], [1]
        )
        
        # Test unified input_ids creation
        unified_ids = mixed_input._create_unified_input_ids(batch)
        self.assertIsInstance(unified_ids, torch.Tensor)
        self.assertGreater(len(unified_ids), 0)
        
        # Test cache allocation
        unified_cache = mixed_input._allocate_unified_cache(batch, page_size=1)
        self.assertIsInstance(unified_cache, torch.Tensor)

    def test_mixed_batch_verification_flow(self):
        """Test the complete verification flow for mixed batches."""
        
        mock_reqs = self._create_mock_requests(prefill_count=1, decode_count=1)
        batch = self._create_mock_batch(mock_reqs)
        
        mixed_input = MixedBatchVerifyInput.create_from_mixed_batch(
            batch, self._create_mock_eagle_input(), [0], [1]
        )
        
        # Create mock logits output
        logits_output = LogitsProcessorOutput(
            next_token_logits=torch.randn(2, 1000),  # 2 requests, vocab_size=1000
            hidden_states=torch.randn(2, 512),       # hidden_size=512
        )
        
        # Test verification
        verify_output = mixed_input.verify(
            batch, logits_output, 
            self._create_mock_allocator(), 
            page_size=1
        )
        
        self.assertIsNotNone(verify_output.verified_id)
        self.assertIsNotNone(verify_output.logits_output)
        self.assertEqual(len(verify_output.verified_id), len(batch.reqs))

    def test_edge_cases(self):
        """Test edge cases like empty batches, single request types."""
        
        # Test prefill-only batch
        prefill_reqs = self._create_mock_requests(prefill_count=2, decode_count=0)
        prefill_batch = self._create_mock_batch(prefill_reqs)
        
        mixed_input_prefill = MixedBatchVerifyInput.create_from_mixed_batch(
            prefill_batch, None, [0, 1], []
        )
        self.assertEqual(len(mixed_input_prefill.decode_indices), 0)
        
        # Test decode-only batch
        decode_reqs = self._create_mock_requests(prefill_count=0, decode_count=2)
        decode_batch = self._create_mock_batch(decode_reqs)
        
        mixed_input_decode = MixedBatchVerifyInput.create_from_mixed_batch(
            decode_batch, self._create_mock_eagle_input(), [], [0, 1]
        )
        self.assertEqual(len(mixed_input_decode.prefill_indices), 0)

    # Helper methods
    def _create_mock_requests(self, prefill_count: int, decode_count: int):
        """Create mock requests for testing."""
        reqs = []
        
        # Prefill requests (extend_input_len > 1)
        for i in range(prefill_count):
            req = self._create_mock_req()
            req.extend_input_len = 50  # Long input
            req.origin_input_ids = list(range(100, 150))
            reqs.append(req)
        
        # Decode requests (extend_input_len = 1)
        for i in range(decode_count):
            req = self._create_mock_req()
            req.extend_input_len = 1  # Single token (decode)
            req.origin_input_ids = [200]
            reqs.append(req)
        
        return reqs
    
    def _create_mock_req(self):
        """Create a single mock request."""
        
        return Req(
            rid=f"test_{id(self)}",
            origin_input_text="test text",
            origin_input_ids=[100, 101, 102],
            sampling_params=SamplingParams(),
        )
    
    def _create_mock_batch(self, reqs):
        """Create mock ScheduleBatch."""
        
        batch = ScheduleBatch()
        batch.reqs = reqs
        batch.forward_mode = ForwardMode.MIXED
        batch.device = "cuda"
        batch.input_ids = torch.cat([torch.tensor(req.origin_input_ids) for req in reqs])
        batch.req_pool_indices = torch.arange(len(reqs))
        batch.seq_lens = torch.tensor([len(req.origin_input_ids) for req in reqs])
        return batch
    
    def _create_mock_eagle_input(self):
        """Create mock EagleVerifyInput for decode requests."""
        
        return EagleVerifyInput(
            draft_token=torch.tensor([1001, 1002, 1003]),
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
        )
    
    def _create_mock_allocator(self):
        """Create mock token allocator."""
        class MockAllocator:
            def free(self, tokens):
                pass
        return MockAllocator()


if __name__ == "__main__":
    unittest.main()
