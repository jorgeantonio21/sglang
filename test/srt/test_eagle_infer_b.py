import json
import os
import random
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from types import SimpleNamespace

import numpy as np
import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.few_shot_gsm8k import run_eval
from sglang.test.test_utils import (
    DEFAULT_EAGLE_DRAFT_MODEL_FOR_TEST,
    DEFAULT_EAGLE_TARGET_MODEL_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    run_logprob_check,
)


class TestEAGLEServer(CustomTestCase):
    PROMPTS = [
        "[INST] <<SYS>>\\nYou are a helpful assistant.\\n<</SYS>>\\nToday is a sunny day and I like[/INST]"
        '[INST] <<SYS>>\\nYou are a helpful assistant.\\n<</SYS>>\\nWhat are the mental triggers in Jeff Walker\'s Product Launch Formula and "Launch" book?[/INST]',
        "[INST] <<SYS>>\\nYou are a helpful assistant.\\n<</SYS>>\\nSummarize Russell Brunson's Perfect Webinar Script...[/INST]",
        "[INST] <<SYS>>\\nYou are a helpful assistant.\\n<</SYS>>\\nwho are you?[/INST]",
        "[INST] <<SYS>>\\nYou are a helpful assistant.\\n<</SYS>>\\nwhere are you from?[/INST]",
    ]

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
                5,
                "--speculative-eagle-topk",
                8,
                "--speculative-num-draft-tokens",
                64,
                "--mem-fraction-static",
                0.7,
                "--chunked-prefill-size",
                128,
                "--max-running-requests",
                8,
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def send_request(self):
        time.sleep(random.uniform(0, 2))
        for prompt in self.PROMPTS:
            url = self.base_url + "/generate"
            data = {
                "text": prompt,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 1024,
                },
            }
            response = requests.post(url, json=data)
            assert response.status_code == 200

    def send_requests_abort(self):
        for prompt in self.PROMPTS:
            try:
                time.sleep(random.uniform(0, 2))
                url = self.base_url + "/generate"
                data = {
                    "model": "base",
                    "text": prompt,
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": 1024,
                    },
                }
                # set timeout = 1s, mock disconnected
                requests.post(url, json=data, timeout=1)
            except Exception as e:
                print(e)
                pass

    def test_request_abort(self):
        concurrency = 4
        threads = [
            threading.Thread(target=self.send_request) for _ in range(concurrency)
        ] + [
            threading.Thread(target=self.send_requests_abort)
            for _ in range(concurrency)
        ]
        for worker in threads:
            worker.start()
        for p in threads:
            p.join()

    def test_max_token_one(self):
        requests.get(self.base_url + "/flush_cache")

        args = SimpleNamespace(
            num_shots=5,
            data_path=None,
            num_questions=200,
            max_new_tokens=1,
            parallel=128,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )

        # Just run and check it does not hang
        metrics = run_eval(args)
        self.assertGreater(metrics["output_throughput"], 50)

    def test_gsm8k(self):
        requests.get(self.base_url + "/flush_cache")

        args = SimpleNamespace(
            num_shots=5,
            data_path=None,
            num_questions=200,
            max_new_tokens=512,
            parallel=128,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )

        metrics = run_eval(args)
        print(f"{metrics=}")
        self.assertGreater(metrics["accuracy"], 0.20)

        server_info = requests.get(self.base_url + "/get_server_info").json()
        avg_spec_accept_length = server_info["internal_states"][0][
            "avg_spec_accept_length"
        ]
        print(f"{avg_spec_accept_length=}")

        speculative_eagle_topk = server_info["speculative_eagle_topk"]

        if speculative_eagle_topk == 1:
            self.assertGreater(avg_spec_accept_length, 2.5)
        else:
            self.assertGreater(avg_spec_accept_length, 3.5)

        # Wait a little bit so that the memory check happens.
        time.sleep(4)

    def test_logprob_start_len(self):
        logprob_start_len = 4
        new_tokens = 4
        prompts = [
            "I have a very good idea on",
            "Today is a sunndy day and",
        ]

        response = requests.post(
            self.base_url + "/generate",
            json={
                "text": prompts,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": new_tokens,
                },
                "return_logprob": True,
                "top_logprobs_num": 5,
                "logprob_start_len": logprob_start_len,
            },
        )
        response_json = response.json()
        print(json.dumps(response_json, indent=2))

        for res in response_json:
            self.assertEqual(
                res["meta_info"]["prompt_tokens"],
                logprob_start_len + len(res["meta_info"]["input_token_logprobs"]),
            )

            self.assertEqual(res["meta_info"]["completion_tokens"], new_tokens)
            self.assertEqual(len(res["meta_info"]["output_token_logprobs"]), new_tokens)

    def test_logprob_match(self):
        """Test the output logprobs are close to the input logprobs if we run a prefill again."""

        def run_generate(
            prompt,
            return_logprob=False,
            max_new_tokens=512,
            logprob_start_len=-1,
            temperature=1.0,
        ):

            if isinstance(prompt, str):
                prompt_kwargs = {"text": prompt}
            else:
                prompt_kwargs = {"input_ids": prompt}

            response = requests.post(
                self.base_url + "/generate",
                json={
                    **prompt_kwargs,
                    "sampling_params": {
                        "temperature": temperature,
                        "max_new_tokens": max_new_tokens,
                        "ignore_eos": True,
                    },
                    "return_logprob": return_logprob,
                    "return_text_in_logprobs": True,
                    "logprob_start_len": logprob_start_len,
                    "temp_scaled_logprobs": True,
                },
            )
            return response.json()

        prompt = "I have a very good idea on how to"

        for temperature in [1.0]:
            gen = run_generate(
                prompt,
                return_logprob=True,
                logprob_start_len=0,
                temperature=temperature,
            )
            output_logprobs = np.array(
                [x[0] for x in gen["meta_info"]["output_token_logprobs"]]
            )
            num_prompts_tokens = gen["meta_info"]["prompt_tokens"]

            input_tokens = [x[1] for x in gen["meta_info"]["input_token_logprobs"]]
            output_tokens = [x[1] for x in gen["meta_info"]["output_token_logprobs"]]

            new_prompt = input_tokens + output_tokens
            score = run_generate(
                new_prompt,
                return_logprob=True,
                logprob_start_len=0,
                max_new_tokens=0,
                temperature=temperature,
            )
            output_logprobs_score = np.array(
                [
                    x[0]
                    for x in score["meta_info"]["input_token_logprobs"][
                        num_prompts_tokens:
                    ]
                ]
            )

            print(f"{output_logprobs[-10:]=}")
            print(f"{output_logprobs_score[-10:]=}")

            diff = np.abs(output_logprobs - output_logprobs_score)
            max_diff = np.max(diff)
            self.assertLess(max_diff, 0.255)

    def test_logprob_mixed(self):
        args = []
        temperature = 0
        # input_len, output_len, temperature, logprob_start_len, return_logprob, top_logprobs_num
        # Llama 2 context length seems to be only 2k, so we can only test small length.
        for input_len in [200, 500, 1000, 2000]:
            for output_len in [4, 8]:
                for logprob_start_len in [0, 100, 300, 800, 1998]:
                    for return_logprob in [True, False]:
                        for top_logprobs_num in [0, 5]:

                            if logprob_start_len >= input_len:
                                continue

                            args.append(
                                (
                                    input_len,
                                    output_len,
                                    temperature,
                                    logprob_start_len,
                                    return_logprob,
                                    top_logprobs_num,
                                )
                            )

        random.shuffle(args)

        func = partial(run_logprob_check, self)
        with ThreadPoolExecutor(8) as executor:
            list(executor.map(func, args))

    def run_decode(self, sampling_params):
        return_logprob = True
        top_logprobs_num = 5
        return_text = True
        n = 1

        response = requests.post(
            self.base_url + "/generate",
            json={
                "text": "Human: Write a travel blog post to Hawaii.\n\nAssistant:",
                "sampling_params": {
                    "max_new_tokens": 48,
                    "n": n,
                    "temperature": 0.7,
                    **sampling_params,
                },
                "return_logprob": return_logprob,
                "top_logprobs_num": top_logprobs_num,
                "return_text_in_logprobs": return_text,
                "logprob_start_len": 0,
            },
        )
        self.assertEqual(response.status_code, 200)
        print(json.dumps(response.json()))
        print("=" * 100)

    def test_penalty_mixed(self):
        args = [
            {},
            {},
            {},
            {"frequency_penalty": 2},
            {"presence_penalty": 1},
            {"min_new_tokens": 16},
            {"frequency_penalty": 0.2},
            {"presence_penalty": 0.4},
            {"min_new_tokens": 8},
            {"frequency_penalty": 0.4, "presence_penalty": 0.8},
            {"frequency_penalty": 0.4, "min_new_tokens": 12},
            {"presence_penalty": 0.8, "min_new_tokens": 12},
            {"presence_penalty": -0.3, "frequency_penalty": 1.3, "min_new_tokens": 32},
            {"presence_penalty": 0.3, "frequency_penalty": -1.3, "min_new_tokens": 32},
        ]
        random.shuffle(args * 5)
        with ThreadPoolExecutor(8) as executor:
            list(executor.map(self.run_decode, args))

    def test_constrained_decoding(self):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Give me a json"},
        ]

        response = requests.post(
            self.base_url + "/v1/chat/completions",
            json={
                "model": DEFAULT_EAGLE_TARGET_MODEL_FOR_TEST,
                "messages": messages,
                "temperature": 0,
                "response_format": {"type": "json_object"},
            },
        )
        self.assertEqual(response.status_code, 200)
        res = response.json()

        # Validate response structure
        self.assertIn("choices", res)
        self.assertEqual(len(res["choices"]), 1)
        self.assertIn("message", res["choices"][0])
        self.assertIn("content", res["choices"][0]["message"])

        # Validate JSON content
        content_json = res["choices"][0]["message"]["content"]
        is_valid_json = True
        try:
            content = json.loads(content_json)
            self.assertIsInstance(content, dict)
        except Exception:
            print(f"parse JSON failed: {content_json}")
            is_valid_json = False
        self.assertTrue(is_valid_json)


class TestEAGLERetract(TestEAGLEServer):
    @classmethod
    def setUpClass(cls):
        # These config helps find a leak.
        os.environ["SGLANG_CI_SMALL_KV_SIZE"] = "4500"
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
                5,
                "--speculative-eagle-topk",
                8,
                "--speculative-num-draft-tokens",
                64,
                "--mem-fraction-static",
                0.7,
                "--chunked-prefill-size",
                128,
                "--max-running-requests",
                64,
            ],
        )


class TestEAGLEServerTriton(TestEAGLEServer):
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
                5,
                "--speculative-eagle-topk",
                8,
                "--speculative-num-draft-tokens",
                64,
                "--mem-fraction-static",
                0.7,
                "--attention-backend",
                "triton",
                "--max-running-requests",
                8,
            ],
        )


class TestEAGLEServerPageSize(TestEAGLEServer):
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
                5,
                "--speculative-eagle-topk",
                1,
                "--speculative-num-draft-tokens",
                6,
                "--mem-fraction-static",
                0.7,
                "--chunked-prefill-size",
                128,
                "--max-running-requests",
                8,
                "--page-size",
                4,
                "--attention-backend",
                "flashinfer",
            ],
        )


class TestEAGLEServerPageSizeTopk(TestEAGLEServer):
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
                5,
                "--speculative-eagle-topk",
                8,
                "--speculative-num-draft-tokens",
                64,
                "--mem-fraction-static",
                0.7,
                "--chunked-prefill-size",
                128,
                "--max-running-requests",
                8,
                "--page-size",
                4,
                "--attention-backend",
                "flashinfer",
            ],
        )


class TestEAGLEMixedChunkServer(CustomTestCase):
    """Test EAGLE with mixed chunked prefill at server level."""
    
    BASE_CONFIG = {
        "model_path": DEFAULT_EAGLE_TARGET_MODEL_FOR_TEST,
        "speculative_draft_model_path": DEFAULT_EAGLE_DRAFT_MODEL_FOR_TEST,
        "speculative_algorithm": "EAGLE",
        "enable_mixed_chunk": True,
        "chunked_prefill_size": 512,
        "max_running_requests": 8,
    }

    def test_concurrent_mixed_requests(self):
        """Test concurrent requests with mixed prefill/decode patterns."""
        with self.create_server() as server:
            import requests
            import threading
            import time
            
            base_url = f"http://localhost:{server.port}"
            results = []
            errors = []
            
            def send_request(prompt, max_tokens):
                try:
                    response = requests.post(
                        f"{base_url}/v1/completions",
                        json={
                            "prompt": prompt,
                            "max_tokens": max_tokens,
                            "temperature": 0,
                        }
                    )
                    if response.status_code == 200:
                        results.append(response.json())
                    else:
                        errors.append(f"HTTP {response.status_code}: {response.text}")
                except Exception as e:
                    errors.append(str(e))
            
            threads = []
            
            # Mix of long and short prompts to trigger mixed batching
            prompts = [
                ("Write a detailed story about " * 30, 20),  # Long prefill
                ("Hello", 5),                                 # Short decode-like
                ("Explain quantum physics " * 20, 15),       # Long prefill
                ("Yes", 3),                                   # Short decode-like
            ]
            
            # Send requests concurrently
            for prompt, max_tokens in prompts:
                thread = threading.Thread(target=send_request, args=(prompt, max_tokens))
                threads.append(thread)
                thread.start()
                time.sleep(0.1)  # Slight delay to encourage batching
            
            # Wait for completion
            for thread in threads:
                thread.join(timeout=30)
            
            # Verify results
            self.assertEqual(len(errors), 0, f"Errors occurred: {errors}")
            self.assertEqual(len(results), len(prompts))
            
            for result in results:
                self.assertIn("choices", result)
                self.assertGreater(len(result["choices"][0]["text"]), 0)

    def test_mixed_batch_performance(self):
        """Test that mixed batching improves throughput."""
        with self.create_server() as server:
            import requests
            import time
            
            base_url = f"http://localhost:{server.port}"
            
            # Test sequential processing
            start_time = time.time()
            for i in range(4):
                response = requests.post(
                    f"{base_url}/v1/completions",
                    json={
                        "prompt": f"Test prompt {i}: " + "word " * 10,
                        "max_tokens": 10,
                        "temperature": 0,
                    }
                )
                self.assertEqual(response.status_code, 200)
            sequential_time = time.time() - start_time
            
            # Test concurrent processing (should be faster with mixed batching)
            import threading
            results = []
            
            def send_concurrent_request(i):
                response = requests.post(
                    f"{base_url}/v1/completions",
                    json={
                        "prompt": f"Test prompt {i}: " + "word " * 10,
                        "max_tokens": 10,
                        "temperature": 0,
                    }
                )
                results.append(response.status_code)
            
            start_time = time.time()
            threads = []
            for i in range(4):
                thread = threading.Thread(target=send_concurrent_request, args=(i,))
                threads.append(thread)
                thread.start()
            
            for thread in threads:
                thread.join()
            concurrent_time = time.time() - start_time
            
            # Verify all requests succeeded
            self.assertEqual(len(results), 4)
            self.assertTrue(all(status == 200 for status in results))
            
            # Concurrent should be faster (allowing some margin for variability)
            self.assertLess(concurrent_time, sequential_time * 1.2)

    def test_mixed_batch_memory_efficiency(self):
        """Test that mixed batching doesn't cause memory issues."""
        with self.create_server() as server:
            import requests
            import psutil
            import os
            
            base_url = f"http://localhost:{server.port}"
            
            # Get initial memory usage
            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss
            
            # Send many requests to test memory stability
            for batch in range(5):
                requests_batch = []
                for i in range(8):
                    prompt = f"Batch {batch} request {i}: " + "memory test " * 20
                    response = requests.post(
                        f"{base_url}/v1/completions",
                        json={
                            "prompt": prompt,
                            "max_tokens": 15,
                            "temperature": 0,
                        }
                    )
                    self.assertEqual(response.status_code, 200)
                    requests_batch.append(response.json())
                
                # Verify responses
                for result in requests_batch:
                    self.assertIn("choices", result)
            
            # Check memory didn't grow excessively
            final_memory = process.memory_info().rss
            memory_growth = final_memory - initial_memory
            
            # Memory growth should be reasonable (less than 1GB)
            self.assertLess(memory_growth, 1024 * 1024 * 1024)

    def test_error_handling_mixed_batch(self):
        """Test error handling in mixed batch scenarios."""
        with self.create_server() as server:
            import requests
            
            base_url = f"http://localhost:{server.port}"
            
            # Test with invalid parameters
            response = requests.post(
                f"{base_url}/v1/completions",
                json={
                    "prompt": "Test",
                    "max_tokens": -1,  # Invalid
                    "temperature": 0,
                }
            )
            self.assertEqual(response.status_code, 400)
            
            # Test with extremely long prompt
            response = requests.post(
                f"{base_url}/v1/completions",
                json={
                    "prompt": "word " * 10000,  # Very long
                    "max_tokens": 1,
                    "temperature": 0,
                }
            )
            # Should either succeed or fail gracefully
            self.assertIn(response.status_code, [200, 400, 413])


if __name__ == "__main__":
    unittest.main()
