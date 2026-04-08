import unittest

from SplitInfer.openai_api import (
    OpenAIRequestError,
    build_chat_response,
    build_stream_events,
    parse_chat_request,
    prepare_inference_input,
)
from SplitInfer.server_config import load_server_settings, parse_model_path_overrides


class OpenAIAPIHelpersTest(unittest.TestCase):
    def test_parse_chat_request_accepts_valid_payload(self):
        payload = {
            "model": "Llama-3-8B-Instruct",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": True,
            "max_tokens": 16,
        }
        parsed = parse_chat_request(payload)
        self.assertEqual(parsed.model, "Llama-3-8B-Instruct")
        self.assertTrue(parsed.stream)
        self.assertEqual(parsed.max_tokens, 16)

    def test_prepare_inference_input_rejects_images_for_text_model(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "look"},
                    {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}},
                ],
            }
        ]
        with self.assertRaises(OpenAIRequestError):
            prepare_inference_input(messages, supports_images=False)

    def test_prepare_inference_input_flattens_messages(self):
        messages = [
            {"role": "system", "content": "You are concise."},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        prepared = prepare_inference_input(messages, supports_images=False)
        self.assertIn("System: You are concise.", prepared.prompt_text)
        self.assertTrue(prepared.prompt_text.endswith("Assistant:"))
        self.assertIsNone(prepared.image_url)

    def test_build_chat_response_contains_usage(self):
        response = build_chat_response(
            model="Llama-3-8B-Instruct",
            content="hello",
            prompt_tokens=10,
            completion_tokens=2,
            created=123,
            completion_id="chatcmpl-test",
        )
        self.assertEqual(response["id"], "chatcmpl-test")
        self.assertEqual(response["usage"]["total_tokens"], 12)

    def test_build_stream_events_finish_with_done(self):
        events = build_stream_events("Llama-3-8B-Instruct", "hello world", completion_id="chatcmpl-test")
        self.assertTrue(events[0].startswith("data: "))
        self.assertEqual(events[-1], "data: [DONE]\n\n")

    def test_parse_model_path_overrides(self):
        overrides = parse_model_path_overrides([
            "Llama-3-8B-Instruct=/models/llama",
            "Qwen2-VL-7B-Instruct=/models/qwen",
        ])
        self.assertEqual(overrides["Llama-3-8B-Instruct"], "/models/llama")
        self.assertEqual(overrides["Qwen2-VL-7B-Instruct"], "/models/qwen")

    def test_load_server_settings_applies_cli_overrides(self):
        settings = load_server_settings(
            cli_overrides={
                "host": "127.0.0.1",
                "port": 9000,
                "weights_root": "/srv/weights",
                "model_paths": {"Llama-3-8B-Instruct": "/srv/llama"},
            }
        )
        self.assertEqual(settings.host, "127.0.0.1")
        self.assertEqual(settings.port, 9000)
        self.assertEqual(settings.weights_root, "/srv/weights")
        self.assertEqual(settings.model_paths["Llama-3-8B-Instruct"], "/srv/llama")


if __name__ == "__main__":
    unittest.main()
