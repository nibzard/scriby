package main

const cohereTranscribePythonScript = `#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path


def best_device(torch_module):
    if torch_module.cuda.is_available():
        return "cuda:0"
    mps = getattr(torch_module.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def write_transcript(path_str, text):
    path = Path(path_str)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Run Cohere Transcribe via transformers")
    parser.add_argument("--input", required=True, dest="input_path")
    parser.add_argument("--output", required=True, dest="output_path")
    parser.add_argument("--language", required=True)
    parser.add_argument("--model-id", required=True, dest="model_id")
    args = parser.parse_args()

    try:
        import torch
        from transformers import AutoProcessor
        from transformers.audio_utils import load_audio
    except Exception as exc:
        print(
            "Missing Python dependencies for the cohere engine. "
            "Install transformers, torch, huggingface_hub, soundfile, librosa, sentencepiece, and protobuf.",
            file=sys.stderr,
        )
        print(str(exc), file=sys.stderr)
        return 2

    processor = None
    model = None
    used_remote_code = False
    device = best_device(torch)

    try:
        from transformers import CohereAsrForConditionalGeneration

        processor = AutoProcessor.from_pretrained(args.model_id)
        try:
            model = CohereAsrForConditionalGeneration.from_pretrained(args.model_id, device_map="auto")
        except Exception:
            model = CohereAsrForConditionalGeneration.from_pretrained(args.model_id).to(device)
    except Exception:
        try:
            from transformers import AutoModelForSpeechSeq2Seq

            processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
            model = AutoModelForSpeechSeq2Seq.from_pretrained(args.model_id, trust_remote_code=True).to(device)
            model.eval()
            used_remote_code = True
        except Exception as exc:
            print(
                "Failed to load Cohere Transcribe from Hugging Face. "
                "Make sure you accepted the model's access conditions and that HF_TOKEN or local Hugging Face auth is configured.",
                file=sys.stderr,
            )
            print(str(exc), file=sys.stderr)
            return 3

    try:
        if used_remote_code and hasattr(model, "transcribe"):
            texts = model.transcribe(
                processor=processor,
                audio_files=[args.input_path],
                language=args.language,
            )
            text = texts[0] if isinstance(texts, list) else texts
        else:
            audio = load_audio(args.input_path, sampling_rate=16000)
            inputs = processor(audio, sampling_rate=16000, return_tensors="pt", language=args.language)
            audio_chunk_index = inputs.get("audio_chunk_index")
            inputs = inputs.to(model.device, dtype=model.dtype)
            outputs = model.generate(**inputs, max_new_tokens=256)
            if audio_chunk_index is not None:
                text = processor.decode(
                    outputs,
                    skip_special_tokens=True,
                    audio_chunk_index=audio_chunk_index,
                    language=args.language,
                )
            else:
                text = processor.decode(outputs, skip_special_tokens=True)
            if isinstance(text, list):
                text = text[0]

        write_transcript(args.output_path, text)
        return 0
    except Exception as exc:
        print("Cohere transcription failed.", file=sys.stderr)
        print(str(exc), file=sys.stderr)
        return 4


if __name__ == "__main__":
    sys.exit(main())
`
