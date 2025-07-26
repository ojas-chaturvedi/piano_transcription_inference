import os
import argparse
import torch
import time
import librosa
from piano_transcription_inference import PianoTranscription, sample_rate


def inference(args):
    """Inference template.

    Args:
        model_type: str
        audio_path: str
    """

    # Arguments & parameters
    audio_path = args.input
    output_midi_path = args.output
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load audio
    audio, _ = librosa.load(path=audio_path, sr=sample_rate, mono=True)

    # Transcriptor
    transcriptor = PianoTranscription(device=device, checkpoint_path="/anvil/scratch/x-ochaturvedi/piano_transcription_inference_data/note_F1=0.9677_pedal_F1=0.9186.pth")

    # Transcribe and write out to MIDI file
    transcribe_time = time.time()
    _ = transcriptor.transcribe(audio, output_midi_path)
    print("Transcribe time: {:.3f} s".format(time.time() - transcribe_time))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Piano transcription inference")
    parser.add_argument(
        "-i", "--input", type=str, required=True, help="Path to the input audio file"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Path to save the output MIDI file",
    )

    args = parser.parse_args()
    inference(args)
