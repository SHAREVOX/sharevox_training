import argparse

import numpy
from scipy.io import wavfile
import numpy as np
from pyopenjtalk import extract_fullcontext

from text import extract_phoneme_and_accents
from text import phoneme_to_id, accent_to_id
import onnxruntime


def preprocess_japanese(text):
    full_context_labels = extract_fullcontext(text)
    phonemes, accents = extract_phoneme_and_accents(full_context_labels)
    phonemes_seq = np.array([phoneme_to_id[phoneme] for phoneme in phonemes])
    accents_seq = np.array([accent_to_id[accent] for accent in accents])

    return phonemes_seq, accents_seq


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="raw text to synthesize, for single-sentence mode only",
    )
    parser.add_argument(
        "--speaker_id",
        type=int,
        default=0,
        help="speaker ID for multi-speaker synthesis, for single-sentence mode only",
    )
    args = parser.parse_args()

    # Check source texts
    assert args.text is not None

    # Preprocess texts
    ids = raw_texts = [args.text[:100]]
    speakers = np.array([args.speaker_id], dtype=np.int64)
    accents = None
    phonemes, accents = [np.array([out]).astype(dtype=np.int64) for out in preprocess_japanese(args.text)]

    variance_session = onnxruntime.InferenceSession("variance_model.onnx", providers=['CUDAExecutionProvider'])
    embedder_session = onnxruntime.InferenceSession("embedder_model.onnx", providers=['CUDAExecutionProvider'])
    decoder_session = onnxruntime.InferenceSession("decoder_model.onnx", providers=['CUDAExecutionProvider'])

    pitches, durations = variance_session.run(["pitches", "durations"], {
        "phonemes": phonemes,
        "accents": accents,
        "speakers": speakers,
    })

    feature_embedded = embedder_session.run(["feature_embedded"], {
        "phonemes": phonemes,
        "pitches": pitches[0].T,
        "speakers": speakers,
    })[0]

    durations = (durations[0].T * (48000 / 256)).astype(dtype=np.int64)
    length_regulated_tensor = numpy.repeat(feature_embedded, durations[0], axis=1)

    wav = decoder_session.run(["wav"], {
        "length_regulated_tensor": length_regulated_tensor,
    })[0]

    wavfile.write("test.wav", 48000, wav[0])
