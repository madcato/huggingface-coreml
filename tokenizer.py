from transformers import BlenderbotSmallTokenizer
from transformers import PreTrainedTokenizer
from typing import Dict, List, Optional, Tuple
import json
import os
import regex as re
from transformers import logging

##  "bos_token_id": 1,
# "decoder_start_token_id": 1,
#  "eos_token_id": 2,
#    "forced_eos_token_id": 2,
#  "pad_token_id": 0,
#   "unk_token_id": 3,
#
#
#

logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",
    "merges_file": "merges.txt",
    "tokenizer_config_file": "tokenizer_config.json",
}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "facebook/blenderbot_small-90M": "https://huggingface.co/facebook/blenderbot_small-90M/resolve/main/vocab.json"
    },
    "merges_file": {
        "facebook/blenderbot_small-90M": "https://huggingface.co/facebook/blenderbot_small-90M/resolve/main/merges.txt"
    },
    "tokenizer_config_file": {
        "facebook/blenderbot_small-90M": (
            "https://huggingface.co/facebook/blenderbot_small-90M/resolve/main/tokenizer_config.json"
        )
    },
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {"facebook/blenderbot_small-90M": 512}


def get_pairs(word):
    """
    Return set of symbol pairs in a word.

    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char

    pairs = set(pairs)
    return pairs

class BlenderbotSmallTokenizer2(PreTrainedTokenizer):
    """
    Constructs a Blenderbot-90M tokenizer based on BPE (Byte-Pair-Encoding)

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
    the superclass for more information regarding methods.

    Args:
        vocab_file (`str`):
            File containing the vocabulary.
        merges_file (`str`):
            Path to the merges file.
        bos_token (`str`, *optional*, defaults to `"__start__"`):
            The beginning of sentence token.
        eos_token (`str`, *optional*, defaults to `"__end__"`):
            The end of sentence token.
        unk_token (`str`, *optional*, defaults to `"__unk__"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        pad_token (`str`, *optional*, defaults to `"__pad__"`):
            The token used for padding, for example when batching sequences of different lengths.
        **kwargs
            Additional keyword arguments passed along to [`PreTrainedTokenizer`]
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file,
        merges_file,
        bos_token="__start__",
        eos_token="__end__",
        unk_token="__unk__",
        pad_token="__null__",
        **kwargs,
    ):
        super().__init__(unk_token=unk_token, bos_token=bos_token, eos_token=eos_token, pad_token=pad_token, **kwargs)

        with open(vocab_file, encoding="utf-8") as vocab_handle:
            self.encoder = json.load(vocab_handle)
        self.decoder = {v: k for k, v in self.encoder.items()}
        with open(merges_file, encoding="utf-8") as merges_handle:
            merges = merges_handle.read().split("\n")[1:-1]
        merges = [tuple(merge.split()) for merge in merges]
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {}

    @property
    def vocab_size(self) -> int:
        return len(self.encoder)

    def get_vocab(self) -> Dict:
        return dict(self.encoder, **self.added_tokens_encoder)

    def bpe(self, token: str) -> str:
        if token in self.cache:
            return self.cache[token]
        token = re.sub("([.,!?()])", r" \1", token)
        token = re.sub("(')", r" \1 ", token)
        token = re.sub(r"\s{2,}", " ", token)
        if "\n" in token:
            token = token.replace("\n", " __newln__")

        tokens = token.split(" ")
        words = []
        for token in tokens:
            if not len(token):
                continue

            token = token.lower()
            word = tuple(token)
            word = tuple(list(word[:-1]) + [word[-1] + "</w>"])
            print(word)
            pairs = get_pairs(word)
            print(pairs)
            
            if not pairs:
                words.append(token)
                continue

            while True:
                bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
                if bigram not in self.bpe_ranks:
                    break
                first, second = bigram
                new_word = []
                i = 0

                while i < len(word):
                    try:
                        j = word.index(first, i)
                        new_word.extend(word[i:j])
                        i = j
                    except ValueError:
                        new_word.extend(word[i:])
                        break

                    if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                        new_word.append(first + second)
                        i += 2
                    else:
                        new_word.append(word[i])
                        i += 1
                new_word = tuple(new_word)
                word = new_word
                if len(word) == 1:
                    break
                else:
                    pairs = get_pairs(word)
            word = "@@ ".join(word)
            word = word[:-4]

            self.cache[token] = word
            words.append(word)
        return " ".join(words)

    def _tokenize(self, text: str) -> List[str]:
        """Split a string into tokens using BPE."""
        split_tokens = []

        print("HOALHOAL")

        words = re.findall(r"\S+\n?", text)

        print(words)
        for token in words:
            split_tokens.extend(list(self.bpe(token).split(" ")))
        return split_tokens

    def _convert_token_to_id(self, token: str) -> int:
        """Converts a token to an id using the vocab."""
        token = token.lower()
        return self.encoder.get(token, self.encoder.get(self.unk_token))

    def _convert_id_to_token(self, index: int) -> str:
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.decoder.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """Converts a sequence of tokens in a single string."""
        out_string = " ".join(tokens).replace("@@ ", "").strip()
        return out_string

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )
        merge_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["merges_file"]
        )

        with open(vocab_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.encoder, indent=2, sort_keys=True, ensure_ascii=False) + "\n")

        index = 0
        with open(merge_file, "w", encoding="utf-8") as writer:
            writer.write("#version: 0.2\n")
            for bpe_tokens, token_index in sorted(self.bpe_ranks.items(), key=lambda kv: kv[1]):
                if index != token_index:
                    logger.warning(
                        f"Saving vocabulary to {merge_file}: BPE merge indices are not consecutive."
                        " Please check that the tokenizer is not corrupted!"
                    )
                    index = token_index
                writer.write(" ".join(bpe_tokens) + "\n")
                index += 1

        return vocab_file, merge_file



##################
model_ckpt = "facebook/blenderbot_small-90M"

tokenizer = BlenderbotSmallTokenizer2.from_pretrained(model_ckpt)

tokensIds = tokenizer("Hello world how are you?")["input_ids"]

print(tokensIds)
