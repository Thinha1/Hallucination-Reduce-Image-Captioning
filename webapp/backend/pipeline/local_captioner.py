from __future__ import annotations

import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from transformers import CLIPVisionModel

PAD_IDX = 0
START_IDX = 1
END_IDX = 2
IMAGE_SIZE = (224, 224)
EMBED_DIM = 512
FF_DIM = 512
NUM_HEADS = 8
NUM_LAYERS = 2
CLIP_VISION_SOURCE = "openai/clip-vit-base-patch16"


class CLIPEncoder(nn.Module):
    def __init__(self, embed_dim: int, clip_source: str = CLIP_VISION_SOURCE) -> None:
        super().__init__()
        # Only the architecture matters here: the checkpoint overwrites these weights.
        self.clip_vision = CLIPVisionModel.from_pretrained(clip_source, use_safetensors=True)
        for param in self.clip_vision.parameters():
            param.requires_grad = False
        self.fc = nn.Linear(768, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, images):
        outputs = self.clip_vision(pixel_values=images)
        return self.layer_norm(self.fc(outputs.last_hidden_state))


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim: int, dropout: float = 0.1, max_len: int = 100) -> None:
        super().__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        return self.dropout(x + self.pe[:, : x.size(1), :])


class ImageCaptioningTransformer(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int = NUM_LAYERS, clip_source: str = CLIP_VISION_SOURCE) -> None:
        super().__init__()
        self.encoder = CLIPEncoder(EMBED_DIM, clip_source)
        self.embedding = nn.Embedding(vocab_size, EMBED_DIM, padding_idx=PAD_IDX)
        self.pos_encoder = PositionalEncoding(EMBED_DIM)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=EMBED_DIM,
            nhead=NUM_HEADS,
            dim_feedforward=FF_DIM,
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(EMBED_DIM, vocab_size, bias=False)
        self.fc_out.weight = self.embedding.weight

    def generate_square_subsequent_mask(self, size: int):
        mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
        return mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))

    def decode_step(self, memory, sequence):
        tgt_embed = self.pos_encoder(self.embedding(sequence))
        tgt_mask = self.generate_square_subsequent_mask(sequence.size(1)).to(sequence.device)
        out = self.transformer_decoder(tgt=tgt_embed, memory=memory, tgt_mask=tgt_mask)
        return self.fc_out(out[:, -1, :])


def align_clip_vision_keys(state_dict: dict, expected_keys) -> dict:
    """transformers bản mới lưu CLIPVisionModel không có tiền tố "vision_model." (checkpoint train trên Kaggle),
    bản cũ thì có. Đổi tên khoá encoder.clip_vision.* theo đúng kiểu model đang chạy mong đợi."""
    expected = set(expected_keys)
    head, inner = "encoder.clip_vision.", "encoder.clip_vision.vision_model."
    out = {}
    for key, value in state_dict.items():
        if key not in expected and key.startswith(head):
            alt = key.replace(inner, head, 1) if key.startswith(inner) else key.replace(head, inner, 1)
            if alt in expected:
                key = alt
        out[key] = value
    return out


class LocalCheckpointCaptioner:
    def __init__(
        self,
        checkpoint_path: str | Path,
        vocab_path: str | Path,
        device: str | None = None,
        max_length: int = 25,
        clip_source: str = CLIP_VISION_SOURCE,
    ) -> None:
        self.clip_source = clip_source
        self.checkpoint_path = Path(checkpoint_path)
        self.vocab_path = Path(vocab_path)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.transform = T.Compose(
            [
                T.Resize(IMAGE_SIZE),
                T.CenterCrop(IMAGE_SIZE),
                T.ToTensor(),
                T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
            ]
        )
        self.vocab = json.loads(self.vocab_path.read_text(encoding="utf-8"))
        self.idx_to_word = {idx: word for word, idx in self.vocab.items()}
        self.model = self._load_model()

    def generate_diverse(
        self,
        image_path: str | Path,
        max_candidates: int,
        num_beams: int,
        num_beam_groups: int,
        diversity_penalty: float,
        logit_bias: torch.Tensor | None = None,
    ) -> list[tuple[str, float, list[float]]]:
        """`logit_bias` ([vocab]) is added to every decoding step's logits (logit blocking). The
        teacher-forcing pass for S_base below does not see it, as in the evaluation scripts."""
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        num_beams = self._valid_num_beams(num_beams, max_candidates, num_beam_groups)
        beams_per_group = max(1, num_beams // num_beam_groups)

        with torch.no_grad():
            memory = self.model.encoder(image_tensor)
            groups = [[([START_IDX], 0.0)] for _ in range(num_beam_groups)]
            for _ in range(self.max_length - 1):
                chosen_tokens = []
                all_finished = True
                for group_idx, beams in enumerate(groups):
                    new_beams = []
                    for seq, score in beams:
                        if seq[-1] == END_IDX:
                            new_beams.append((seq, score))
                            continue
                        all_finished = False
                        sequence = torch.tensor(seq, dtype=torch.long, device=self.device).unsqueeze(0)
                        logits = self.model.decode_step(memory, sequence).squeeze(0)
                        if logit_bias is not None:
                            logits = logits + logit_bias
                        for token_id in chosen_tokens:
                            logits[token_id] -= diversity_penalty
                        log_probs = F.log_softmax(logits, dim=-1)
                        top_scores, top_ids = log_probs.topk(beams_per_group)
                        for token_score, token_id in zip(top_scores.tolist(), top_ids.tolist()):
                            new_beams.append((seq + [int(token_id)], score + float(token_score)))
                    new_beams.sort(key=self._length_normalized_score, reverse=True)
                    groups[group_idx] = new_beams[:beams_per_group]
                    chosen_tokens.extend(seq[-1] for seq, _ in groups[group_idx] if seq[-1] != END_IDX)
                if all_finished:
                    break

            merged = [beam for group in groups for beam in group]
            merged.sort(key=self._length_normalized_score, reverse=True)
            top = merged[:max_candidates]
            # Recompute per-word confidence via one teacher-forcing forward pass per candidate,
            # reusing `memory` (no re-encoding the image). This is decoupled from the
            # diversity_penalty distortion applied above during beam search, so it reflects the
            # model's genuine confidence in each chosen word -- used by FormulaReranker for
            # S_base (mean per-token log probability).
            return [(self._decode(seq), score, self._teacher_force_log_probs(memory, seq)) for seq, score in top]

    def _teacher_force_log_probs(self, memory: torch.Tensor, seq: list[int]) -> list[float]:
        """`seq` is [START_IDX, w1, ..., wN, (END_IDX)]. Returns one log-prob per generated
        word (w1..wN) -- the model's own log P(wt | w<t, I), excluding START and the
        END-prediction step. Must be called inside the same `torch.no_grad()`/encoder context
        that produced `memory`."""
        if seq[-1] != END_IDX:
            seq = seq + [END_IDX]  # beam search hit max_length before emitting END
        if len(seq) < 3:  # need START + at least 1 word + END
            return []
        input_ids = torch.tensor(seq[:-1], dtype=torch.long, device=self.device).unsqueeze(0)
        tgt_embed = self.model.pos_encoder(self.model.embedding(input_ids))
        tgt_mask = self.model.generate_square_subsequent_mask(input_ids.size(1)).to(self.device)
        out = self.model.transformer_decoder(tgt=tgt_embed, memory=memory, tgt_mask=tgt_mask)
        log_probs = F.log_softmax(self.model.fc_out(out), dim=-1).squeeze(0)  # [T, vocab]
        target_ids = seq[1:]  # position t predicts target_ids[t]: w1..wN, then END
        token_log_probs = [log_probs[t, target_ids[t]].item() for t in range(len(target_ids))]
        return token_log_probs[:-1]  # drop the END-prediction, keep exactly N word log-probs

    def _load_model(self):
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"Caption checkpoint not found: {self.checkpoint_path}")
        if not self.vocab_path.is_file():
            raise FileNotFoundError(f"Vocab file not found: {self.vocab_path}")

        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        if any(key.startswith("module.") for key in state_dict):
            state_dict = {key.replace("module.", "", 1): value for key, value in state_dict.items()}
        # Số lớp decoder lấy từ chính checkpoint (2 hoặc 3...), tránh strict=False bỏ lặng lớp thừa.
        layer_ids = {int(k.split(".")[2]) for k in state_dict if k.startswith("transformer_decoder.layers.")}
        num_layers = max(layer_ids) + 1 if layer_ids else NUM_LAYERS
        model = ImageCaptioningTransformer(
            vocab_size=len(self.vocab), num_layers=num_layers, clip_source=self.clip_source
        ).to(self.device)
        # strict=False: checkpoint tu cac ablation co them tham so (vd image_proj/text_proj cua
        # contrastive loss trong N009/N010, xem research/) khong ton tai trong kien truc suy dien
        # o day (khong can cho inference/generate) - bo qua thay vi crash. In canh bao neu co
        # missing/unexpected key ngoai du kien de de phat hien loi that (vd nham file checkpoint).
        state_dict = align_clip_vision_keys(state_dict, model.state_dict().keys())
        result = model.load_state_dict(state_dict, strict=False)
        if result.missing_keys:
            raise RuntimeError(f"Checkpoint {self.checkpoint_path} thiếu trọng số captioner: {result.missing_keys[:3]}")
        if result.missing_keys or result.unexpected_keys:
            print(
                f"[!] load_state_dict (strict=False) tren {self.checkpoint_path}: "
                f"missing_keys={result.missing_keys}, unexpected_keys={result.unexpected_keys}"
            )
        model.eval()
        return model

    def _decode(self, sequence: list[int]) -> str:
        words = []
        for token in sequence:
            if token == END_IDX:
                break
            if token in (PAD_IDX, START_IDX):
                continue
            words.append(self.idx_to_word.get(token, "<unk>"))
        return " ".join(words)

    @staticmethod
    def _valid_num_beams(num_beams: int, max_candidates: int, num_beam_groups: int) -> int:
        num_beams = max(num_beams, max_candidates, num_beam_groups)
        remainder = num_beams % num_beam_groups
        return num_beams if remainder == 0 else num_beams + num_beam_groups - remainder

    @staticmethod
    def _length_normalized_score(beam: tuple[list[int], float]) -> float:
        sequence, score = beam
        penalty = ((5.0 + len(sequence)) ** 0.7) / (6.0**0.7)
        return score / penalty
