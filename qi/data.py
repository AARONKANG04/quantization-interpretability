"""Token streaming for SAE training: documents packed back to back into fixed-length chunks."""
from __future__ import annotations

import torch


def stream_token_chunks(tokenizer, texts, seq_len: int = 2048):
    """Yield int64 [seq_len] tensors from an iterable of texts, packing with EOS between documents."""
    buf = []
    eos = tokenizer.eos_token_id
    for t in texts:
        buf.extend(tokenizer(t, add_special_tokens=False)["input_ids"] + [eos])
        while len(buf) >= seq_len:
            yield torch.tensor(buf[:seq_len], dtype=torch.long)
            buf = buf[seq_len:]


def fineweb_edu_texts(min_chars: int = 50):
    from datasets import load_dataset

    for row in load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True).shuffle(seed=0, buffer_size=10_000):
        t = row["text"]
        if t and len(t) > min_chars:
            yield t


class ActivationStream:
    """Model forward on token chunks -> resid_post at one layer, shuffled through a buffer, served as [B, d] fp32."""

    def __init__(self, model, layer: int, chunks, batch_seqs: int = 8, buffer_tokens: int = 262_144, batch_size: int = 4096):
        from .models import ResidualCapture

        self.model, self.layer, self.chunks = model, layer, iter(chunks)
        self.batch_seqs, self.buffer_tokens, self.batch_size = batch_seqs, buffer_tokens, batch_size
        self.capture = ResidualCapture
        self.buf = None
        self.ptr = 0

    @torch.no_grad()
    def _fill(self):
        parts = []
        n = 0
        while n < self.buffer_tokens:
            seqs = [next(self.chunks) for _ in range(self.batch_seqs)]
            x = torch.stack(seqs).to(self.model.device)
            with self.capture(self.model, layers=[self.layer]) as cap:
                self.model(input_ids=x)
            h = cap.acts[self.layer].reshape(-1, cap.acts[self.layer].shape[-1]).float()
            parts.append(h)
            n += h.shape[0]
        buf = torch.cat(parts)
        self.buf = buf[torch.randperm(buf.shape[0], device=buf.device)]
        self.ptr = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.buf is None or self.ptr + self.batch_size > self.buf.shape[0]:
            self._fill()
        out = self.buf[self.ptr : self.ptr + self.batch_size]
        self.ptr += self.batch_size
        return out
