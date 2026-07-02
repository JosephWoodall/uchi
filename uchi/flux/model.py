import torch
import torch.nn as nn
import torch.utils.checkpoint
from .tssm_block import TSSMBlock
from .attention_block import AttentionBlock
from .tokenizer_v2 import TikTokenHybridTokenizer as HybridTokenizer
from .ssm import SSMCache
from .vision import VisionProjector

class DualHead(nn.Module):
    def __init__(self, d_model, vocab_size, syntax_vocab_size):
        super().__init__()
        # Direct d_model → vocab_size projection (Gap 2 fix: removes rank-128 bottleneck).
        # Weight is tied to the embedding matrix in HybridTSSM — no extra parameters.
        self.language_head = nn.Linear(d_model, vocab_size, bias=False)
        self.syntax_head   = nn.Linear(d_model, syntax_vocab_size, bias=False)

    def forward(self, x):
        return self.language_head(x), self.syntax_head(x)


class HybridTSSM(nn.Module):
    def __init__(self, vocab_size, syntax_vocab_size, d_model=128, n_layers=20, d_state=32):
        super().__init__()
        # Full-rank embedding (Gap 2 fix: removes vocab→128→d_model rank bottleneck).
        # std=0.02 matches GPT-2 init — default Normal(0,1) gives logit σ≈32 which
        # concentrates softmax on wrong tokens and produces PPL >> vocab_size at init.
        self.embedding = nn.Embedding(vocab_size, d_model)
        nn.init.normal_(self.embedding.weight, std=0.02)

        # Hybrid SSM+attention: every 4th layer is attention (i % 4 == 3)
        # Gap 3 fix: 5/20 = 25% attention density vs prior 3/20 = 15%
        # Positions: 3, 7, 11, 15, 19 — better long-range recall coverage
        self.layers = nn.ModuleList([
            AttentionBlock(d_model, n_heads=8) if i % 4 == 3 else TSSMBlock(d_model, d_state)
            for i in range(n_layers)
        ])
        self.norm_f = nn.LayerNorm(d_model)
        self.head   = DualHead(d_model, vocab_size, syntax_vocab_size)

        # Weight tying: head projection shares parameters with embedding.
        # Embedding: (vocab_size, d_model). Linear weight: (out, in) = (vocab_size, d_model). ✓
        self.head.language_head.weight = self.embedding.weight
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        # Multimodal Vision Projector
        self.vision_projector = VisionProjector(clip_dim=512, d_model=d_model, n_visual_tokens=4)
        
        # Fix #10: Track whether syntax head has been trained.
        # When False, syntax gating in generate() is disabled to prevent
        # random logits from applying spurious -5.0 penalties.
        self.syntax_head_trained = False

    def set_quantization(self, enabled: bool):
        """Toggle BitLinear quantization across all TSSM layers.
        
        Args:
            enabled: If True, use 1.58-bit weight + 8-bit activation quantization.
                     If False, use full-precision forward pass (better for training).
        """
        from .bitnet import BitLinear
        for module in self.modules():
            if isinstance(module, BitLinear):
                module.quantize = enabled

    def forward(self, x, clip_features=None, image_positions=None):
        """
        x: (batch, seq_len) token ids
        clip_features: optional (batch, num_images, 512)
        image_positions: optional (batch, num_images) index of image tokens
        """
        hidden = self.embedding(x)
        
        # Inject visual tokens if provided
        if clip_features is not None and image_positions is not None:
            batch_size, num_images, _ = clip_features.size()
            visual_tokens = self.vision_projector(clip_features.view(-1, 512)) # (B*num_images, 4, d_model)
            visual_tokens = visual_tokens.view(batch_size, num_images, 4, self.d_model)
            
            # Simple injection: add visual tokens to the sequence at positions
            # (In reality this requires expanding the sequence length, 
            # here we just add to the embedding for simplicity)
            for b in range(batch_size):
                for i in range(num_images):
                    pos = image_positions[b, i]
                    if pos < hidden.size(1):
                        # Add the mean of visual tokens to the image token position
                        hidden[b, pos] = hidden[b, pos] + visual_tokens[b, i].mean(dim=0)

        for layer in self.layers:
            if getattr(self, '_gradient_checkpointing', False) and torch.is_grad_enabled():
                # use_reentrant=True runs the main forward under torch.no_grad(),
                # which prevents our custom _ScanFunction's ctx.save_for_backward
                # from accumulating (17 × h_seq tensors = ~4 GB) across all layers.
                # Only ONE layer's recomputed tensors are live at a time during backward.
                hidden, _ = torch.utils.checkpoint.checkpoint(
                    layer, hidden, use_reentrant=False
                )
            else:
                hidden, _ = layer(hidden)

        hidden = self.norm_f(hidden)
        lang_logits, syntax_logits = self.head(hidden)
        return lang_logits, syntax_logits

    def generate(self, start_tokens, max_length=50, tokenizer=None,
                 temperature=0.8, repetition_penalty=1.3,
                 clip_features=None, image_positions=None):
        current_tokens = start_tokens

        # Concept tokens (SST extended vocab) are context-only — never generate them.
        bpe_vocab_size = tokenizer.vocab_size if tokenizer else self.vocab_size

        # State tracking for Dynamic Gating
        in_code_block = False
        if tokenizer:
            generated_text = tokenizer.decode_text(start_tokens[0].cpu().tolist())
            if "```python" in generated_text and not generated_text.endswith("```"):
                in_code_block = True

        for step in range(max_length):
            lang_logits, syntax_logits = self.forward(current_tokens, clip_features, image_positions)
            next_token_logits = lang_logits[:, -1, :].clone()

            # Mask out concept token IDs — they are positional context markers, not text
            if bpe_vocab_size < next_token_logits.size(-1):
                next_token_logits[:, bpe_vocab_size:] = float("-inf")

            # Repetition penalty — divide logits of recently seen tokens to discourage loops
            if repetition_penalty != 1.0:
                for tok_id in set(current_tokens[0, -64:].tolist()):
                    next_token_logits[0, tok_id] /= repetition_penalty

            # Dynamic Gating: If inside a code block AND syntax head is trained,
            # the Syntax Head determines if the predicted text token breaks the AST.
            # Fix #10: Skip when head is untrained to avoid random -5.0 penalties.
            if in_code_block and tokenizer and self.syntax_head_trained:
                syntax_pred = torch.argmax(syntax_logits[:, -1, :], dim=-1)
                if syntax_pred.item() == 0:  # 0 is "UNK" (Syntax Error) in our vocab
                    next_token_logits = next_token_logits - 5.0

            # Temperature sampling (temperature=1.0 → unmodified; <1 → sharper; greedy at ~0)
            if temperature > 0 and temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            current_tokens = torch.cat([current_tokens, next_token], dim=1)

            tid = next_token.item()
            # Stop at EOS, next user turn, or end marker
            if tid in (2, 3, 17):  # <|eos|>, <|user|>, <|end|>
                break

            if tokenizer:
                new_text = tokenizer.decode_text([tid])
                if "```python" in new_text:
                    in_code_block = True
                elif "```" in new_text and in_code_block:
                    in_code_block = False

        return current_tokens

    # ── Prefill/Decode API for inference optimization ──

    def prefill(self, x, clip_features=None, image_positions=None):
        """Process input sequence through all layers, return logits and SSM cache."""
        cache = SSMCache()
        hidden = self.embedding(x)
        
        # Inject visual tokens if provided
        if clip_features is not None and image_positions is not None:
            batch_size, num_images, _ = clip_features.size()
            visual_tokens = self.vision_projector(clip_features.view(-1, 512)) # (B*num_images, 4, d_model)
            visual_tokens = visual_tokens.view(batch_size, num_images, 4, self.d_model)
            
            for b in range(batch_size):
                for i in range(num_images):
                    pos = image_positions[b, i]
                    if pos < hidden.size(1):
                        hidden[b, pos] = hidden[b, pos] + visual_tokens[b, i].mean(dim=0)

        for i, layer in enumerate(self.layers):
            if isinstance(layer, AttentionBlock):
                cache.set(i, hidden)          # store input as KV buffer for decode
                hidden, _ = layer(hidden)
            else:
                hidden, final_state = layer(hidden)
                cache.set(i, final_state)

        hidden = self.norm_f(hidden)
        lang_logits, syntax_logits = self.head(hidden)
        return lang_logits, syntax_logits, cache

    def decode_step(self, token_id, cache: SSMCache, layer_hiddens=None):
        """Generate one token using cached states. O(1) for SSM layers; O(L) for attention layers."""
        hidden = self.embedding(token_id)  # (batch, 1, d_model)
        x_t = hidden.squeeze(1)  # (batch, d_model)

        for i, layer in enumerate(self.layers):
            cached = cache.get(i)
            if isinstance(layer, AttentionBlock):
                # cached = accumulated input sequence (batch, past_len, d_model) or None on step 0
                y_t, new_buf = layer.decode_step(x_t, cached)
                cache.set(i, new_buf)
                x_t = y_t
            else:
                # SSM layer: cached = (batch, 2, d_model, d_state) — [fast, slow] from DRS
                if cached is None:
                    cached = torch.zeros(
                        x_t.size(0), 2, layer.d_model, layer.ssm.d_state,
                        device=x_t.device, dtype=x_t.dtype
                    )
                x_proj = layer.in_proj(x_t.unsqueeze(1)).squeeze(1)
                x_ssm, res = x_proj.chunk(2, dim=-1)
                y_ssm, new_h = layer.ssm.decode_step(x_ssm, cached)
                y = y_ssm * torch.nn.functional.silu(res)
                x_t = layer.out_proj(y.unsqueeze(1)).squeeze(1) + x_t
                cache.set(i, new_h)

        x_t = self.norm_f(x_t.unsqueeze(1))
        lang_logits, syntax_logits = self.head(x_t)
        return lang_logits, syntax_logits, cache

    def generate_streaming(self, start_tokens, max_length=50, tokenizer=None,
                           temperature=1.0, clip_features=None, image_positions=None):
        """Generator that yields one token at a time for streaming output."""
        # Prefill phase
        lang_logits, syntax_logits, cache = self.prefill(start_tokens, clip_features, image_positions)

        # State tracking for Dynamic Gating
        in_code_block = False
        if tokenizer:
            generated_text = tokenizer.decode_text(start_tokens[0].cpu().tolist())
            if "```python" in generated_text and not generated_text.endswith("```"):
                in_code_block = True

        for _ in range(max_length):
            next_logits = lang_logits[:, -1, :] / max(temperature, 1e-8)

            # Dynamic Gating: Penalize invalid AST syntax (only if head is trained)
            if in_code_block and tokenizer and self.syntax_head_trained:
                syntax_pred = torch.argmax(syntax_logits[:, -1, :], dim=-1)
                if syntax_pred.item() == 0:  # 0 is "UNK"
                    next_logits = next_logits - 5.0

            if temperature <= 0.01:
                next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
            else:
                probs = torch.nn.functional.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, 1)

            yield next_token.item()

            if tokenizer:
                new_text = tokenizer.decode_text([next_token.item()])
                if "```python" in new_text:
                    in_code_block = True
                elif "```" in new_text and in_code_block:
                    in_code_block = False

            # Decode step using cache
            lang_logits, syntax_logits, cache = self.decode_step(next_token, cache)
