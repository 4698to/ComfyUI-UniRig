"""
UniRig autoregressive skeleton predictor — ComfyUI-native.

Wraps HuggingFace OPT-350m with a Michelangelo mesh encoder for
mesh-conditioned autoregressive joint prediction.

Inference-only: training_step, forward, and process_fn are removed.
"""

import json
import torch
from torch import nn, FloatTensor, LongTensor
from torch.nn.functional import pad
from pathlib import Path
from typing import Dict, List, Union
from transformers import AutoModelForCausalLM, OPTConfig, LogitsProcessor, LogitsProcessorList
from copy import deepcopy

import comfy.ops

from .model_spec import ModelSpec
from .parse_encoder import MAP_MESH_ENCODER, get_mesh_encoder
from .tokenizer_spec import TokenizerSpec, DetokenizeOutput

import logging

log = logging.getLogger("unirig")

# Load OPT-350m config from local JSON file (no HuggingFace download needed)
_CONFIG_PATH = Path(__file__).parent / "opt_350m_config.json"
with open(_CONFIG_PATH) as f:
    _OPT_350M_CONFIG = json.load(f)

# Check flash_attn availability
try:
    import flash_attn
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False
    log.info("flash_attn not available for AR model, using standard PyTorch attention (slower but functional)")


class VocabSwitchingLogitsProcessor(LogitsProcessor):
    def __init__(self, tokenizer: TokenizerSpec, start_tokens: LongTensor):
        self.tokenizer = tokenizer
        self.start_tokens = start_tokens
        assert start_tokens.ndim == 1

    def __call__(self, input_ids: LongTensor, scores: FloatTensor) -> FloatTensor:
        for batch_idx, sequence in enumerate(input_ids):
            mask = torch.full_like(scores[batch_idx], float('-inf'))
            sequence = torch.cat([self.start_tokens, sequence])
            tokens = self.tokenizer.next_posible_token(ids=sequence.detach().cpu().numpy())
            mask[tokens] = 0
            scores[batch_idx] = scores[batch_idx] + mask
        return scores


class UniRigAR(ModelSpec):

    def __init__(self, llm, mesh_encoder, dtype=None, device=None, operations=None, **kwargs):
        super().__init__()
        if operations is None:
            operations = comfy.ops.disable_weight_init
        self.dtype = dtype
        log.info("Initializing UniRigAR model...")

        self.tokenizer: TokenizerSpec = kwargs.get('tokenizer')
        self.vocab_size = self.tokenizer.vocab_size
        log.info("Vocab size: %s", self.vocab_size)

        _d = llm.copy()
        _d['vocab_size'] = self.tokenizer.vocab_size

        # Remove flash_attention_2 requirement if flash_attn is not available
        if not FLASH_ATTN_AVAILABLE and '_attn_implementation' in _d:
            original_impl = _d['_attn_implementation']
            _d.pop('_attn_implementation')
            log.info("Removed '%s' from config, using default attention", original_impl)
        elif FLASH_ATTN_AVAILABLE:
            log.info("Using flash_attention_2")

        # Build config from local JSON file (no HuggingFace download)
        config_dict = _OPT_350M_CONFIG.copy()
        config_dict['vocab_size'] = _d['vocab_size']
        log.info("Loading transformer model from local OPT-350m config")
        llm_config = OPTConfig(**config_dict)
        llm_config.pre_norm = True
        self.transformer = AutoModelForCausalLM.from_config(config=llm_config)
        log.info("[OK] Transformer loaded")

        self.hidden_size = llm['hidden_size']

        # Thread dtype/device/operations through to mesh encoder
        mesh_encoder = dict(mesh_encoder)
        mesh_encoder.setdefault('dtype', dtype)
        mesh_encoder.setdefault('device', device)
        mesh_encoder.setdefault('operations', operations)
        log.info("Loading mesh encoder...")
        self.mesh_encoder = get_mesh_encoder(**mesh_encoder)
        log.info("[OK] Mesh encoder loaded")

        if (
            isinstance(self.mesh_encoder, MAP_MESH_ENCODER.michelangelo) or
            isinstance(self.mesh_encoder, MAP_MESH_ENCODER.michelangelo_encoder)
        ):
            self.output_proj = operations.Linear(self.mesh_encoder.width, self.hidden_size, device=device, dtype=dtype)
        else:
            raise NotImplementedError()

    def encode_mesh_cond(self, vertices: FloatTensor, normals: FloatTensor) -> FloatTensor:
        # Cast inputs to model dtype (safety net)
        if self.dtype is not None:
            vertices = vertices.to(dtype=self.dtype)
            normals = normals.to(dtype=self.dtype)
        assert not torch.isnan(vertices).any()
        assert not torch.isnan(normals).any()
        if (
            isinstance(self.mesh_encoder, MAP_MESH_ENCODER.michelangelo) or
            isinstance(self.mesh_encoder, MAP_MESH_ENCODER.michelangelo_encoder)
        ):
            if (len(vertices.shape) == 3):
                shape_embed, latents, token_num, pre_pc = self.mesh_encoder.encode_latents(pc=vertices, feats=normals)
            else:
                shape_embed, latents, token_num, pre_pc = self.mesh_encoder.encode_latents(pc=vertices.unsqueeze(0), feats=normals.unsqueeze(0))
            latents = self.output_proj(latents)
            return latents
        else:
            raise NotImplementedError()

    @torch.no_grad()
    def generate(
        self,
        vertices: FloatTensor,
        normals: FloatTensor,
        cls: Union[str, None] = None,
        **kwargs,
    ) -> DetokenizeOutput:
        """Single-sample generation (no batch support)."""
        cond = self.encode_mesh_cond(vertices=vertices, normals=normals).to(dtype=self.transformer.dtype)

        start_tokens = [self.tokenizer.bos]

        if cls is not None:
            start_tokens.append(self.tokenizer.cls_name_to_token(cls=cls))
        start_tokens = torch.tensor(start_tokens).to(cond.device)
        start_embed = self.transformer.get_input_embeddings()(
            start_tokens.unsqueeze(0)
        ).to(dtype=self.transformer.dtype)
        cond = torch.cat([cond, start_embed], dim=1)

        processor = VocabSwitchingLogitsProcessor(
            tokenizer=self.tokenizer,
            start_tokens=start_tokens,
        )
        results = self.transformer.generate(
            inputs_embeds=cond,
            bos_token_id=self.tokenizer.bos,
            eos_token_id=self.tokenizer.eos,
            pad_token_id=self.tokenizer.pad,
            logits_processor=LogitsProcessorList([processor]),
            **kwargs,
        )
        output_ids = results[0, :]
        for token in reversed(start_tokens):
            output_ids = pad(output_ids, (1, 0), value=token)
        output_ids = output_ids.detach().cpu().numpy()

        res = self.tokenizer.detokenize(ids=output_ids)
        return res

    def predict_step(self, batch: Dict, no_cls: bool = False):
        vertices: FloatTensor = batch['vertices']
        normals: FloatTensor = batch['normals']
        paths: List[str] = batch['path']
        cls = batch['cls']
        generate_kwargs = deepcopy(batch['generate_kwargs'])

        no_cls = generate_kwargs.get('no_cls', False)
        use_dir_cls = generate_kwargs.get('use_dir_cls', False)
        assign_cls = generate_kwargs.get('assign_cls', None)

        generate_kwargs.pop('no_cls', None)
        generate_kwargs.pop('use_dir_cls', None)
        generate_kwargs.pop('assign_cls', None)

        if vertices.dim() == 2:
            vertices = vertices.unsqueeze(0)
            normals = normals.unsqueeze(0)
        outputs = []
        for i in range(vertices.shape[0]):
            if no_cls:
                _cls = None
            elif assign_cls is not None:
                _cls = assign_cls
            elif use_dir_cls:
                _cls = paths[i].removeprefix('./').split('/')[0]
            else:
                _cls = cls[i]
            res = self.generate(vertices=vertices[i], normals=normals[i], cls=_cls, **generate_kwargs)
            outputs.append(res)
        return outputs
