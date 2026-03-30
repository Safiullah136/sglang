from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch
from sgl_kernel.flash_attn import flash_attn_with_kvcache as flash_attn_with_kvcache_fa3

from sglang.jit_kernel.flash_attention_v4 import (
    flash_attn_with_kvcache as flash_attn_with_kvcache_fa4,
)
from sglang.srt.distributed.communication_op import AttnContextParallelCommunicate
from sglang.srt.distributed.parallel_state import (
    get_attn_context_model_parallel_rank,
    get_attn_context_model_parallel_world_size,
)
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention


@dataclass
class RingAttentionMetadata:
    """Metadata to be init once in the model forward pass,
    each layer's forward pass can reuse the metadata.

    For each init metadata function, we will try set up them in below order
    """

    # Sequence lengths for the forward batch
    cache_seqlens_int32: torch.Tensor = None
    # Maximum sequence length for query
    max_seq_len_q: int = 1
    # Maximum sequence length for key
    max_seq_len_k: int = 0
    # Cumulative sequence lengths for query
    cu_seqlens_q: torch.Tensor = None
    # Cumulative sequence lengths for key
    cu_seqlens_k: torch.Tensor = None
    # Window size (typically used by Gemma)
    window_size: tuple = (-1, -1)
    # Page table, the index of KV Cache Tables/Blocks
    page_table: torch.Tensor = None
    # Page table for Sliding Window Attention
    swa_page_table: torch.Tensor = None

    # Encoder metadata
    # Cumulative sequence lengths for encoder key
    encoder_cu_seqlens_k: torch.Tensor = None
    # Maximum sequence length for encoder key
    encoder_max_seq_len_k: int = 0
    # Sequence lengths for the forward batch
    encoder_lens_int32: torch.Tensor = None
    # Page table for the encoder
    encoder_page_table: torch.Tensor = None


class RingAttentionBackend(AttentionBackend):
    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """Initialize forward metadata hence all layers in the forward pass can reuse it."""
        metadata = RingAttentionMetadata()
        seqlens_in_batch = forward_batch.seq_lens
        batch_size = forward_batch.batch_size
        device = seqlens_in_batch.device

        if forward_batch.forward_mode.is_extend():
            metadata.cache_seqlens_int32 = seqlens_in_batch.to(torch.int32)
            metadata.max_seq_len_k = forward_batch.seq_lens_cpu.max().item()
            metadata.cu_seqlens_k = torch.nn.functional.pad(
                torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0)
            )
            metadata.page_table = forward_batch.req_to_token_pool.req_to_token[
                forward_batch.req_pool_indices, : metadata.max_seq_len_k
            ]

            metadata.max_seq_len_q = metadata.max_seq_len_k
            metadata.cu_seqlens_q = metadata.cu_seqlens_k

            # # Setup local attention if enabled
            # if forward_batch.forward_mode == ForwardMode.EXTEND:
            #     self._maybe_init_local_attn_metadata(forward_batch, metadata, device)

        # Encoder metadata for cross attention
        if forward_batch.encoder_lens is not None:
            assert (
                forward_batch.encoder_lens.numel() == 1
            ), "Only encoder size 1 is supported for now"

            metadata.encoder_lens_int32 = forward_batch.encoder_lens.to(torch.int32)
            metadata.encoder_cu_seqlens_k = torch.nn.functional.pad(
                torch.cumsum(metadata.encoder_lens_int32, dim=0, dtype=torch.int32),
                (1, 0),
            )
            metadata.encoder_max_seq_len_k = metadata.encoder_lens_int32.max().item()
            metadata.encoder_page_table = forward_batch.req_to_token_pool.req_to_token[
                forward_batch.req_pool_indices, : metadata.encoder_max_seq_len_k
            ]

            # Currently only support forward_batch.encoder_lens.numel() == 1
            metadata.page_table = forward_batch.req_to_token_pool.req_to_token[
                forward_batch.req_pool_indices,
                metadata.encoder_max_seq_len_k : (
                    metadata.encoder_max_seq_len_k + metadata.max_seq_len_k
                ),
            ]

        if self.use_sliding_window_kv_pool:
            metadata.swa_page_table = (
                self.token_to_kv_pool.translate_loc_from_full_to_swa(
                    metadata.page_table
                )
            )

        # Convert the page table to a strided format which is needed by FA3 API
        if self.page_size > 1:
            self.strided_indices = torch.arange(
                0, metadata.page_table.shape[1], self.page_size, device=self.device
            )

            if self.use_sliding_window_kv_pool:
                metadata.swa_page_table = (
                    metadata.swa_page_table[:, self.strided_indices] // self.page_size
                )

            metadata.page_table = (
                metadata.page_table[:, self.strided_indices] // self.page_size
            )

        self.forward_metadata = metadata

    # TODO (Support MLA, see flashattention_backend)
    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
    ):
        if k is not None:
            assert v is not None
            if save_kv_cache:
                cache_loc = (
                    forward_batch.out_cache_loc
                    if not layer.is_cross_attention
                    else forward_batch.encoder_out_cache_loc
                )
                forward_batch.token_to_kv_pool.set_kv_buffer(
                    layer, cache_loc, k, v, layer.k_scale, layer.v_scale
                )

        # Use precomputed metadata across all layers
        metadata = self.forward_metadata

        k_descale, v_descale = None, None
        # only use kv scaling if: 1) fp8 kv is explicitly enabled, 2) RadixAttention
        # has corresponding quantization method so that layer.k_scale is not None,
        # 3) layer.head_dim <= 256 since fa3 kernel require fp16 and bf16 data type in this case,
        # 4) fa_impl_ver != 4 since fa4 does not currently support fp8 queries and keys.
        if (
            self.kv_cache_dtype_str != "auto"
            and layer.head_dim <= 256
            and self.fa_impl_ver != 4
        ):
            if layer.k_scale is not None:
                descale_shape = (forward_batch.batch_size, layer.tp_k_head_num)
                k_descale = layer.k_scale.expand(descale_shape)
                v_descale = layer.v_scale.expand(descale_shape)
            q = q.to(self.kv_cache_dtype)

        causal = True
        if layer.is_cross_attention or layer.attn_type == AttentionType.ENCODER_ONLY:
            causal = False

        flash_attn_with_kvcache_base = flash_attn_with_kvcache_fa3

        flash_attn_with_kvcache = (
            flash_attn_with_kvcache_fa4
            if self.fa_impl_ver == 4
            else flash_attn_with_kvcache_base
        )

        page_table = metadata.page_table
        cu_seqlens_q = metadata.cu_seqlens_q
        cache_seqlens = metadata.cache_seqlens_int32
        max_seqlen_q = metadata.max_seq_len_q
        cu_seqlens_k = metadata.cu_seqlens_k

        # Use Flash Attention for prefill
        key_cache, value_cache = forward_batch.token_to_kv_pool.get_kv_buffer(
            layer.layer_id
        )
        key_cache = key_cache.view(
            -1, self.page_size, layer.tp_k_head_num, layer.head_dim
        )
        value_cache = value_cache.view(
            -1, self.page_size, layer.tp_v_head_num, layer.v_head_dim
        )
        if layer.is_cross_attention:
            page_table = metadata.encoder_page_table
            cache_seqlens = metadata.encoder_lens_int32
            cu_seqlens_k = metadata.encoder_cu_seqlens_k
            window_size = (-1, -1)

        cp_ring_comm = AttnContextParallelCommunicate()
        cp_world_size = get_attn_context_model_parallel_world_size()
        cp_rank = get_attn_context_model_parallel_rank()

        out, lse = None, None

        for step in range(cp_world_size):
            if step + 1 != cp_world_size:
                next_k = cp_ring_comm.send_recv(k)
                next_v = cp_ring_comm.send_recv(v)
                cp_ring_comm.commit()

            if not causal or step <= cp_ring_comm.rank:
                block_out, block_softmax_lse, *rest = flash_attn_with_kvcache(
                    q=q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
                    k_cache=key_cache,
                    v_cache=value_cache,
                    page_table=page_table,
                    cache_seqlens=cache_seqlens,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k_new=cu_seqlens_k,
                    max_seqlen_q=max_seqlen_q,
                    softmax_scale=layer.scaling,
                    causal=causal,
                    window_size=window_size,
                    softcap=layer.logit_cap,
                    k_descale=k_descale,
                    v_descale=v_descale,
                    return_softmax_lse=True,
                    num_splits=self.num_splits,
                )
                out, lse = update_out_and_lse(out, lse, block_out, block_softmax_lse)

            if step + 1 != cp_ring_comm.world_size:
                cp_ring_comm.wait()
                k = next_k
                v = next_v

        return out.view(-1, layer.tp_q_head_num * layer.v_head_dim)


def update_out_and_lse(
    out: Optional[torch.Tensor],
    lse: Optional[torch.Tensor],
    block_out: torch.Tensor,
    block_lse: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:

    def _update(current_out, current_lse):
        # new_lse = lse + torch.log(1 + torch.exp(block_lse - lse))
        # out = torch.exp(lse - new_lse) * out + torch.exp(block_lse - new_lse) * block_out
        # For additional context and discussion, please refer to:
        # https://github.com/zhuzilin/ring-flash-attention/pull/34#issuecomment-2076126795
        current_out = current_out - torch.nn.functional.sigmoid(
            block_lse - current_lse
        ) * (current_out - block_out)
        current_lse = current_lse - torch.nn.functional.logsigmoid(
            current_lse - block_lse
        )
        return current_out, current_lse

    block_out = block_out.to(torch.float32)
    block_lse = block_lse.unsqueeze(dim=-1)

    if out is None:
        return block_out, block_lse

    out, lse = _update(out, lse)

    return out, lse
