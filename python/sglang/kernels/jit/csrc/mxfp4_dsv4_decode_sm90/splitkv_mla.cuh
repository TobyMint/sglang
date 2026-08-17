#pragma once

#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/barrier.h>
#include <cutlass/cluster_launch.hpp>

#include "components/helpers.h"
#include "config.h"
#include "dequant.h"
#include "flashmla_utils.h"
#include "kerutils/kerutils.cuh"
#include "splitkv_mla.h"
#include <math_constants.h>
using namespace cute;

namespace sm90::decode::sparse_mxfp4_dsv4 {

static constexpr float MAX_INIT_VAL = -1e30;  // Prevent (-inf) - (-inf) = nan
using cutlass::arch::fence_view_async_shared;
using cutlass::arch::NamedBarrier;

// Every 368-byte MXFP4 row (and its page-block stride) is a multiple of 16
// bytes, so the NoPE and RoPE payloads can be loaded with single aligned
// vector loads.
__device__ __forceinline__ bf16x8 load_bf16x8(const void* address) {
  uint4 value;
  asm volatile("ld.global.nc.L1::evict_last.L2::128B.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(value.x), "=r"(value.y), "=r"(value.z), "=r"(value.w)
               : "l"(address));
  bf16x8 result;
  *reinterpret_cast<uint4*>(&result) = value;
  return result;
}

template <typename Tensor0, typename Tensor1, typename Tensor2>
__forceinline__ __device__ void scale_softmax(
    Tensor0& rP,
    Tensor1& rS,
    Tensor2& rO,
    float scale_softmax_log2,
    float sScale[],
    float rM[2],
    float rL[2],
    bool is_kv_valid[],
    int block_idx,
    int idx_in_warpgroup) {
  float scale_for_olds[2];
  CUTE_UNROLL
  for (int local_row_idx = 0; local_row_idx < 2; ++local_row_idx) {
    Tensor cur_rP = flatten(rP(make_coord(_, local_row_idx, _), _, _));
    Tensor cur_rS = flatten(rS(make_coord(_, local_row_idx, _), _, _));
    Tensor cur_rO = flatten(rO(make_coord(_, local_row_idx, _), _, _));

    float cur_max = -INFINITY;
    CUTE_UNROLL
    for (int i = 0; i < size(cur_rP); ++i) {
      if (!is_kv_valid[(i & 1) + (i / 2) * 8 + (idx_in_warpgroup % 4) * 2]) cur_rP(i) = -INFINITY;
      cur_max = max(cur_max, cur_rP(i));
    }
    cur_max = max(cur_max, __shfl_xor_sync(0xffffffff, cur_max, 1));
    cur_max = max(cur_max, __shfl_xor_sync(0xffffffff, cur_max, 2));

    cur_max *= scale_softmax_log2;
    float old_max = rM[local_row_idx];
    rM[local_row_idx] = max(cur_max, old_max);
    float scale_for_old = exp2f(old_max - rM[local_row_idx]);
    scale_for_olds[local_row_idx] = scale_for_old;

    CUTE_UNROLL
    for (int i = 0; i < size(cur_rO); ++i) {
      cur_rO(i) *= scale_for_old;
    }

    float cur_sum = 0;
    CUTE_UNROLL
    for (int i = 0; i < size(cur_rP); ++i) {
      cur_rP(i) = exp2f(cur_rP(i) * scale_softmax_log2 - rM[local_row_idx]);
      cur_rS(i) = (bf16)cur_rP(i);
      cur_sum += cur_rP(i);
    }

    rL[local_row_idx] = rL[local_row_idx] * scale_for_old + cur_sum;
  }
  if (idx_in_warpgroup % 4 == 0) *(float2*)(sScale + 2 * (idx_in_warpgroup / 4)) = *(float2*)(scale_for_olds);
}

// ---------------------------------------------------------------------------
// FP8-MMA mainloop (Phase 2): E4M3-repacked NoPE payload with the E8M0 block
// scales applied outside the WGMMA — on the FP32 QK partials per 128-dim
// block (DeepGEMM's paged-MQA-logits pattern) and on P per block before PV
// (exact power-of-two rescales, folded back through the row scale).  RoPE
// stays BF16 end to end.  CLUSTER_SIZE == 1 (h_q = 64) only for now.
// ---------------------------------------------------------------------------
template <int NUM_HEADS, typename TMAParams>
__device__ void devfunc_fp8_mainloop(const SparseAttnDecodeParams& params, const TMAParams& tma_params) {
  static_assert(NUM_HEADS == 64, "FP8-MMA path initially covers the h_q=64 geometry");
  constexpr int NUM_K_BUFS = 2;
  constexpr int BLOCK_M = 64;
  constexpr int TOPK_BLOCK_SIZE = 64;

  const int s_q_idx = blockIdx.y;
  const int partition_idx = blockIdx.z;
  const int warpgroup_idx = cutlass::canonical_warp_group_idx();
  const int idx_in_warpgroup = threadIdx.x % 128;
  const int warp_idx = cutlass::canonical_warp_idx_sync();

  extern __shared__ char wksp_buf[];
  using Kt = KernelTemplate<NUM_HEADS, true>;
  using Fp8T = typename Kt::Fp8T;
  auto& plan = *reinterpret_cast<typename Kt::SharedMemoryPlan*>(wksp_buf);
  Tensor sQ8 = make_tensor(make_smem_ptr(plan.q8.data()), typename Kt::SmemLayoutQ8{});
  Tensor sQRope = make_tensor(make_smem_ptr(plan.q_rope.data()), typename Kt::SmemLayoutQRope{});
  Tensor sS = make_tensor(make_smem_ptr(plan.s.data()), typename Kt::SmemLayoutS{});
  float* sM = plan.sM;
  float* sL = plan.sL;
  float* sScale = plan.sScale;

  if (warp_idx == 0 && elect_one_sync()) {
    cute::prefetch_tma_descriptor(&tma_params.tensor_map_o);
    plan.bar_q.init(1);
    CUTE_UNROLL
    for (int i = 0; i < NUM_K_BUFS; ++i) {
      plan.bar_k_local_ready[i].init(128);
      plan.bar_k_avail[i].init(256);
    }
    plan.bar_o_done.init(256);
    cutlass::arch::fence_barrier_init();
  }
  ku::barrier_cluster_arrive_relaxed();

  int bar_phase_k = 0;
  int bar_phase_o = 0;

  DecodingSchedMeta sched_meta = params.tile_scheduler_metadata_ptr[partition_idx];
  if (sched_meta.begin_req_idx >= params.b) return;
  ku::barrier_cluster_wait_acquire();

  struct MainloopArgs {
    int start_block_idx, end_block_idx;
    bool is_no_split;
    int topk_length, extra_topk_length, num_orig_kv_blocks;
  };
  auto get_cur_req_info = [&](int batch_idx) -> MainloopArgs {
    MainloopArgs args;
    int topk_length = params.topk_length ? __ldg(params.topk_length + batch_idx) : params.topk;
    topk_length = max(0, min(topk_length, params.topk));
    int orig_topk_padded = max(ku::ceil(topk_length, (int)TOPK_BLOCK_SIZE), (int)TOPK_BLOCK_SIZE);
    int extra_topk_length = params.extra_topk_length ? __ldg(params.extra_topk_length + batch_idx) : params.extra_topk;
    extra_topk_length = max(0, min(extra_topk_length, params.extra_topk));
    int total_topk_padded = orig_topk_padded + ku::ceil(extra_topk_length, (int)TOPK_BLOCK_SIZE);
    args.topk_length = topk_length;
    args.extra_topk_length = extra_topk_length;
    args.num_orig_kv_blocks = orig_topk_padded / TOPK_BLOCK_SIZE;
    args.start_block_idx = batch_idx == sched_meta.begin_req_idx ? sched_meta.begin_block_idx : 0;
    args.end_block_idx =
        batch_idx == sched_meta.end_req_idx ? sched_meta.end_block_idx : total_topk_padded / TOPK_BLOCK_SIZE;
    args.is_no_split = batch_idx == sched_meta.begin_req_idx
                           ? !sched_meta.is_first_req_splitted
                           : (batch_idx == sched_meta.end_req_idx ? !sched_meta.is_last_req_splitted : true);
    return args;
  };

  // Q staging: all 384 threads copy the wrapper-quantized Q tiles in
  // 16-byte granules through the cute layouts (16 E4M3 / 8 BF16 per granule).
  constexpr int NUM_Q8_GRANULES = BLOCK_M * Kt::HEAD_DIM_NOPE / 16;
  constexpr int NUM_QROPE_GRANULES = BLOCK_M * Kt::HEAD_DIM_ROPE / 8;
  auto stage_q = [&](int batch_idx) {
    const int tid = threadIdx.x;
    const Fp8T* g_q8 = params.q_nope_fp8 + batch_idx * params.stride_q8_b + s_q_idx * params.stride_q8_s_q;
    const bf16* g_qr = params.q_rope + batch_idx * params.stride_qrope_b + s_q_idx * params.stride_qrope_s_q;
    CUTE_UNROLL
    for (int g = tid; g < NUM_Q8_GRANULES; g += 384) {
      const int head = g / (Kt::HEAD_DIM_NOPE / 16);
      const int dim16 = (g % (Kt::HEAD_DIM_NOPE / 16)) * 16;
      *reinterpret_cast<uint4*>(&sQ8(head, dim16)) =
          *reinterpret_cast<const uint4*>(g_q8 + head * params.stride_q8_h_q + dim16);
    }
    CUTE_UNROLL
    for (int g = tid; g < NUM_QROPE_GRANULES; g += 384) {
      const int head = g / (Kt::HEAD_DIM_ROPE / 8);
      const int dim8 = (g % (Kt::HEAD_DIM_ROPE / 8)) * 8;
      *reinterpret_cast<uint4*>(&sQRope(head, dim8)) =
          *reinterpret_cast<const uint4*>(g_qr + head * params.stride_qrope_h_q + dim8);
    }
    NamedBarrier::arrive_and_wait(384, 15 /* scratch id, unused by the FP8 path */);
  };
  stage_q(sched_meta.begin_req_idx);

  if (warpgroup_idx == 2) {
    // ------------------------------------------------------------------
    // Producer: repack E2M1 -> E4M3 (lossless) into BOTH orientations, keep
    // RoPE BF16, stage the four per-token block scales.  No scale multiplies
    // here — consumers apply the E8M0 exponents outside the WGMMA.
    // ------------------------------------------------------------------
    cutlass::arch::warpgroup_reg_dealloc<152>();

    const int producer_warp_idx = __shfl_sync(0xffffffff, idx_in_warpgroup / 32, 0);
    const int lane_idx = idx_in_warpgroup % 32;
    const int my_token_idx = producer_warp_idx * 8 + lane_idx % 8;

    CUTE_NO_UNROLL
    for (int batch_idx = sched_meta.begin_req_idx; batch_idx <= sched_meta.end_req_idx; ++batch_idx) {
      MainloopArgs args = get_cur_req_info(batch_idx);
      if (batch_idx != sched_meta.begin_req_idx) {
        plan.bar_o_done.wait(bar_phase_o);
        bar_phase_o ^= 1;
      }
      stage_q(batch_idx);  // rendezvous with the consumers' per-request restage

      int* gIndices = params.indices + batch_idx * params.stride_indices_b + s_q_idx * params.stride_indices_s_q;
      int* gExtraIndices = params.extra_indices == nullptr
                               ? nullptr
                               : params.extra_indices + batch_idx * params.stride_extra_indices_b +
                                     s_q_idx * params.stride_extra_indices_s_q;

      struct IsOrigBlock {};
      struct IsExtraBlock {};
      auto process_one_block = [&](int block_idx, auto is_extra_block_t) {
        static constexpr bool IS_EXTRA_BLOCK = std::is_same_v<decltype(is_extra_block_t), IsExtraBlock>;
        const int buf_idx = (block_idx - args.start_block_idx) % NUM_K_BUFS;
        const int rel_block_idx = IS_EXTRA_BLOCK ? block_idx - args.num_orig_kv_blocks : block_idx;
        int* const indices_base = (IS_EXTRA_BLOCK ? gExtraIndices : gIndices) + rel_block_idx * TOPK_BLOCK_SIZE;
        const int page_block_size = IS_EXTRA_BLOCK ? params.extra_page_block_size : params.page_block_size;
        const int num_blocks = IS_EXTRA_BLOCK ? params.extra_num_blocks : params.num_blocks;
        const int topk_length = IS_EXTRA_BLOCK ? args.extra_topk_length : args.topk_length;
        const int64_t k_block_stride = IS_EXTRA_BLOCK ? params.stride_extra_kv_block : params.stride_kv_block;
        const int64_t k_row_stride = IS_EXTRA_BLOCK ? params.stride_extra_kv_row : params.stride_kv_row;
        const uint8_t* const k_ptr = reinterpret_cast<const uint8_t*>(IS_EXTRA_BLOCK ? params.extra_kv : params.kv);
        const int64_t token_capacity = static_cast<int64_t>(num_blocks) * page_block_size;

        Tensor sK8buf = make_tensor(make_smem_ptr(plan.u.k[buf_idx].k8.data()), typename Kt::SmemLayoutK8{});
        Tensor sK8PVbuf = make_tensor(make_smem_ptr(plan.u.k[buf_idx].k8pv.data()), typename Kt::SmemLayoutK8PV{});
        Tensor sKRopeBuf = make_tensor(make_smem_ptr(plan.u.k[buf_idx].rope.data()), typename Kt::SmemLayoutKRope{});
        const mxfp4::E2m1E4m3Lut repack_lut = mxfp4::make_e2m1_e4m3_lut();

        CUTE_UNROLL
        for (int round = 0; round < 2; ++round) {
          const int my_token = my_token_idx + round * 32;
          const int selected_pos = rel_block_idx * TOPK_BLOCK_SIZE + my_token;
          const int token_index = selected_pos < topk_length ? __ldg(indices_base + my_token) : -1;
          const bool token_is_valid = token_index >= 0 && static_cast<int64_t>(token_index) < token_capacity;
          const uint8_t* gK_base = nullptr;
          if (token_is_valid) {
            const int block_index =
                static_cast<int>(static_cast<uint32_t>(token_index) / static_cast<uint32_t>(page_block_size));
            const int rel_idx_in_block =
                static_cast<int>(static_cast<uint32_t>(token_index) % static_cast<uint32_t>(page_block_size));
            gK_base = k_ptr + block_index * k_block_stride + rel_idx_in_block * k_row_stride;
          }

          if (round == 0) {
            plan.bar_k_avail[buf_idx].wait((bar_phase_k >> buf_idx & 1) ^ 1);
          }

          CUTE_UNROLL
          for (int dim_idx = 0; dim_idx < Kt::HEAD_DIM_NOPE / 64; ++dim_idx) {
            const int logical_dim = dim_idx * 64 + (lane_idx / 8) * 16;
            uint64_t packed = 0;
            if (token_is_valid) {
              packed = nvfp4::load_packed_e2m1x16(gK_base + logical_dim / 2);
            }
            const uint2 lo = mxfp4::repack_e2m1x8_e4m3x8((uint32_t)packed, repack_lut);
            const uint2 hi = mxfp4::repack_e2m1x8_e4m3x8((uint32_t)(packed >> 32), repack_lut);
            const uint4 bytes = {lo.x, lo.y, hi.x, hi.y};
            // QK orientation: one 16-byte granule at (token, dim).
            *reinterpret_cast<uint4*>(&sK8buf(my_token, logical_dim)) = bytes;
            // PV orientation: scattered per-byte writes at (dim, token).
            const Fp8T* vals = reinterpret_cast<const Fp8T*>(&bytes);
            CUTE_UNROLL
            for (int j = 0; j < 16; ++j) {
              sK8PVbuf(logical_dim + j, my_token) = vals[j];
            }
          }

          CUTE_UNROLL
          for (int dim_idx = 0; dim_idx < Kt::HEAD_DIM_ROPE / 32; ++dim_idx) {
            const int rope_dim = dim_idx * 32 + (lane_idx / 8) * 8;
            bf16x8 rope = {};
            if (token_is_valid) {
              rope = load_bf16x8(
                  gK_base + Kt::PACKED_NOPE_BYTES + Kt::NUM_SCALES + Kt::PAD_BYTES + rope_dim * sizeof(bf16));
            }
            *reinterpret_cast<__int128_t*>(&sKRopeBuf(my_token, rope_dim)) =
                *reinterpret_cast<const __int128_t*>(&rope);
          }
        }

        // Per-token block scales (slots 0/4/8/12 = byte 0 of each u32 of
        // the scale area's first 16 bytes), staged as floats.
        if ((lane_idx / 8) == 0) {
          CUTE_UNROLL
          for (int round = 0; round < 2; ++round) {
            const int my_token = my_token_idx + round * 32;
            const int selected_pos = rel_block_idx * TOPK_BLOCK_SIZE + my_token;
            const int token_index = selected_pos < topk_length ? __ldg(indices_base + my_token) : -1;
            const bool token_is_valid = token_index >= 0 && static_cast<int64_t>(token_index) < token_capacity;
            const uint8_t* gK_base = token_is_valid ? k_ptr + (token_index / page_block_size) * k_block_stride +
                                                          (token_index % page_block_size) * k_row_stride
                                                    : nullptr;
            uint4 words = {};
            if (token_is_valid) {
              asm volatile("ld.global.nc.v4.u32 {%0, %1, %2, %3}, [%4];"
                           : "=r"(words.x), "=r"(words.y), "=r"(words.z), "=r"(words.w)
                           : "l"(gK_base + Kt::PACKED_NOPE_BYTES));
            }
            float4 scales = {
                mxfp4::e8m0_bits_to_float((uint8_t)(words.x & 0xff)),
                mxfp4::e8m0_bits_to_float((uint8_t)(words.y & 0xff)),
                mxfp4::e8m0_bits_to_float((uint8_t)(words.z & 0xff)),
                mxfp4::e8m0_bits_to_float((uint8_t)(words.w & 0xff))};
            *reinterpret_cast<float4*>(&plan.u.k[buf_idx].block_scale[my_token][0]) = scales;
          }
        }

        fence_view_async_shared();

        if (idx_in_warpgroup < 32) {
          const int pos0 = rel_block_idx * TOPK_BLOCK_SIZE + lane_idx * 2;
          const int pos1 = pos0 + 1;
          const int selected0 = pos0 < topk_length ? __ldg(indices_base + lane_idx * 2) : -1;
          const int selected1 = pos1 < topk_length ? __ldg(indices_base + lane_idx * 2 + 1) : -1;
          *reinterpret_cast<char2*>(&plan.is_kv_valid[buf_idx][lane_idx * 2]) = {
              selected0 >= 0 && static_cast<int64_t>(selected0) < token_capacity,
              selected1 >= 0 && static_cast<int64_t>(selected1) < token_capacity,
          };
        }

        plan.bar_k_local_ready[buf_idx].arrive();
        bar_phase_k ^= 1 << buf_idx;
      };

      CUTE_NO_UNROLL
      for (int block_idx = args.start_block_idx; block_idx < min(args.num_orig_kv_blocks, args.end_block_idx);
           ++block_idx) {
        process_one_block(block_idx, IsOrigBlock{});
      }
      CUTE_NO_UNROLL
      for (int block_idx = max(args.start_block_idx, args.num_orig_kv_blocks); block_idx < args.end_block_idx;
           ++block_idx) {
        process_one_block(block_idx, IsExtraBlock{});
      }

      Kt::sync_all_threads_in_cluster();
    }
    return;
  }

  // ------------------------------------------------------------------------
  // Consumers.  WG0: block-decomposed FP8 QK -> RoPE BF16 QK -> online
  // softmax -> per-block rescaled P (E4M3, staged for both PV sides) -> PV
  // over dims 0-255.  WG1: PV over dims 256-511 (two FP8 blocks, the tail
  // 64-dim FP8 block, and the RoPE BF16 block).
  // ------------------------------------------------------------------------
  auto qk_tiles = [](int t) {  // (token, dim) 64-wide sub-tile pair helpers
    return t;
  };
  (void)qk_tiles;

  if (warpgroup_idx == 0) {
    cutlass::arch::warpgroup_reg_alloc<192>();

    auto tiled_mma_QK8 = typename Kt::TiledMMA_QK_Fp8{};
    auto thr_QK8 = tiled_mma_QK8.get_slice(idx_in_warpgroup);
    auto tiled_mma_rope = typename Kt::TiledMMA_QK{};
    auto thr_rope = tiled_mma_rope.get_slice(idx_in_warpgroup);
    auto tiled_mma_PV128 = typename Kt::TiledMMA_PV128_Fp8{};
    auto thr_PV128 = tiled_mma_PV128.get_slice(idx_in_warpgroup);

    float rL[2], rM[2], rK[2];
    Tensor rP = partition_fragment_C(tiled_mma_QK8, Shape<Int<BLOCK_M>, Int<TOPK_BLOCK_SIZE>>{});
    Tensor acc = partition_fragment_C(tiled_mma_QK8, Shape<Int<BLOCK_M>, Int<TOPK_BLOCK_SIZE>>{});
    Tensor rS = make_tensor<bf16>(partition_shape_A(typename Kt::TiledMMA_PV_LocalP_Fp8{}, Shape<Int<BLOCK_M>, Int<TOPK_BLOCK_SIZE>>{}));
    Tensor rO0 = partition_fragment_C(tiled_mma_PV128, Shape<Int<BLOCK_M>, _128>{});
    Tensor rO1 = partition_fragment_C(tiled_mma_PV128, Shape<Int<BLOCK_M>, _128>{});

    float rAttn_sink[2] = {-CUDART_INF_F, -CUDART_INF_F};
    if (params.attn_sink != nullptr) {
      for (int i = 0; i < 2; ++i) {
        int head_idx = get_AorC_row_idx(i, idx_in_warpgroup);
        rAttn_sink[i] = __ldg((float*)params.attn_sink + head_idx) * CUDART_L2E_F;
      }
    }

#pragma unroll 1
    for (int batch_idx = sched_meta.begin_req_idx; batch_idx <= sched_meta.end_req_idx; ++batch_idx) {
      MainloopArgs args = get_cur_req_info(batch_idx);
      stage_q(batch_idx);

      rL[0] = rL[1] = 0.0f;
      rM[0] = rM[1] = MAX_INIT_VAL;
      rK[0] = rK[1] = 0.0f;
      cute::fill(rO0, 0.f);
      cute::fill(rO1, 0.f);

      float rShift[2] = {1.0f, 1.0f};
      {
        const float* g_shift =
            params.q_shift + batch_idx * params.stride_qshift_b + s_q_idx * params.stride_qshift_s_q;
        for (int i = 0; i < 2; ++i) {
          rShift[i] = __ldg(g_shift + get_AorC_row_idx(i, idx_in_warpgroup));
        }
      }

      CUTE_NO_UNROLL
      for (int block_idx = args.start_block_idx; block_idx < args.end_block_idx; ++block_idx) {
        const int buf_idx = (block_idx - args.start_block_idx) % NUM_K_BUFS;

        plan.bar_k_local_ready[buf_idx].wait(bar_phase_k >> buf_idx & 1);
        bar_phase_k ^= 1 << buf_idx;

        Tensor sK8buf = make_tensor(make_smem_ptr(plan.u.k[buf_idx].k8.data()), typename Kt::SmemLayoutK8{});
        Tensor sKRopeBuf = make_tensor(make_smem_ptr(plan.u.k[buf_idx].rope.data()), typename Kt::SmemLayoutKRope{});
        const float* bscale = &plan.u.k[buf_idx].block_scale[0][0];

        // QK over the 448 NoPE dims: seven 64-dim FP8 tiles, each
        // accumulated fresh and folded into rP with its 128-dim block scale
        // (times the per-head Q shift).
        cute::clear(rP);
#pragma unroll
        for (int t = 0; t < Kt::HEAD_DIM_NOPE / 64; ++t) {
          Tensor sQt = make_tensor(
              make_smem_ptr(plan.q8.data() + (typename Kt::SmemLayoutQ8{})(_0{}, Int<64>{} * t)),
              typename Kt::SmemLayoutQ8Tiles<1>{});
          Tensor sKt = make_tensor(
              make_smem_ptr(plan.u.k[buf_idx].k8.data() + (typename Kt::SmemLayoutK8{})(_0{}, Int<64>{} * t)),
              typename Kt::SmemLayoutK8Tiles<1>{});
          gemm<true, -1>(tiled_mma_QK8, thr_QK8.partition_fragment_A(sQt), thr_QK8.partition_fragment_B(sKt), acc);
          cute::warpgroup_wait<0>();
          const int blk = t >> 1;
          CUTE_UNROLL
          for (int lr = 0; lr < 2; ++lr) {
            Tensor cur_rP = flatten(rP(make_coord(_, lr, _), _, _));
            Tensor cur_acc = flatten(acc(make_coord(_, lr, _), _, _));
            CUTE_UNROLL
            for (int i = 0; i < size(cur_rP); ++i) {
              const int tok = (i & 1) + (i / 2) * 8 + (idx_in_warpgroup % 4) * 2;
              cur_rP(i) += cur_acc(i) * (bscale[tok * 4 + blk] * rShift[lr]);
            }
          }
        }
        // RoPE QK (BF16) accumulates directly into rP.
        {
          Tensor sKRopeT = make_tensor(make_smem_ptr(plan.u.k[buf_idx].rope.data()), typename Kt::SmemLayoutKRope{});
          Tensor sQR = make_tensor(make_smem_ptr(plan.q_rope.data()), typename Kt::SmemLayoutQRope{});
          gemm<false, -1>(tiled_mma_rope, thr_rope.partition_fragment_A(sQR), thr_rope.partition_fragment_B(sKRopeT),
                          rP);
          cute::warpgroup_wait<0>();
        }
        (void)sK8buf;
        (void)sKRopeBuf;

        // Online softmax (rP <- P), rescaling both PV accumulators.
        if (block_idx != args.start_block_idx)
          NamedBarrier::arrive_and_wait(256, Kt::NamedBarriers::sScale_and_sS_free);
        {
          float scale_for_olds[2];
          CUTE_UNROLL
          for (int lr = 0; lr < 2; ++lr) {
            Tensor cur_rP = flatten(rP(make_coord(_, lr, _), _, _));
            Tensor cur_rO0 = flatten(rO0(make_coord(_, lr, _), _, _));
            Tensor cur_rO1 = flatten(rO1(make_coord(_, lr, _), _, _));
            float cur_max = -INFINITY;
            CUTE_UNROLL
            for (int i = 0; i < size(cur_rP); ++i) {
              if (!plan.is_kv_valid[buf_idx][(i & 1) + (i / 2) * 8 + (idx_in_warpgroup % 4) * 2]) cur_rP(i) = -INFINITY;
              cur_max = max(cur_max, cur_rP(i));
            }
            cur_max = max(cur_max, __shfl_xor_sync(0xffffffff, cur_max, 1));
            cur_max = max(cur_max, __shfl_xor_sync(0xffffffff, cur_max, 2));

            cur_max *= params.sm_scale_div_log2;
            float old_max = rM[lr];
            rM[lr] = max(cur_max, old_max);
            float scale_for_old = exp2f(old_max - rM[lr]);
            scale_for_olds[lr] = scale_for_old;

            CUTE_UNROLL
            for (int i = 0; i < size(cur_rO0); ++i) cur_rO0(i) *= scale_for_old;
            CUTE_UNROLL
            for (int i = 0; i < size(cur_rO1); ++i) cur_rO1(i) *= scale_for_old;

            float cur_sum = 0;
            CUTE_UNROLL
            for (int i = 0; i < size(cur_rP); ++i) {
              cur_rP(i) = exp2f(cur_rP(i) * params.sm_scale_div_log2 - rM[lr]);
              cur_sum += cur_rP(i);
            }
            rL[lr] = rL[lr] * scale_for_old + cur_sum;
          }
          if (idx_in_warpgroup % 4 == 0)
            *(float2*)(sScale + 2 * (idx_in_warpgroup / 4)) = *(float2*)(scale_for_olds);
        }

        // Per-block rescaled P: P̃_b = P * 2^{k_row} * 2^{s_{t,b}}, staged as
        // E4M3 for the FP8 PV gemms; the RoPE PV keeps BF16 P * 2^{k_row}.
        {
          Tensor sS8buf = make_tensor(make_smem_ptr(plan.s8.data()), typename Kt::SmemLayoutS8{});
          float k_row[2] = {0.f, 0.f};
          CUTE_UNROLL
          for (int lr = 0; lr < 2; ++lr) {
            Tensor cur_rP = flatten(rP(make_coord(_, lr, _), _, _));
            float mx = 0.f;
            CUTE_UNROLL
            for (int i = 0; i < size(cur_rP); ++i) mx = max(mx, cur_rP(i));
            k_row[lr] = mx > 0.f ? (6.f - log2f(mx)) : 0.f;  // headroom for x2^{s_b}
            rK[lr] = k_row[lr];
          }
          CUTE_UNROLL
          for (int lr = 0; lr < 2; ++lr) {
            Tensor cur_rP = flatten(rP(make_coord(_, lr, _), _, _));
            Tensor cur_rS = flatten(rS(make_coord(_, lr, _), _, _));
            CUTE_UNROLL
            for (int i = 0; i < size(cur_rP); ++i) {
              cur_rS(i) = (bf16)(cur_rP(i) * exp2f(k_row[lr]));
            }
          }
          CUTE_UNROLL
          for (int b = 0; b < 4; ++b) {
            CUTE_UNROLL
            for (int lr = 0; lr < 2; ++lr) {
              Tensor cur_rP = flatten(rP(make_coord(_, lr, _), _, _));
              const int row = get_AorC_row_idx(lr, idx_in_warpgroup);
              CUTE_UNROLL
              for (int i = 0; i < size(cur_rP); ++i) {
                const int tok = (i & 1) + (i / 2) * 8 + (idx_in_warpgroup % 4) * 2;
                const float v = cur_rP(i) * exp2f(k_row[lr]) * bscale[tok * 4 + b];
                sS8buf(row, tok) = Fp8T(v);
              }
            }
          }
          Kt::save_rPb_to_sP(rS, sS, idx_in_warpgroup);
          fence_view_async_shared();
        }

        // PV over dims 0-255: blocks 0 and 1.
        {
          Tensor sB0 = make_tensor(
              make_smem_ptr(plan.u.k[buf_idx].k8pv.data() + (typename Kt::SmemLayoutK8PV{})(_0{}, _0{})),
              typename Kt::SmemLayoutK8PVTiles<2>{});
          Tensor sB1 = make_tensor(
              make_smem_ptr(plan.u.k[buf_idx].k8pv.data() + (typename Kt::SmemLayoutK8PV{})(Int<128>{}, _0{})),
              typename Kt::SmemLayoutK8PVTiles<2>{});
          Tensor sS8_0 = make_tensor(make_smem_ptr(plan.s8.data()), typename Kt::SmemLayoutS8{});
          Tensor sS8_1 =
              make_tensor(make_smem_ptr(plan.s8.data() + cosize_v<typename Kt::SmemLayoutS8>),
                          typename Kt::SmemLayoutS8{});
          gemm<false, -1>(tiled_mma_PV128, thr_PV128.partition_fragment_A(sS8_0),
                          thr_PV128.partition_fragment_B(sB0), rO0);
          gemm<false, -1>(tiled_mma_PV128, thr_PV128.partition_fragment_A(sS8_1),
                          thr_PV128.partition_fragment_B(sB1), rO1);
          cute::warpgroup_wait<0>();
        }

        NamedBarrier::arrive(256, Kt::NamedBarriers::sScale_and_sS_ready);

        plan.bar_k_avail[buf_idx].arrive();
      }

      // ---- epilogue (WG0): row stats, lse, dims 0-255 ----
      rL[0] += __shfl_xor_sync(0xffffffff, rL[0], 1);
      rL[0] += __shfl_xor_sync(0xffffffff, rL[0], 2);
      rL[1] += __shfl_xor_sync(0xffffffff, rL[1], 1);
      rL[1] += __shfl_xor_sync(0xffffffff, rL[1], 2);

      if (idx_in_warpgroup % 4 == 0) {
        CUTE_UNROLL
        for (int i = 0; i < 2; ++i) {
          int row = get_AorC_row_idx(i, idx_in_warpgroup);
          sL[row] = rL[i];
          sM[row] = rM[i];
          plan.sKrow[row] = rK[i];
        }
      }

      float o_scales[2];
      CUTE_UNROLL
      for (int i = 0; i < 2; ++i) {
        if (args.is_no_split) {
          o_scales[i] = rL[i] == 0.0f ? 0.0f : __fdividef(1.0f, rL[i] + exp2f(rAttn_sink[i] - rM[i]));
        } else {
          o_scales[i] = rL[i] == 0.0f ? 0.0f : __fdividef(1.0f, rL[i]);
        }
        o_scales[i] *= exp2f(-rK[i]);  // undo the P rescale headroom
        if (idx_in_warpgroup % 4 == 0) {
          int row = get_AorC_row_idx(i, idx_in_warpgroup);
          plan.sOScale[row] = o_scales[i];
        }
      }

      NamedBarrier::arrive_and_wait(256, Kt::NamedBarriers::oBuf_free_and_sL_ready);

      CUTE_UNROLL
      for (int i = 0; i < 2; ++i)
        rL[i] = rL[i] == 0.0f ? 1.0f : rL[i];

      const int start_head_idx = 0;
      const int num_valid_seq_q = min(params.h_q - start_head_idx, BLOCK_M);
      {
        int n_split_idx = batch_idx == sched_meta.begin_req_idx ? sched_meta.begin_split_idx : 0;
        int split_idx = __ldg(params.num_splits_ptr + batch_idx) + n_split_idx;
        float* oaccum_ptr = (float*)params.o_accum + split_idx * params.stride_o_accum_split +
                            s_q_idx * params.stride_o_accum_s_q + start_head_idx * params.stride_o_accum_h_q;
        float* gLseAccum = (float*)params.lse_accum + split_idx * params.stride_lse_accum_split +
                           s_q_idx * params.stride_lse_accum_s_q + start_head_idx;

        Tensor gO0 = make_tensor(
            make_gmem_ptr(oaccum_ptr),
            make_layout(Shape<Int<BLOCK_M>, _128>{}, make_stride(params.stride_o_accum_h_q, _1{})));
        Tensor gO1 = make_tensor(
            make_gmem_ptr(oaccum_ptr + 128),
            make_layout(Shape<Int<BLOCK_M>, _128>{}, make_stride(params.stride_o_accum_h_q, _1{})));
        Tensor tOg0 = thr_PV128.partition_C(gO0);
        Tensor tOg1 = thr_PV128.partition_C(gO1);
        CUTE_UNROLL
        for (int lr = 0; lr < 2; ++lr) {
          Tensor cur_g0 = flatten(tOg0(make_coord(_, lr, _), _, _));
          Tensor cur_g1 = flatten(tOg1(make_coord(_, lr, _), _, _));
          Tensor cur_r0 = flatten(rO0(make_coord(_, lr, _), _, _));
          Tensor cur_r1 = flatten(rO1(make_coord(_, lr, _), _, _));
          CUTE_UNROLL
          for (int i = 0; i < size(cur_r0); ++i) cur_g0(i) = cur_r0(i) * o_scales[lr];
          CUTE_UNROLL
          for (int i = 0; i < size(cur_r1); ++i) cur_g1(i) = cur_r1(i) * o_scales[lr];
        }

        if (idx_in_warpgroup < num_valid_seq_q) {
          float cur_L = sL[idx_in_warpgroup];
          gLseAccum[idx_in_warpgroup] =
              cur_L == 0.0f ? -INFINITY : log2f(cur_L) + sM[idx_in_warpgroup] - plan.sKrow[idx_in_warpgroup];
        }
      }

      plan.bar_o_done.arrive();
      Kt::sync_all_threads_in_cluster();
    }
    return;
  }

  if (cute::thread0()) CUTE_INVALID_CONTROL_PATH("FP8-MMA WG1 lands next");
}

template <int NUM_HEADS, bool USE_FP8_MMA>
template <typename TMAParams>
__device__ void
KernelTemplate<NUM_HEADS, USE_FP8_MMA>::devfunc(const SparseAttnDecodeParams& params, const TMAParams& tma_params) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 900)) || (defined(__CLION_IDE__) || defined(__VSCODE_IDE__))
  if constexpr (USE_FP8_MMA) {
    devfunc_fp8_mainloop<NUM_HEADS>(params, tma_params);
    return;
  } else {
    const int head_block_idx = NUM_M_BLOCKS == 1 ? 0 : blockIdx.x;
    const int s_q_idx = blockIdx.y;
    const int partition_idx = blockIdx.z;
    const int idx_in_cluster = CLUSTER_SIZE == 1 ? 0 : head_block_idx % 2;
    const int warpgroup_idx = cutlass::canonical_warp_group_idx();
    const int idx_in_warpgroup = threadIdx.x % 128;
    const int warp_idx = cutlass::canonical_warp_idx_sync();

    // Define shared tensors
    extern __shared__ char wksp_buf[];
    SharedMemoryPlan& plan = *reinterpret_cast<SharedMemoryPlan*>(wksp_buf);
    Tensor sQ = make_tensor(make_smem_ptr(plan.q.data()), SmemLayoutQ{});
    Tensor sOBuf = make_tensor(make_smem_ptr(plan.u.oBuf.data()), SmemLayoutOBuf{});
    Tensor sOAccumBuf = make_tensor(make_smem_ptr(plan.u.oAccumBuf.data()), SmemLayoutOAccumBuf{});
    Tensor sS = make_tensor(make_smem_ptr(plan.s.data()), SmemLayoutS{});
    float* sM = plan.sM;
    float* sL = plan.sL;
    float* sScale = plan.sScale;

    // Prefetch TMA descriptors
    if (warp_idx == 0 && elect_one_sync()) {
      cute::prefetch_tma_descriptor(tma_params.tma_Q.get_tma_descriptor());
      cute::prefetch_tma_descriptor(&tma_params.tensor_map_o);
    }

    // Initialize TMA barriers
    if (warp_idx == 0 && elect_one_sync()) {
      plan.bar_q.init(1);
      if constexpr (CLUSTER_SIZE == 2) {
        CUTE_UNROLL
        for (int i = 0; i < NUM_K_BUFS; ++i) {
          plan.bar_k_local_ready[i].init(128);
          plan.bar_k_remote_ready[i].init(1);
          plan.bar_k_avail[i].init(4);
        }
      } else {
        CUTE_UNROLL
        for (int i = 0; i < NUM_K_BUFS; ++i) {
          plan.bar_k_local_ready[i].init(128);
          plan.bar_k_avail[i].init(256);
        }
      }
      plan.bar_o_done.init(256);
      cutlass::arch::fence_barrier_init();
    }
    ku::barrier_cluster_arrive_relaxed();

    int bar_phase_k = 0;  // Don't use array here to prevent using local memory
    int bar_phase_o = 0;  // phase tracker for bar_o_done (epilogue drain)

    // Programmatic Dependent Launch: Wait for the previous kernel to finish
    // Don't use PDL because of compiler bugs!
    // cudaGridDependencySynchronize();

    DecodingSchedMeta sched_meta = params.tile_scheduler_metadata_ptr[partition_idx];

    if (sched_meta.begin_req_idx >= params.b) return;

    // The barrier init above is performed by one elected warp-0 lane while
    // arrive_relaxed only asynchronously signals arrival. Issue the first TMA
    // (which arrives on bar_q) only after the cluster-wide acquire so every
    // barrier init is visible — otherwise bar_q can be used uninitialized.
    ku::barrier_cluster_wait_acquire();

    if (warp_idx == 0 && elect_one_sync()) {
      Tensor gQ = flat_divide(
          tma_params.tma_Q.get_tma_tensor(tma_params.shape_Q)(_, _, s_q_idx, sched_meta.begin_req_idx),
          Tile<Int<BLOCK_M>, Int<HEAD_DIM_K>>{})(_, _, head_block_idx, _0{});
      launch_tma_copy(tma_params.tma_Q, gQ, sQ, plan.bar_q, TMA::CacheHintSm90::EVICT_FIRST);
      plan.bar_q.arrive_and_expect_tx(BLOCK_M * HEAD_DIM_K * sizeof(bf16));
    }

    struct MainloopArgs {
      int start_block_idx, end_block_idx;
      bool is_no_split;

      int topk_length, extra_topk_length, num_orig_kv_blocks;
    };
    auto get_cur_req_info = [&](int batch_idx) -> MainloopArgs {
      MainloopArgs args;
      int topk_length = params.topk_length ? __ldg(params.topk_length + batch_idx) : params.topk;
      topk_length = max(0, min(topk_length, params.topk));
      int orig_topk_padded = max(ku::ceil(topk_length, (int)TOPK_BLOCK_SIZE), (int)TOPK_BLOCK_SIZE);
      int extra_topk_length =
          params.extra_topk_length ? __ldg(params.extra_topk_length + batch_idx) : params.extra_topk;
      extra_topk_length = max(0, min(extra_topk_length, params.extra_topk));
      int total_topk_padded = orig_topk_padded + ku::ceil(extra_topk_length, (int)TOPK_BLOCK_SIZE);
      args.topk_length = topk_length;
      args.extra_topk_length = extra_topk_length;
      args.num_orig_kv_blocks = orig_topk_padded / TOPK_BLOCK_SIZE;

      args.start_block_idx = batch_idx == sched_meta.begin_req_idx ? sched_meta.begin_block_idx : 0;
      args.end_block_idx =
          batch_idx == sched_meta.end_req_idx ? sched_meta.end_block_idx : total_topk_padded / TOPK_BLOCK_SIZE;
      args.is_no_split = batch_idx == sched_meta.begin_req_idx
                             ? !sched_meta.is_first_req_splitted
                             : (batch_idx == sched_meta.end_req_idx ? !sched_meta.is_last_req_splitted : true);

      return args;
    };

    if (warpgroup_idx == 0) {
      cutlass::arch::warpgroup_reg_alloc<192>();

      TiledMMA tiled_mma_QK = TiledMMA_QK{};
      ThrMMA thr_mma_QK = tiled_mma_QK.get_slice(idx_in_warpgroup);
      TiledMMA tiled_mma_PV = TiledMMA_PV_LocalP{};
      ThrMMA thr_mma_PV = tiled_mma_PV.get_slice(idx_in_warpgroup);

      float rL[2], rM[2];
      Tensor rO = partition_fragment_C(TiledMMA_PV_LocalP{}, Shape<Int<BLOCK_M>, Int<HEAD_DIM_V / 2>>{});
      Tensor rP = partition_fragment_C(TiledMMA_QK{}, Shape<Int<BLOCK_M>, Int<TOPK_BLOCK_SIZE>>{});
      Tensor rS =
          make_tensor<bf16>(partition_shape_A(TiledMMA_PV_LocalP{}, Shape<Int<BLOCK_M>, Int<TOPK_BLOCK_SIZE>>{}));

      float rAttn_sink[2] = {-CUDART_INF_F, -CUDART_INF_F};
      if (params.attn_sink != nullptr) {
        for (int i = 0; i < 2; ++i) {
          int head_idx = head_block_idx * BLOCK_M + get_AorC_row_idx(i, idx_in_warpgroup);
          rAttn_sink[i] = __ldg((float*)params.attn_sink + head_idx) * CUDART_L2E_F;
        }
      }

#pragma unroll 1
      for (int batch_idx = sched_meta.begin_req_idx; batch_idx <= sched_meta.end_req_idx; ++batch_idx) {
        MainloopArgs args = get_cur_req_info(batch_idx);

        rL[0] = rL[1] = 0.0f;
        rM[0] = rM[1] = MAX_INIT_VAL;
        cute::fill(rO, 0.);

        // Wait for Q
        plan.bar_q.wait((sched_meta.begin_req_idx - batch_idx) & 1);

        CUTE_NO_UNROLL
        for (int block_idx = args.start_block_idx; block_idx < args.end_block_idx; block_idx++) {
          int buf_idx = (block_idx - args.start_block_idx) % NUM_K_BUFS;
          Tensor sK = make_tensor(make_smem_ptr(plan.u.k[buf_idx].data()), SmemLayoutK{});
          Tensor sV = make_tensor(make_smem_ptr(plan.u.k[buf_idx].data()), SmemLayoutHalfV{});

          // Wait, issue WGMMA
          plan.bar_k_local_ready[buf_idx].wait(bar_phase_k >> buf_idx & 1);
          if constexpr (CLUSTER_SIZE == 2) {
            plan.bar_k_remote_ready[buf_idx].wait(bar_phase_k >> buf_idx & 1);
          }

          gemm<true, -1>(tiled_mma_QK, thr_mma_QK.partition_fragment_A(sQ), thr_mma_QK.partition_fragment_B(sK), rP);

          bar_phase_k ^= 1 << buf_idx;

          cute::warpgroup_wait<0>();

          // Calculate S = softmax(mask(scale(P)))
          if (block_idx != args.start_block_idx)
            NamedBarrier::arrive_and_wait(
                256, NamedBarriers::sScale_and_sS_free);  // Make sure that sScale and sS is free

          // Since in our case TOPK_BLOCK_SIZE == BLOCK_M, so we only need to do OOB checking for the last 2 blocks
          scale_softmax(
              rP,
              rS,
              rO,
              params.sm_scale_div_log2,
              sScale,
              rM,
              rL,
              plan.is_kv_valid[buf_idx],
              block_idx,
              idx_in_warpgroup);

          // Store S into shared, inform warpgroup 1
          save_rPb_to_sP(rS, sS, idx_in_warpgroup);
          fence_view_async_shared();

          // Issue O += S @ V
          gemm<false, -1>(tiled_mma_PV, rS, thr_mma_PV.partition_fragment_B(sV), rO);

          NamedBarrier::arrive(256, NamedBarriers::sScale_and_sS_ready);

          cute::warpgroup_wait<0>();

          if constexpr (CLUSTER_SIZE == 2) {
            plan.bar_k_avail[buf_idx].arrive(0, idx_in_warpgroup == 32);
            plan.bar_k_avail[buf_idx].arrive(1, idx_in_warpgroup == 64);
          } else {
            plan.bar_k_avail[buf_idx].arrive();
          }
        }

        // Copy the next q
        if (threadIdx.x / 32 == 0 && elect_one_sync()) {
          if (batch_idx != sched_meta.end_req_idx) {
            Tensor gQ = flat_divide(
                tma_params.tma_Q.get_tma_tensor(tma_params.shape_Q)(_, _, s_q_idx, batch_idx + 1),
                Tile<Int<BLOCK_M>, Int<HEAD_DIM_K>>{})(_, _, head_block_idx, _0{});
            launch_tma_copy(tma_params.tma_Q, gQ, sQ, plan.bar_q, TMA::CacheHintSm90::EVICT_FIRST);
            plan.bar_q.arrive_and_expect_tx(BLOCK_M * HEAD_DIM_K * sizeof(bf16));
          }
        }

        // Synchronize L and M across warpgroups
        rL[0] += __shfl_xor_sync(0xffffffff, rL[0], 1);
        rL[0] += __shfl_xor_sync(0xffffffff, rL[0], 2);
        rL[1] += __shfl_xor_sync(0xffffffff, rL[1], 1);
        rL[1] += __shfl_xor_sync(0xffffffff, rL[1], 2);

        if (idx_in_warpgroup % 4 == 0) {
          CUTE_UNROLL
          for (int i = 0; i < 2; ++i) {
            int row = get_AorC_row_idx(i, idx_in_warpgroup);
            sL[row] = rL[i];
            sM[row] = rM[i];
          }
        }

        float o_scales[2];
        CUTE_UNROLL
        for (int i = 0; i < 2; ++i) {
          if (args.is_no_split) {
            o_scales[i] = rL[i] == 0.0f ? 0.0f : __fdividef(1.0f, rL[i] + exp2f(rAttn_sink[i] - rM[i]));
          } else {
            o_scales[i] = rL[i] == 0.0f ? 0.0f : __fdividef(1.0f, rL[i]);
          }
          if (idx_in_warpgroup % 4 == 0) {
            int row = get_AorC_row_idx(i, idx_in_warpgroup);
            plan.sOScale[row] = o_scales[i];
          }
        }

        // This is a synchronization point for warpgroup 0/1.
        // Warpgroup 0 should wait wg 1 for oBuf/oAccumBuf (overlapped with k) to be free
        // Warpgroup 1 should wait wg 0 for sL to be ready
        NamedBarrier::arrive_and_wait(256, NamedBarriers::oBuf_free_and_sL_ready);

        CUTE_UNROLL
        for (int i = 0; i < 2; ++i)
          rL[i] = rL[i] == 0.0f ? 1.0f : rL[i];

        int start_head_idx = head_block_idx * BLOCK_M;
        int num_valid_seq_q = min(params.h_q - start_head_idx, BLOCK_M);
        if (args.is_no_split) {
          bf16* o_ptr = (bf16*)params.out + batch_idx * params.stride_o_b + s_q_idx * params.stride_o_s_q +
                        start_head_idx * params.stride_o_h_q;  // (BLOCK_M, HEAD_DIM_V) : (params.stride_o_h_q, 1)
          Tensor gO = make_tensor(
              make_gmem_ptr(o_ptr),
              make_layout(Shape<Int<BLOCK_M>, Int<HEAD_DIM_V>>{}, make_stride(params.stride_o_h_q, _1{})));
          float* gSoftmaxLse = (float*)params.lse + batch_idx * params.stride_lse_b + s_q_idx * params.stride_lse_s_q +
                               start_head_idx;  // (BLOCK_M) : (1)

          store_o<true>(
              rO,
              gO,
              sOBuf,
              sOAccumBuf,
              plan,
              o_scales,
              tma_params,
              batch_idx,
              s_q_idx,
              head_block_idx,
              num_valid_seq_q,
              warpgroup_idx,
              idx_in_warpgroup);

          int i = threadIdx.x;
          if (i < num_valid_seq_q) {
            float cur_L = sL[i];
            gSoftmaxLse[i] = cur_L == 0.0f ? INFINITY : logf(cur_L) + sM[i] / (float)M_LOG2E;
          }

          cute::tma_store_wait<0>();
        } else {
          int n_split_idx = batch_idx == sched_meta.begin_req_idx ? sched_meta.begin_split_idx : 0;
          int split_idx = __ldg(params.num_splits_ptr + batch_idx) + n_split_idx;
          float* oaccum_ptr =
              (float*)params.o_accum + split_idx * params.stride_o_accum_split + s_q_idx * params.stride_o_accum_s_q +
              start_head_idx * params.stride_o_accum_h_q;  // (BLOCK_M, HEAD_DIM_V) : (params.stride_o_accum_h_q, 1)
          float* gSoftmaxLseAccum = (float*)params.lse_accum + split_idx * params.stride_lse_accum_split +
                                    s_q_idx * params.stride_lse_accum_s_q + start_head_idx;  // (BLOCK_M) : (1)
          Tensor gOAccum = make_tensor(
              make_gmem_ptr(oaccum_ptr),
              make_layout(Shape<Int<BLOCK_M>, Int<HEAD_DIM_V>>{}, make_stride(params.stride_o_accum_h_q, _1{})));
          store_o<false>(
              rO,
              gOAccum,
              sOBuf,
              sOAccumBuf,
              plan,
              o_scales,
              tma_params,
              batch_idx,
              s_q_idx,
              head_block_idx,
              num_valid_seq_q,
              warpgroup_idx,
              idx_in_warpgroup);

          int i = threadIdx.x;
          if (i < num_valid_seq_q) {
            float cur_L = sL[i];
            gSoftmaxLseAccum[i] = cur_L == 0.0f ? -INFINITY : log2f(cur_L) + sM[i];
          }

          cute::tma_store_wait<0>();
        }

        // Tell the producer that the epilogue (oBuf) has fully drained the K
        // buffer overlay before it starts writing the next request's rows.
        plan.bar_o_done.arrive();

        // Signal programmatic launch completion as soon as the epilogue has
        // drained o_accum/lse_accum (tma_store_wait above); combine can start
        // while the remaining threads still reach the cluster sync below.
        if (batch_idx == sched_meta.end_req_idx && threadIdx.x / 32 == 0 && elect_one_sync()) {
          cudaTriggerProgrammaticLaunchCompletion();
        }

        sync_all_threads_in_cluster();
      }
    } else if (warpgroup_idx == 1) {
      cutlass::arch::warpgroup_reg_dealloc<160>();

      TiledMMA tiled_mma_PV = TiledMMA_PV_RemoteP{};
      ThrMMA thr_mma_PV = tiled_mma_PV.get_slice(idx_in_warpgroup);
      Tensor rO = partition_fragment_C(tiled_mma_PV, Shape<Int<BLOCK_M>, Int<HEAD_DIM_V / 2>>{});

#pragma unroll 1
      for (int batch_idx = sched_meta.begin_req_idx; batch_idx <= sched_meta.end_req_idx; ++batch_idx) {
        MainloopArgs args = get_cur_req_info(batch_idx);
        cute::fill(rO, 0.);

        CUTE_NO_UNROLL
        for (int block_idx = args.start_block_idx; block_idx < args.end_block_idx; block_idx++) {
          int buf_idx = (block_idx - args.start_block_idx) % NUM_K_BUFS;
          Tensor sV =
              make_tensor(make_smem_ptr(plan.u.k[buf_idx].data() + (SmemLayoutV{})(_256{}, _0{})), SmemLayoutHalfV{});

          // Wait for S and sScale
          NamedBarrier::arrive_and_wait(256, NamedBarriers::sScale_and_sS_ready);

          // Scale O
          float cur_scales[2];
          *(float2*)cur_scales = *(float2*)(sScale + (idx_in_warpgroup / 4) * 2);
          CUTE_UNROLL
          for (int local_row_idx = 0; local_row_idx < 2; ++local_row_idx) {
            Tensor cur_rO = flatten(rO(make_coord(_, local_row_idx, _), _, _));
            CUTE_UNROLL
            for (int i = 0; i < size(cur_rO); ++i) {
              cur_rO(i) *= cur_scales[local_row_idx];
            }
          }

          // Issue O += S @ V, and wait
          gemm<false, -1>(tiled_mma_PV, thr_mma_PV.partition_fragment_A(sS), thr_mma_PV.partition_fragment_B(sV), rO);
          cute::warpgroup_wait<0>();

          if constexpr (CLUSTER_SIZE == 2) {
            plan.bar_k_avail[buf_idx].arrive(0, idx_in_warpgroup == 32);
            plan.bar_k_avail[buf_idx].arrive(1, idx_in_warpgroup == 64);
          } else {
            plan.bar_k_avail[buf_idx].arrive();
          }

          if (block_idx != args.end_block_idx - 1)
            NamedBarrier::arrive(256, NamedBarriers::sScale_and_sS_free);  // Tell WG0 that sScale and sS are available
        }

        NamedBarrier::arrive_and_wait(256, NamedBarriers::oBuf_free_and_sL_ready);

        float o_scales[2];
        CUTE_UNROLL
        for (int i = 0; i < 2; ++i) {
          int row = get_AorC_row_idx(i, idx_in_warpgroup);
          o_scales[i] = plan.sOScale[row];
        }

        int start_head_idx = head_block_idx * BLOCK_M;
        int num_valid_seq_q = min(params.h_q - start_head_idx, BLOCK_M);
        if (args.is_no_split) {
          bf16* o_ptr = (bf16*)params.out + batch_idx * params.stride_o_b + s_q_idx * params.stride_o_s_q +
                        start_head_idx * params.stride_o_h_q;  // (BLOCK_M, HEAD_DIM_V) : (params.stride_o_h_q, 1)
          Tensor gO = make_tensor(
              make_gmem_ptr(o_ptr),
              make_layout(Shape<Int<BLOCK_M>, Int<HEAD_DIM_V>>{}, make_stride(params.stride_o_h_q, _1{})));

          store_o<true>(
              rO,
              gO,
              sOBuf,
              sOAccumBuf,
              plan,
              o_scales,
              tma_params,
              batch_idx,
              s_q_idx,
              head_block_idx,
              num_valid_seq_q,
              warpgroup_idx,
              idx_in_warpgroup);

          cute::tma_store_wait<0>();
        } else {
          int n_split_idx = batch_idx == sched_meta.begin_req_idx ? sched_meta.begin_split_idx : 0;
          int split_idx = __ldg(params.num_splits_ptr + batch_idx) + n_split_idx;
          float* oaccum_ptr =
              (float*)params.o_accum + split_idx * params.stride_o_accum_split + s_q_idx * params.stride_o_accum_s_q +
              start_head_idx * params.stride_o_accum_h_q;  // (BLOCK_M, HEAD_DIM_V) : (params.stride_o_accum_h_q, 1)
          Tensor gOAccum = make_tensor(
              make_gmem_ptr(oaccum_ptr),
              make_layout(Shape<Int<BLOCK_M>, Int<HEAD_DIM_V>>{}, make_stride(params.stride_o_accum_h_q, _1{})));
          store_o<false>(
              rO,
              gOAccum,
              sOBuf,
              sOAccumBuf,
              plan,
              o_scales,
              tma_params,
              batch_idx,
              s_q_idx,
              head_block_idx,
              num_valid_seq_q,
              warpgroup_idx,
              idx_in_warpgroup);

          cute::tma_store_wait<0>();
        }

        // Same epilogue-drain signal as warpgroup 0 (256 arrivals per request).
        plan.bar_o_done.arrive();

        sync_all_threads_in_cluster();
      }
    } else {
      // Producer warpgroup. Each group of four lanes owns one selected token;
      // each lane dequantizes one 16-element E2M1 block at a time.
      cutlass::arch::warpgroup_reg_dealloc<152>();

      static_assert(CLUSTER_SIZE == 1 || CLUSTER_SIZE == 2);
      static constexpr int NUM_TOKENS_PER_THREAD = CLUSTER_SIZE == 1 ? 2 : 1;
      static constexpr int NUM_TOKENS_PER_ROUND = 32;
      const int producer_warp_idx = __shfl_sync(0xffffffff, idx_in_warpgroup / 32, 0);
      const int lane_idx = idx_in_warpgroup % 32;
      const int my_token_idx_base = producer_warp_idx * 8 + lane_idx % 8;

      CUTE_NO_UNROLL
      for (int batch_idx = sched_meta.begin_req_idx; batch_idx <= sched_meta.end_req_idx; ++batch_idx) {
        MainloopArgs args = get_cur_req_info(batch_idx);
        // A fast producer must not overwrite the K buffers while the consumer
        // warpgroups' epilogue is still draining oBuf from the previous request.
        // (The first request has no prior epilogue; the phase then advances one
        // flip per request, matching the 256 consumer arrivals above.)
        if (batch_idx != sched_meta.begin_req_idx) {
          plan.bar_o_done.wait(bar_phase_o);
          bar_phase_o ^= 1;
        }

        int* gIndices = params.indices + batch_idx * params.stride_indices_b + s_q_idx * params.stride_indices_s_q;
        int* gExtraIndices = params.extra_indices == nullptr
                                 ? nullptr
                                 : params.extra_indices + batch_idx * params.stride_extra_indices_b +
                                       s_q_idx * params.stride_extra_indices_s_q;

        struct IsOrigBlock {};
        struct IsExtraBlock {};

        auto process_one_block = [&](int block_idx, auto is_extra_block_t) {
          static constexpr bool IS_EXTRA_BLOCK = std::is_same_v<decltype(is_extra_block_t), IsExtraBlock>;
          const int buf_idx = (block_idx - args.start_block_idx) % NUM_K_BUFS;
          const int rel_block_idx = IS_EXTRA_BLOCK ? block_idx - args.num_orig_kv_blocks : block_idx;
          int* const indices_base = (IS_EXTRA_BLOCK ? gExtraIndices : gIndices) + rel_block_idx * TOPK_BLOCK_SIZE;
          const int page_block_size = IS_EXTRA_BLOCK ? params.extra_page_block_size : params.page_block_size;
          const int num_blocks = IS_EXTRA_BLOCK ? params.extra_num_blocks : params.num_blocks;
          const int topk_length = IS_EXTRA_BLOCK ? args.extra_topk_length : args.topk_length;
          const int64_t k_block_stride = IS_EXTRA_BLOCK ? params.stride_extra_kv_block : params.stride_kv_block;
          const int64_t k_row_stride = IS_EXTRA_BLOCK ? params.stride_extra_kv_row : params.stride_kv_row;
          const uint8_t* const k_ptr = reinterpret_cast<const uint8_t*>(IS_EXTRA_BLOCK ? params.extra_kv : params.kv);
          const int64_t token_capacity = static_cast<int64_t>(num_blocks) * page_block_size;
          transac_bar_t* const peer_bar_k_remote_ready = get_peer_addr(&plan.bar_k_remote_ready[buf_idx]);

          CUTE_UNROLL
          for (int round = 0; round < NUM_TOKENS_PER_THREAD; ++round) {
            const int my_token_idx = my_token_idx_base + round * NUM_TOKENS_PER_ROUND;
            const int selected_pos =
                rel_block_idx * TOPK_BLOCK_SIZE + idx_in_cluster * (TOPK_BLOCK_SIZE / 2) + my_token_idx;
            const int token_index = selected_pos < topk_length
                                        ? __ldg(indices_base + idx_in_cluster * (TOPK_BLOCK_SIZE / 2) + my_token_idx)
                                        : -1;
            const bool token_is_valid = token_index >= 0 && static_cast<int64_t>(token_index) < token_capacity;
            const int safe_token_index = token_is_valid ? token_index : 0;
            const uint8_t* gK_base = nullptr;
            if (token_is_valid) {
              const int block_index =
                  page_block_size > 0
                      ? static_cast<int>(
                            static_cast<uint32_t>(safe_token_index) / static_cast<uint32_t>(page_block_size))
                      : 0;
              const int rel_idx_in_block =
                  page_block_size > 0
                      ? static_cast<int>(
                            static_cast<uint32_t>(safe_token_index) % static_cast<uint32_t>(page_block_size))
                      : 0;
              gK_base = k_ptr + block_index * k_block_stride + rel_idx_in_block * k_row_stride;
            }

            bf16* const sK_nope_base = plan.u.k[buf_idx].data() +
                                       (idx_in_cluster * (TOPK_BLOCK_SIZE / 2) + my_token_idx) * 8 +
                                       ((lane_idx / 8) * 16) * TOPK_BLOCK_SIZE;
            bf16* const sK_nope_peer_base = get_peer_addr(sK_nope_base);

            // The buffer can overlap with the prior consumer. Do not
            // write any K/V data until both consumer warpgroups release it.
            if (round == 0) {
              plan.bar_k_avail[buf_idx].wait((bar_phase_k >> buf_idx & 1) ^ 1);
            }

            if constexpr (CLUSTER_SIZE == 2) {
              if (round == 0 && idx_in_warpgroup == 0) {
                plan.bar_k_remote_ready[buf_idx].arrive_and_expect_tx(
                    (TOPK_BLOCK_SIZE / 2) * (HEAD_DIM_NOPE + HEAD_DIM_ROPE) * sizeof(bf16));
              }
            }

            CUTE_UNROLL
            for (int dim_idx = 0; dim_idx < HEAD_DIM_NOPE / 64; ++dim_idx) {
              // Each lane group owns 16 consecutive NoPE dims within the 64-dim
              // tile; SCALE_BLOCK_SIZE only groups the E8M0 scale bytes below.
              const int logical_dim = dim_idx * 64 + (lane_idx / 8) * 16;
              uint64_t packed = 0;
              uint8_t block_scale_bits = 0;
              if (token_is_valid) {
                packed = nvfp4::load_packed_e2m1x16(gK_base + logical_dim / 2);
                block_scale_bits = mxfp4::load_scale_bits(gK_base + PACKED_NOPE_BYTES + logical_dim / SCALE_BLOCK_SIZE);
              }
              const float effective_scale = token_is_valid ? mxfp4::e8m0_bits_to_float(block_scale_bits) : 0.0f;
              const nvfp4::bf16x16 dequant = nvfp4::dequant_e2m1x16(packed, effective_scale);
              const int smem_offset = dim_idx * 64 * TOPK_BLOCK_SIZE;

              *reinterpret_cast<__int128_t*>(sK_nope_base + smem_offset) =
                  *reinterpret_cast<const __int128_t*>(&dequant.lo);
              *reinterpret_cast<__int128_t*>(sK_nope_base + smem_offset + 8 * TOPK_BLOCK_SIZE) =
                  *reinterpret_cast<const __int128_t*>(&dequant.hi);
              if constexpr (CLUSTER_SIZE == 2) {
                st_async_128b(sK_nope_peer_base + smem_offset, dequant.lo, peer_bar_k_remote_ready);
                st_async_128b(
                    sK_nope_peer_base + smem_offset + 8 * TOPK_BLOCK_SIZE, dequant.hi, peer_bar_k_remote_ready);
              }
            }

            const uint8_t* const gK_rope = token_is_valid ? gK_base + PACKED_NOPE_BYTES + NUM_SCALES + PAD_BYTES +
                                                                (lane_idx / 8) * 8 * sizeof(bf16)
                                                          : nullptr;
            bf16* const sK_rope_base = plan.u.k[buf_idx].data() +
                                       (idx_in_cluster * (TOPK_BLOCK_SIZE / 2) + my_token_idx) * 8 +
                                       ((lane_idx / 8) * 8) * TOPK_BLOCK_SIZE;
            bf16* const sK_rope_peer_base = get_peer_addr(sK_rope_base);

            CUTE_UNROLL
            for (int dim_idx = 0; dim_idx < HEAD_DIM_ROPE / 32; ++dim_idx) {
              bf16x8 rope = {};
              if (token_is_valid) {
                rope = load_bf16x8(gK_rope + dim_idx * 32 * sizeof(bf16));
              }
              const int smem_offset = (HEAD_DIM_NOPE + dim_idx * 32) * TOPK_BLOCK_SIZE;
              *reinterpret_cast<__int128_t*>(sK_rope_base + smem_offset) = *reinterpret_cast<const __int128_t*>(&rope);
              if constexpr (CLUSTER_SIZE == 2) {
                st_async_128b(sK_rope_peer_base + smem_offset, rope, peer_bar_k_remote_ready);
              }
            }
          }

          fence_view_async_shared();

          if (idx_in_warpgroup < 32) {
            const int pos0 = rel_block_idx * TOPK_BLOCK_SIZE + lane_idx * 2;
            const int pos1 = pos0 + 1;
            const int selected0 = pos0 < topk_length ? __ldg(indices_base + lane_idx * 2) : -1;
            const int selected1 = pos1 < topk_length ? __ldg(indices_base + lane_idx * 2 + 1) : -1;
            *reinterpret_cast<char2*>(&plan.is_kv_valid[buf_idx][lane_idx * 2]) = {
                selected0 >= 0 && static_cast<int64_t>(selected0) < token_capacity,
                selected1 >= 0 && static_cast<int64_t>(selected1) < token_capacity,
            };
          }

          plan.bar_k_local_ready[buf_idx].arrive();
          bar_phase_k ^= 1 << buf_idx;
        };

        CUTE_NO_UNROLL
        for (int block_idx = args.start_block_idx; block_idx < min(args.num_orig_kv_blocks, args.end_block_idx);
             ++block_idx) {
          process_one_block(block_idx, IsOrigBlock{});
        }

        CUTE_NO_UNROLL
        for (int block_idx = max(args.start_block_idx, args.num_orig_kv_blocks); block_idx < args.end_block_idx;
             ++block_idx) {
          process_one_block(block_idx, IsExtraBlock{});
        }

        sync_all_threads_in_cluster();
      }
    }
  }  // !USE_FP8_MMA
#else
  if (cute::thread0()) {
    CUTE_INVALID_CONTROL_PATH("This kernel only supports sm90");
  }
#endif
}

template <typename Kernel, typename TMAParams>
__global__ void __launch_bounds__(Kernel::NUM_THREADS, 1, Kernel::CLUSTER_SIZE)
    flash_fwd_splitkv_mla_mxfp4_dsv4_scaled_sparse_kernel(
        __grid_constant__ const SparseAttnDecodeParams params, __grid_constant__ const TMAParams tma_params) {
  Kernel::devfunc(params, tma_params);
}

template <int NUM_HEADS, bool USE_FP8_MMA>
void KernelTemplate<NUM_HEADS, USE_FP8_MMA>::run(const SparseAttnDecodeParams& params) {
  KU_ASSERT(params.h_kv == 1);
  KU_ASSERT(params.topk % TOPK_BLOCK_SIZE == 0);
  KU_ASSERT(params.extra_topk % TOPK_BLOCK_SIZE == 0);
  KU_ASSERT(params.d_qk == HEAD_DIM_K);
  KU_ASSERT(params.d_v == HEAD_DIM_V);
  KU_ASSERT(params.h_q == NUM_HEADS);
  KU_ASSERT(params.page_block_size > 0);
  KU_ASSERT(params.num_blocks > 0);
  KU_ASSERT(params.kv != nullptr);
  KU_ASSERT(
      params.stride_kv_row == BYTES_PER_TOKEN,
      "Each primary MXFP4 KV row must match the specialization's contiguous byte width");
  if (params.extra_kv != nullptr) {
    KU_ASSERT(params.extra_indices != nullptr);
    KU_ASSERT(params.extra_page_block_size > 0);
    KU_ASSERT(params.extra_num_blocks > 0);
    KU_ASSERT(
        params.stride_extra_kv_row == BYTES_PER_TOKEN,
        "Each extra MXFP4 KV row must contain exactly 368 contiguous bytes");
  } else {
    KU_ASSERT(params.extra_indices == nullptr);
    KU_ASSERT(params.extra_topk == 0);
  }

  auto shape_Q = make_shape(params.h_q, params.d_qk, params.s_q, params.b);
  auto tma_Q = cute::make_tma_copy(
      SM90_TMA_LOAD{},
      make_tensor(
          make_gmem_ptr((bf16*)params.q),
          make_layout(shape_Q, make_stride(params.stride_q_h_q, _1{}, params.stride_q_s_q, params.stride_q_b))),
      SmemLayoutQ{});

  CUtensorMap tensor_map_o;
  {
    // Here we manually construct TMA descriptor to store O, in order to leverage 5D TMA
    uint64_t size[5] = {
        OBUF_SW, (unsigned long)params.h_q, HEAD_DIM_V / OBUF_SW, (unsigned long)params.s_q, (unsigned long)params.b};
    uint64_t stride[4] = {
        params.stride_o_h_q * sizeof(bf16),
        OBUF_SW * sizeof(bf16),
        params.stride_o_s_q * sizeof(bf16),
        params.stride_o_b * sizeof(bf16)};
    uint32_t box_size[5] = {OBUF_SW, BLOCK_M, HEAD_DIM_V / OBUF_SW, 1, 1};
    uint32_t elem_stride[5] = {1, 1, 1, 1, 1};
    CUresult res = CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)(
        &tensor_map_o,
        CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        5,
        params.out,
        size,
        stride,
        box_size,
        elem_stride,
        CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
        OBUF_SW == 64   ? CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B
        : OBUF_SW == 32 ? CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_64B
        : OBUF_SW == 16 ? CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_32B
                        : CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
        CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    KU_ASSERT(res == CUresult::CUDA_SUCCESS);
  }

  TmaParams<decltype(shape_Q), decltype(tma_Q)> tma_params = {shape_Q, tma_Q, tensor_map_o};
  auto mla_kernel = &flash_fwd_splitkv_mla_mxfp4_dsv4_scaled_sparse_kernel<
      KernelTemplate<NUM_HEADS, USE_FP8_MMA>,
      decltype(tma_params)>;

  constexpr size_t smem_size = sizeof(SharedMemoryPlan);
  KU_CUDA_CHECK(cudaFuncSetAttribute(mla_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

  // NOTE Don't use PDL because of potential compiler bugs!
  // cudaLaunchAttribute mla_kernel_attributes[1];
  // mla_kernel_attributes[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  // mla_kernel_attributes[0].val.programmaticStreamSerializationAllowed = 1;
  // cudaLaunchConfig_t mla_kernel_config = {
  //     dim3(num_m_block, params.h_k, params.num_sm_parts),
  //     dim3(NUM_THREADS, 1, 1),
  //     smem_size,
  //     stream,
  //     mla_kernel_attributes,
  //     1
  // };
  // cudaLaunchKernelEx(&mla_kernel_config, mla_kernel, params, tma_params);
  cutlass::ClusterLaunchParams launch_params = {
      dim3(NUM_M_BLOCKS, params.s_q, params.num_sm_parts),
      dim3(NUM_THREADS, 1, 1),
      dim3(CLUSTER_SIZE, 1, 1),
      smem_size,
      params.stream};
  cutlass::launch_kernel_on_cluster(launch_params, (void*)mla_kernel, params, tma_params);
  KU_CHECK_KERNEL_LAUNCH();
}

template <int NUM_HEADS>
void run_flash_splitkv_mla_mxfp4_dsv4_sparse_kernel_impl(const SparseAttnDecodeParams& params) {
  KernelTemplate<NUM_HEADS>::run(params);
}

template <int NUM_HEADS>
void run_flash_splitkv_mla_mxfp4_dsv4_sparse_fp8_mma_kernel_impl(const SparseAttnDecodeParams& params) {
  KernelTemplate<NUM_HEADS, true>::run(params);
}

}  // namespace sm90::decode::sparse_mxfp4_dsv4
