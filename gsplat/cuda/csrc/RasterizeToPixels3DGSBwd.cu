#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/Atomic.cuh>
#include <c10/cuda/CUDAStream.h>
#include <cooperative_groups.h>

#include "Common.h"
#include "Rasterization.h"
#include "Utils.cuh"

namespace gsplat {

namespace cg = cooperative_groups;

template <
    uint32_t CDIM,
    typename scalar_t,
    KernelT kernel_t = KernelT::GAUSSIAN>
__global__ void rasterize_to_pixels_3dgs_bwd_kernel(
    const uint32_t I,
    const uint32_t N,
    const uint32_t n_isects,
    const bool packed,
    // fwd inputs
    const vec2 *__restrict__ means2d,         // [..., N, 2] or [nnz, 2]
    const vec3 *__restrict__ conics,          // [..., N, 3] or [nnz, 3]
    const scalar_t *__restrict__ colors,      // [..., N, CDIM] or [nnz, CDIM]
    const scalar_t *__restrict__ opacities,   // [..., N] or [nnz]
    const scalar_t *__restrict__ backgrounds, // [..., CDIM] or [nnz, CDIM]
    const bool *__restrict__ masks,           // [..., tile_height, tile_width]
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    const uint32_t tile_width,
    const uint32_t tile_height,
    const int32_t *__restrict__ tile_offsets, // [..., tile_height, tile_width]
    const int32_t *__restrict__ flatten_ids,  // [n_isects]
    // fwd outputs
    const scalar_t
        *__restrict__ render_alphas,      // [..., image_height, image_width, 1]
    const int32_t *__restrict__ last_ids, // [..., image_height, image_width]
    // grad outputs
    const scalar_t *__restrict__ v_render_colors, // [..., image_height,
                                                  // image_width, CDIM]
    const scalar_t
        *__restrict__ v_render_alphas, // [..., image_height, image_width, 1]
    // grad inputs
    vec2 *__restrict__ v_means2d_abs,  // [..., N, 2] or [nnz, 2]
    vec2 *__restrict__ v_means2d,      // [..., N, 2] or [nnz, 2]
    vec3 *__restrict__ v_conics,       // [..., N, 3] or [nnz, 3]
    scalar_t *__restrict__ v_colors,   // [..., N, CDIM] or [nnz, CDIM]
    scalar_t *__restrict__ v_opacities // [..., N] or [nnz]
) {
    auto block = cg::this_thread_block();
    uint32_t image_id = block.group_index().x;
    uint32_t tile_id =
        block.group_index().y * tile_width + block.group_index().z;
    uint32_t i = block.group_index().y * tile_size + block.thread_index().y;
    uint32_t j = block.group_index().z * tile_size + block.thread_index().x;

    tile_offsets += image_id * tile_height * tile_width;
    render_alphas += image_id * image_height * image_width;
    last_ids += image_id * image_height * image_width;
    v_render_colors += image_id * image_height * image_width * CDIM;
    v_render_alphas += image_id * image_height * image_width;
    if (backgrounds != nullptr) {
        backgrounds += image_id * CDIM;
    }
    if (masks != nullptr) {
        masks += image_id * tile_height * tile_width;
    }

    // when the mask is provided, do nothing and return if
    // this tile is labeled as False
    if (masks != nullptr && !masks[tile_id]) {
        return;
    }

    const float px = (float)j + 0.5f;
    const float py = (float)i + 0.5f;
    // clamp this value to the last pixel
    const int32_t pix_id =
        min(i * image_width + j, image_width * image_height - 1);

    // keep not rasterizing threads around for reading data
    bool inside = (i < image_height && j < image_width);

    // have all threads in tile process the same gaussians in batches
    // first collect gaussians between range.x and range.y in batches
    // which gaussians to look through in this tile
    int32_t range_start = tile_offsets[tile_id];
    int32_t range_end =
        (image_id == I - 1) && (tile_id == tile_width * tile_height - 1)
            ? n_isects
            : tile_offsets[tile_id + 1];
    const uint32_t block_size = block.size();
    const uint32_t num_batches =
        (range_end - range_start + block_size - 1) / block_size;

    extern __shared__ int s[];
    int32_t *id_batch = (int32_t *)s; // [block_size]
    vec3 *xy_opacity_batch =
        reinterpret_cast<vec3 *>(&id_batch[block_size]); // [block_size]
    vec3 *conic_batch =
        reinterpret_cast<vec3 *>(&xy_opacity_batch[block_size]); // [block_size]
    float *rgbs_batch =
        (float *)&conic_batch[block_size]; // [block_size * CDIM]

    // this is the T AFTER the last gaussian in this pixel
    float T_final = 1.0f - render_alphas[pix_id];
    float T = T_final;
    // the contribution from gaussians behind the current one
    float buffer[CDIM] = {0.f};
    // index of last gaussian to contribute to this pixel
    const int32_t bin_final = inside ? last_ids[pix_id] : 0;

    // df/d_out for this pixel
    float v_render_c[CDIM];
#pragma unroll
    for (uint32_t k = 0; k < CDIM; ++k) {
        v_render_c[k] = v_render_colors[pix_id * CDIM + k];
    }
    const float v_render_a = v_render_alphas[pix_id];

    // collect and process batches of gaussians
    // each thread loads one gaussian at a time before rasterizing
    const uint32_t tr = block.thread_rank();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    const int32_t warp_bin_final =
        cg::reduce(warp, bin_final, cg::greater<int>());
    for (uint32_t b = 0; b < num_batches; ++b) {
        // resync all threads before writing next batch of shared mem
        block.sync();

        // each thread fetch 1 gaussian from back to front
        // 0 index will be furthest back in batch
        // index of gaussian to load
        // batch end is the index of the last gaussian in the batch
        // These values can be negative so must be int32 instead of uint32
        const int32_t batch_end = range_end - 1 - block_size * b;
        const int32_t batch_size = min(block_size, batch_end + 1 - range_start);
        const int32_t idx = batch_end - tr;
        if (idx >= range_start) {
            int32_t g = flatten_ids[idx]; // flatten index in [I * N] or [nnz]
            id_batch[tr] = g;
            const vec2 xy = means2d[g];
            const float opac = opacities[g];
            xy_opacity_batch[tr] = {xy.x, xy.y, opac};
            conic_batch[tr] = conics[g];
#pragma unroll
            for (uint32_t k = 0; k < CDIM; ++k) {
                rgbs_batch[tr * CDIM + k] = colors[g * CDIM + k];
            }
        }
        // wait for other threads to collect the gaussians in batch
        block.sync();
        // process gaussians in the current batch for this pixel
        // 0 index is the furthest back gaussian in the batch
        for (uint32_t t = max(0, batch_end - warp_bin_final); t < batch_size; ++t) {
            bool valid = inside;
            if (batch_end - t > bin_final) {
                valid = 0;
            }

            // Only initialize variables if valid (no IILE)
            float alpha = 0.f;
            float vis = 0.f;
            float opac = 0.f;
            vec2 delta = {0.f, 0.f};
            vec3 conic = {0.f, 0.f, 0.f};

            if (valid) {
                conic = conic_batch[t];
                const vec3 xy_opac = xy_opacity_batch[t];
                opac = xy_opac.z;
                delta = {xy_opac.x - px, xy_opac.y - py};

                // Common Mahalanobis terms
                const float dx2 = delta.x * delta.x;
                const float dy2 = delta.y * delta.y;
                const float dxdy = delta.x * delta.y;

                static_assert(kernel_t == KernelT::GAUSSIAN || kernel_t == KernelT::EPANECH,
                              "Unknown kernel type for rasterize_to_pixels_3dgs_bwd_kernel");

                if constexpr (kernel_t == KernelT::GAUSSIAN) {
                    const float sigma = 0.5f * (conic.x * dx2 + conic.z * dy2) + conic.y * dxdy;
                    vis = __expf(-sigma);
                    if (sigma < 0.f) {
                        alpha = 0.f;
                    } else {
                        alpha = min(0.999f, opac * vis);
                        if (alpha < ALPHA_THRESHOLD) alpha = 0.f;
                    }
                } else if constexpr (kernel_t == KernelT::EPANECH) {
                    const float u2 = conic.x * dx2 + conic.z * dy2 + 2.0f * conic.y * dxdy;
                    if (u2 > 1.0f) {
                        alpha = 0.f;
                    } else {
                        vis = 0.75f * (1.0f - u2);
                        alpha = min(0.999f, opac * vis);
                        if (alpha < ALPHA_THRESHOLD) alpha = 0.f;
                    }
                }
            }

            if (valid && alpha == 0.0f) {
                valid = false;
            }

            // if all threads are inactive in this warp, skip this loop
            if (!warp.any(valid)) {
                continue;
            }

            float v_rgb_local[CDIM] = {0.f};
            // initialize everything to 0, only set if the lane is valid
            float ra = 1.0f;
            float fac = 0.0f;
            float v_alpha = 0.0f;
            if (valid) {
                // compute the current T for this gaussian
                ra = 1.0f / (1.0f - alpha);
                T *= ra;
                // update v_rgb for this gaussian
                fac = alpha * T;
#pragma unroll
                for (uint32_t k = 0; k < CDIM; ++k) {
                    v_rgb_local[k] = fac * v_render_c[k];
                }
#pragma unroll
                for (uint32_t k = 0; k < CDIM; ++k) {
                    v_alpha += (rgbs_batch[t * CDIM + k] * T - buffer[k] * ra) * v_render_c[k];
                }
                v_alpha += T_final * ra * v_render_a;
                if (backgrounds != nullptr) {
                    float accum = 0.f;
#pragma unroll
                    for (uint32_t k = 0; k < CDIM; ++k) {
                        accum += backgrounds[k] * v_render_c[k];
                    }
                    v_alpha += -T_final * ra * accum;
                }
#pragma unroll
                for (uint32_t k = 0; k < CDIM; ++k) {
                    buffer[k] += rgbs_batch[t * CDIM + k] * fac;
                }
            }

            // Common Mahalanobis terms for gradients
            const float dx2 = delta.x * delta.x;
            const float dy2 = delta.y * delta.y;
            const float dxdy = delta.x * delta.y;

            // Kernel-specific gradient computation (no IILE)
            static_assert(kernel_t == KernelT::GAUSSIAN || kernel_t == KernelT::EPANECH,
                          "Unknown kernel type for rasterize_to_pixels_3dgs_bwd_kernel");

            vec3 v_conic_local = {0.f, 0.f, 0.f};
            vec2 v_xy_local = {0.f, 0.f};
            vec2 v_xy_abs_local = {0.f, 0.f};
            float v_opacity_local = 0.f;

            if (valid) {
                if constexpr (kernel_t == KernelT::GAUSSIAN) {
                    if (opac * vis <= 0.999f) {
                        const float v_sigma = -opac * vis * v_alpha;
                        v_conic_local = {
                            0.5f * v_sigma * dx2,
                            v_sigma * dxdy,
                            0.5f * v_sigma * dy2
                        };
                        v_xy_local = {
                            v_sigma * (conic.x * delta.x + conic.y * delta.y),
                            v_sigma * (conic.y * delta.x + conic.z * delta.y)
                        };
                        v_xy_abs_local = v_means2d_abs != nullptr ? vec2{abs(v_xy_local.x), abs(v_xy_local.y)} : vec2{0.f, 0.f};
                        v_opacity_local = vis * v_alpha;
                    }
                } else if constexpr (kernel_t == KernelT::EPANECH) {
                    if (opac * vis <= 0.999f) {
                        const float v_u2 = -opac * 0.75f * v_alpha;
                        v_conic_local = {
                            v_u2 * dx2,
                            2.0f * v_u2 * dxdy,
                            v_u2 * dy2
                        };
                        v_xy_local = {
                            v_u2 * (2.0f * conic.x * delta.x + 2.0f * conic.y * delta.y),
                            v_u2 * (2.0f * conic.z * delta.y + 2.0f * conic.y * delta.x)
                        };
                        v_xy_abs_local = v_means2d_abs != nullptr ? vec2{abs(v_xy_local.x), abs(v_xy_local.y)} : vec2{0.f, 0.f};
                        v_opacity_local = vis * v_alpha;
                    }
                }
            }

#pragma unroll
            for (uint32_t k = 0; k < CDIM; ++k) {
                buffer[k] += rgbs_batch[t * CDIM + k] * fac;
            }
        }
    }
}

template <uint32_t CDIM>
void launch_rasterize_to_pixels_3dgs_bwd_kernel(
    // Gaussian parameters
    const at::Tensor means2d,                   // [..., N, 2] or [nnz, 2]
    const at::Tensor conics,                    // [..., N, 3] or [nnz, 3]
    const at::Tensor colors,                    // [..., N, 3] or [nnz, 3]
    const at::Tensor opacities,                 // [..., N] or [nnz]
    const at::optional<at::Tensor> backgrounds, // [..., 3]
    const at::optional<at::Tensor> masks, // [..., tile_height, tile_width]
    // image size
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    // intersections
    const at::Tensor tile_offsets, // [..., tile_height, tile_width]
    const at::Tensor flatten_ids,  // [n_isects]
    // forward outputs
    const at::Tensor render_alphas, // [..., image_height, image_width, 1]
    const at::Tensor last_ids,      // [..., image_height, image_width]
    // gradients of outputs
    const at::Tensor v_render_colors, // [..., image_height, image_width, 3]
    const at::Tensor v_render_alphas, // [..., image_height, image_width, 1]
    // outputs
    at::optional<at::Tensor> v_means2d_abs, // [..., N, 2] or [nnz, 2]
    at::Tensor v_means2d,                   // [..., N, 2] or [nnz, 2]
    at::Tensor v_conics,                    // [..., N, 3] or [nnz, 3]
    at::Tensor v_colors,                    // [..., N, 3] or [nnz, 3]
    at::Tensor v_opacities,                 // [..., N] or [nnz]
    const KernelT kernel_t
) {
    bool packed = means2d.dim() == 2;

    uint32_t N = packed ? 0 : means2d.size(-2); // number of gaussians
    uint32_t I = render_alphas.numel() /
                 (image_height * image_width); // number of images
    uint32_t tile_height = tile_offsets.size(-2);
    uint32_t tile_width = tile_offsets.size(-1);
    uint32_t n_isects = flatten_ids.size(0);

    // Each block covers a tile on the image. In total there are
    // I * tile_height * tile_width blocks.
    dim3 threads = {tile_size, tile_size, 1};
    dim3 grid = {I, tile_height, tile_width};

    int64_t shmem_size =
        tile_size * tile_size *
        (sizeof(int32_t) + sizeof(vec3) + sizeof(vec3) + sizeof(float) * CDIM);

    if (n_isects == 0) {
        // skip the kernel launch if there are no elements
        return;
    }

    // TODO: an optimization can be done by passing the actual number of
    // channels into the kernel functions and avoid necessary global memory
    // writes. This requires moving the channel padding from python to C side.
    if (cudaFuncSetAttribute(
            rasterize_to_pixels_3dgs_bwd_kernel<CDIM, float>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            shmem_size
        ) != cudaSuccess) {
        AT_ERROR(
            "Failed to set maximum shared memory size (requested ",
            shmem_size,
            " bytes), try lowering tile_size."
        );
    }

#define CASE_LAUNCH_BWD(KERNEL_T)                                              \
    case KERNEL_T:                                                             \
        rasterize_to_pixels_3dgs_bwd_kernel<CDIM, float, KERNEL_T>             \
            <<<grid, threads, shmem_size, at::cuda::getCurrentCUDAStream()>>>( \
                I,                                                             \
                N,                                                             \
                n_isects,                                                      \
                packed,                                                        \
                reinterpret_cast<vec2 *>(means2d.data_ptr<float>()),           \
                reinterpret_cast<vec3 *>(conics.data_ptr<float>()),            \
                colors.data_ptr<float>(),                                      \
                opacities.data_ptr<float>(),                                   \
                backgrounds.has_value()                                        \
                    ? backgrounds.value().data_ptr<float>()                    \
                    : nullptr,                                                 \
                masks.has_value() ? masks.value().data_ptr<bool>() : nullptr,  \
                image_width,                                                   \
                image_height,                                                  \
                tile_size,                                                     \
                tile_width,                                                    \
                tile_height,                                                   \
                tile_offsets.data_ptr<int32_t>(),                              \
                flatten_ids.data_ptr<int32_t>(),                               \
                render_alphas.data_ptr<float>(),                               \
                last_ids.data_ptr<int32_t>(),                                  \
                v_render_colors.data_ptr<float>(),                             \
                v_render_alphas.data_ptr<float>(),                             \
                v_means2d_abs.has_value()                                      \
                    ? reinterpret_cast<vec2 *>(                                \
                          v_means2d_abs.value().data_ptr<float>()              \
                      )                                                        \
                    : nullptr,                                                 \
                reinterpret_cast<vec2 *>(v_means2d.data_ptr<float>()),         \
                reinterpret_cast<vec3 *>(v_conics.data_ptr<float>()),          \
                v_colors.data_ptr<float>(),                                    \
                v_opacities.data_ptr<float>()                                  \
            );                                                                 \
        break;

    switch (kernel_t) {
        CASE_LAUNCH_BWD(KernelT::GAUSSIAN)
        CASE_LAUNCH_BWD(KernelT::EPANECH)
        default:
            AT_ERROR(
                "Unknown kernel type for rasterize_to_pixels_3dgs_bwd_kernel: ",
                static_cast<int>(kernel_t)
            );
    }

#undef CASE_LAUNCH_BWD
}

// Explicit Instantiation: this should match how it is being called in .cpp
// file.
// TODO: this is slow to compile, can we do something about it?
#define __INS__(CDIM)                                                          \
    template void launch_rasterize_to_pixels_3dgs_bwd_kernel<CDIM>(            \
        const at::Tensor means2d,                                              \
        const at::Tensor conics,                                               \
        const at::Tensor colors,                                               \
        const at::Tensor opacities,                                            \
        const at::optional<at::Tensor> backgrounds,                            \
        const at::optional<at::Tensor> masks,                                  \
        uint32_t image_width,                                                  \
        uint32_t image_height,                                                 \
        uint32_t tile_size,                                                    \
        const at::Tensor tile_offsets,                                         \
        const at::Tensor flatten_ids,                                          \
        const at::Tensor render_alphas,                                        \
        const at::Tensor last_ids,                                             \
        const at::Tensor v_render_colors,                                      \
        const at::Tensor v_render_alphas,                                      \
        at::optional<at::Tensor> v_means2d_abs,                                \
        at::Tensor v_means2d,                                                  \
        at::Tensor v_conics,                                                   \
        at::Tensor v_colors,                                                   \
        at::Tensor v_opacities,                                                \
        const KernelT kernel_t                                                 \
    );

__INS__(1)
__INS__(2)
__INS__(3)
__INS__(4)
__INS__(5)
__INS__(8)
__INS__(9)
__INS__(16)
__INS__(17)
__INS__(32)
__INS__(33)
__INS__(64)
__INS__(65)
__INS__(128)
__INS__(129)
__INS__(256)
__INS__(257)
__INS__(512)
__INS__(513)

#undef __INS__

} // namespace gsplat
