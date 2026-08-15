#include "feasible_map_core.hpp"

#include <algorithm>
#include <limits>

namespace ogc2026_ckernel {
namespace {

constexpr int kMaxH = 29;
constexpr int kMaxW = 179;

std::uint32_t low_bits(int n) {
    if (n <= 0) {
        return 0u;
    }
    if (n >= 32) {
        return std::numeric_limits<std::uint32_t>::max();
    }
    return (std::uint32_t{1} << n) - 1u;
}

int normalize_slice_bound(int value, int dim) {
    if (value < 0) {
        value += dim;
    }
    if (value < 0) {
        return 0;
    }
    if (value > dim) {
        return dim;
    }
    return value;
}

bool normalize_slice(int start, int stop, int dim, int* out_start, int* out_len) {
    if (dim < 0 || out_start == nullptr || out_len == nullptr) {
        return false;
    }
    const int s = normalize_slice_bound(start, dim);
    const int e = normalize_slice_bound(stop, dim);
    *out_start = s;
    *out_len = std::max(0, e - s);
    return true;
}

bool set_error(std::string* error, const char* msg) {
    if (error != nullptr) {
        *error = msg;
    }
    return false;
}

}  // namespace

bool feasible_map_bitset_core(
    int H,
    int W,
    int h,
    int w,
    int nlay,
    int ox,
    int oy,
    int x_lo,
    int x_hi,
    int y_lo,
    int y_hi,
    const std::uint32_t* forbidden_cols,
    int forbidden_layer_stride,
    const std::uint32_t* mask_cols,
    int mask_layer_stride,
    MapResult* out,
    std::string* error) {
    if (out == nullptr) {
        return set_error(error, "null output result");
    }
    *out = MapResult{};

    if (x_lo > x_hi || y_lo > y_hi) {
        out->is_none = true;
        return true;
    }
    if (H <= 0 || H > kMaxH || W <= 0 || W > kMaxW) {
        return set_error(error, "bay dimensions outside H<=29/W<=179 contract");
    }
    if (h <= 0 || w <= 0 || h > H || w > W) {
        return set_error(error, "mask dimensions outside bay bounds");
    }
    if (nlay <= 0) {
        return set_error(error, "nlay must be positive");
    }
    if (forbidden_cols == nullptr || mask_cols == nullptr) {
        return set_error(error, "null packed input columns");
    }
    if (forbidden_layer_stride < W || mask_layer_stride < w) {
        return set_error(error, "packed layer stride too small");
    }

    const int out_rows = y_hi - y_lo + 1;
    const int out_cols = x_hi - x_lo + 1;
    if (out_rows <= 0 || out_cols <= 0) {
        return set_error(error, "invalid output range");
    }

    const int bad_rows = H - h + 1;
    const int bad_cols = W - w + 1;
    if (bad_rows <= 0 || bad_cols <= 0) {
        return set_error(error, "invalid bad-map dimensions");
    }

    out->rows = out_rows;
    out->cols = out_cols;
    out->feasible.assign(static_cast<std::size_t>(out_rows) * out_cols, 1u);

    const std::uint32_t f_row_mask = low_bits(H);
    const std::uint32_t m_row_mask = low_bits(h);

    for (int k = 0; k < nlay; ++k) {
        const std::uint32_t* f_layer =
            forbidden_cols + static_cast<std::size_t>(k) * forbidden_layer_stride;
        const std::uint32_t* m_layer =
            mask_cols + static_cast<std::size_t>(k) * mask_layer_stride;

        bool any_forbidden = false;
        for (int c = 0; c < W; ++c) {
            if ((f_layer[c] & f_row_mask) != 0u) {
                any_forbidden = true;
                break;
            }
        }
        if (!any_forbidden) {
            continue;
        }

        std::vector<std::uint8_t> bad(
            static_cast<std::size_t>(bad_rows) * bad_cols, 0u);

        for (int c = 0; c < bad_cols; ++c) {
            for (int q = 0; q < w; ++q) {
                const std::uint32_t mask_bits = m_layer[q] & m_row_mask;
                if (mask_bits == 0u) {
                    continue;
                }
                const std::uint32_t f_bits = f_layer[c + q] & f_row_mask;
                if (f_bits == 0u) {
                    continue;
                }
                for (int r = 0; r < bad_rows; ++r) {
                    if (((f_bits >> r) & mask_bits) != 0u) {
                        bad[static_cast<std::size_t>(r) * bad_cols + c] = 1u;
                    }
                }
            }
        }

        int rs = 0;
        int cs = 0;
        int slice_rows = 0;
        int slice_cols = 0;
        if (!normalize_slice(y_lo + oy, y_hi + oy + 1,
                             bad_rows, &rs, &slice_rows) ||
            !normalize_slice(x_lo + ox, x_hi + ox + 1,
                             bad_cols, &cs, &slice_cols)) {
            return set_error(error, "failed to normalize Python slice");
        }

        const bool row_ok = (slice_rows == out_rows) || (slice_rows == 1);
        const bool col_ok = (slice_cols == out_cols) || (slice_cols == 1);
        if (!row_ok || !col_ok || slice_rows == 0 || slice_cols == 0) {
            return set_error(error, "bad-map slice is not NumPy-broadcastable");
        }

        for (int r = 0; r < out_rows; ++r) {
            const int br = rs + ((slice_rows == 1) ? 0 : r);
            for (int c = 0; c < out_cols; ++c) {
                const int bc = cs + ((slice_cols == 1) ? 0 : c);
                if (bad[static_cast<std::size_t>(br) * bad_cols + bc] != 0u) {
                    out->feasible[static_cast<std::size_t>(r) * out_cols + c] = 0u;
                }
            }
        }
    }

    return true;
}

}  // namespace ogc2026_ckernel
