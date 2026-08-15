#ifndef OGC2026_CKERNEL_FEASIBLE_MAP_CORE_HPP
#define OGC2026_CKERNEL_FEASIBLE_MAP_CORE_HPP

#include <cstdint>
#include <string>
#include <vector>

namespace ogc2026_ckernel {

struct MapResult {
    bool is_none = false;
    int rows = 0;
    int cols = 0;
    std::vector<std::uint8_t> feasible;
};

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
    std::string* error);

}  // namespace ogc2026_ckernel

#endif  // OGC2026_CKERNEL_FEASIBLE_MAP_CORE_HPP
