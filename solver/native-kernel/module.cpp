#include "feasible_map_core.hpp"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace {

using BoolArray = py::array_t<bool, py::array::c_style | py::array::forcecast>;

std::vector<std::uint32_t> pack_columns(
    const py::sequence& arrays,
    int nlay,
    int rows,
    int cols,
    const char* label) {
    if (py::len(arrays) != static_cast<py::size_t>(nlay)) {
        throw std::invalid_argument(std::string(label) + " layer count mismatch");
    }
    std::vector<std::uint32_t> packed(
        static_cast<std::size_t>(nlay) * cols, 0u);
    for (int k = 0; k < nlay; ++k) {
        BoolArray arr = py::cast<BoolArray>(arrays[k]);
        if (arr.ndim() != 2 || arr.shape(0) != rows || arr.shape(1) != cols) {
            throw std::invalid_argument(std::string(label) + " shape mismatch");
        }
        auto view = arr.unchecked<2>();
        for (int c = 0; c < cols; ++c) {
            std::uint32_t bits = 0u;
            for (int r = 0; r < rows; ++r) {
                if (view(r, c)) {
                    bits |= (std::uint32_t{1} << r);
                }
            }
            packed[static_cast<std::size_t>(k) * cols + c] = bits;
        }
    }
    return packed;
}

py::object feasible_map_bitset(
    const py::sequence& forbidden_layers,
    const py::sequence& mask_layers,
    int H,
    int W,
    int h,
    int w,
    int ox,
    int oy,
    int x_lo,
    int x_hi,
    int y_lo,
    int y_hi) {
    const int nlay = static_cast<int>(py::len(mask_layers));
    if (nlay <= 0) {
        throw std::invalid_argument("mask_layers must be non-empty");
    }
    std::vector<std::uint32_t> forbidden =
        pack_columns(forbidden_layers, nlay, H, W, "forbidden");
    std::vector<std::uint32_t> masks =
        pack_columns(mask_layers, nlay, h, w, "mask");

    ogc2026_ckernel::MapResult result;
    std::string error;
    bool ok = false;
    {
        py::gil_scoped_release release;
        ok = ogc2026_ckernel::feasible_map_bitset_core(
            H, W, h, w, nlay, ox, oy, x_lo, x_hi, y_lo, y_hi,
            forbidden.data(), W, masks.data(), w, &result, &error);
    }
    if (!ok) {
        throw std::runtime_error(error);
    }
    if (result.is_none) {
        return py::none();
    }

    py::array_t<bool> out({result.rows, result.cols});
    auto view = out.mutable_unchecked<2>();
    for (int r = 0; r < result.rows; ++r) {
        for (int c = 0; c < result.cols; ++c) {
            view(r, c) =
                result.feasible[static_cast<std::size_t>(r) * result.cols + c] != 0u;
        }
    }
    return py::make_tuple(out, x_lo, y_lo);
}

}  // namespace

PYBIND11_MODULE(_ogc2026_ckernel, m) {
    m.doc() = "OGC 2026 bit-packed feasible_map kernel";
    m.attr("__version__") = "0.1.0";
    m.def("feasible_map_bitset", &feasible_map_bitset,
          py::arg("forbidden_layers"),
          py::arg("mask_layers"),
          py::arg("H"),
          py::arg("W"),
          py::arg("h"),
          py::arg("w"),
          py::arg("ox"),
          py::arg("oy"),
          py::arg("x_lo"),
          py::arg("x_hi"),
          py::arg("y_lo"),
          py::arg("y_hi"));
}
