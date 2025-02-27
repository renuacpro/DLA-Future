//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2022, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "dlaf/blas/enum_output.h"

namespace dlaf {
namespace matrix {

template <class T, Device device>
template <template <class, Device> class MatrixType, class T2,
          std::enable_if_t<std::is_same_v<T, std::remove_const_t<T2>>, int>>
MatrixView<const T, device>::MatrixView(blas::Uplo uplo, MatrixType<T2, device>& matrix)
    : MatrixBase(matrix) {
  if (uplo != blas::Uplo::General)
    DLAF_UNIMPLEMENTED(uplo);
  setUpTiles(matrix);
}

template <class T, Device device>
pika::shared_future<Tile<const T, device>> MatrixView<const T, device>::read(
    const LocalTileIndex& index) noexcept {
  const auto i = tileLinearIndex(index);
  return tile_shared_futures_[i];
}

template <class T, Device device>
void MatrixView<const T, device>::done(const LocalTileIndex& index) noexcept {
  const auto i = tileLinearIndex(index);
  tile_shared_futures_[i] = {};
}

template <class T, Device device>
template <template <class, Device> class MatrixType, class T2,
          std::enable_if_t<std::is_same_v<T, std::remove_const_t<T2>>, int>>
void MatrixView<const T, device>::setUpTiles(MatrixType<T2, device>& matrix) noexcept {
  const auto& nr_tiles = matrix.distribution().localNrTiles();
  tile_shared_futures_.reserve(futureVectorSize(nr_tiles));

  for (SizeType j = 0; j < nr_tiles.cols(); ++j) {
    for (SizeType i = 0; i < nr_tiles.rows(); ++i) {
      LocalTileIndex ind(i, j);
      tile_shared_futures_.emplace_back(matrix.read(ind));
    }
  }
}

}
}
