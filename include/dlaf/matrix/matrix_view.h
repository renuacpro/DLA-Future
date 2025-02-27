//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2022, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <vector>

#include <blas.hh>
#include <pika/future.hpp>

#include "dlaf/communication/communicator_grid.h"
#include "dlaf/matrix/distribution.h"
#include "dlaf/matrix/layout_info.h"
#include "dlaf/matrix/matrix_base.h"
#include "dlaf/matrix/tile.h"
#include "dlaf/types.h"

namespace dlaf {
namespace matrix {

template <class T, Device device>
class MatrixView;

template <class T, Device device>
class MatrixView<const T, device> : public internal::MatrixBase {
public:
  using ElementType = T;
  using TileType = Tile<ElementType, device>;
  using ConstTileType = Tile<const ElementType, device>;

  template <template <class, Device> class MatrixType, class T2,
            std::enable_if_t<std::is_same_v<T, std::remove_const_t<T2>>, int> = 0>
  MatrixView(blas::Uplo uplo, MatrixType<T2, device>& matrix);

  MatrixView(const MatrixView& rhs) = delete;
  MatrixView(MatrixView&& rhs) = default;

  MatrixView& operator=(const MatrixView& rhs) = delete;
  MatrixView& operator=(MatrixView&& rhs) = default;
  // MatrixView& operator=(MatrixView<T, device>&& rhs);

  /// Returns a read-only shared_future of the Tile with local index @p index.
  ///
  /// TODO: Sync details.
  ///
  /// @pre index.isIn(distribution().localNrTiles()).
  pika::shared_future<ConstTileType> read(const LocalTileIndex& index) noexcept;

  /// Returns a read-only shared_future of the Tile with global index @p index.
  ///
  /// TODO: Sync details.
  ///
  /// @pre index.isIn(globalNrTiles()),
  /// @pre global tile stored in current process.
  pika::shared_future<ConstTileType> read(const GlobalTileIndex& index) noexcept {
    return read(distribution().localTileIndex(index));
  }

  /// Notify that all the operation on the @p index tile has been performed.
  ///
  /// The tile of the original matrix gets ready.
  /// @pre index.isIn(distribution().localNrTiles()),
  /// @post any call to read() with index or the equivalent GlobalTileIndex returns an invalid future.
  void done(const LocalTileIndex& index) noexcept;

  /// Notify that all the operation on the @p index tile has been performed.
  ///
  /// The tile of the original matrix gets ready.
  /// @pre index.isIn(globalNrTiles()),
  /// @post any call to read() with index or the equivalent LocalTileIndex returns an invalid future.
  void done(const GlobalTileIndex& index) noexcept {
    done(distribution().localTileIndex(index));
  }

private:
  template <template <class, Device> class MatrixType, class T2,
            std::enable_if_t<std::is_same_v<T, std::remove_const_t<T2>>, int> = 0>
  void setUpTiles(MatrixType<T2, device>& matrix) noexcept;

  std::vector<pika::shared_future<ConstTileType>> tile_shared_futures_;
};

template <template <class, Device> class MatrixType, class T, Device device>
MatrixView<std::add_const_t<T>, device> getConstView(blas::Uplo uplo, MatrixType<T, device>& matrix) {
  return MatrixView<std::add_const_t<T>, device>(uplo, matrix);
}

template <template <class, Device> class MatrixType, class T, Device device>
MatrixView<std::add_const_t<T>, device> getConstView(MatrixType<T, device>& matrix) {
  return getConstView(blas::Uplo::General, matrix);
}

}
}

#include "dlaf/matrix/matrix_view_const.tpp"
