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

#include <pika/execution.hpp>
#include <pika/future.hpp>
#include <pika/unwrap.hpp>

#include "dlaf/matrix/copy_tile.h"
#include "dlaf/types.h"
#include "dlaf/util_matrix.h"

namespace dlaf {
namespace matrix {

/// Copy values from another matrix.
///
/// Given a matrix with the same geometries and distribution, this function submits tasks that will
/// perform the copy of each tile.
template <class T, Device Source, Device Destination>
void copy(Matrix<const T, Source>& source, Matrix<T, Destination>& dest) {
  const auto& distribution = source.distribution();

  DLAF_ASSERT(matrix::equal_size(source, dest), source, dest);
  DLAF_ASSERT(matrix::equal_blocksize(source, dest), source, dest);
  DLAF_ASSERT(matrix::equal_distributions(source, dest), source, dest);

  const SizeType local_tile_rows = distribution.localNrTiles().rows();
  const SizeType local_tile_cols = distribution.localNrTiles().cols();

  namespace ex = pika::execution::experimental;

  for (SizeType j = 0; j < local_tile_cols; ++j) {
    for (SizeType i = 0; i < local_tile_rows; ++i) {
      ex::when_all(source.read_sender(LocalTileIndex(i, j)),
                   dest.readwrite_sender(LocalTileIndex(i, j))) |
          copy(dlaf::internal::Policy<internal::CopyBackend_v<Source, Destination>>{}) |
          ex::start_detached();
    }
  }
}
}
}
