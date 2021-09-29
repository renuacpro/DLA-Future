//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2021, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//
#pragma once

#include <blas/axpy.hh>
#include <hpx/local/future.hpp>

#include "dlaf/blas/tile.h"
#include "dlaf/common/index2d.h"
#include "dlaf/common/pipeline.h"
#include "dlaf/common/range2d.h"
#include "dlaf/common/round_robin.h"
#include "dlaf/communication/broadcast_panel.h"
#include "dlaf/communication/communicator.h"
#include "dlaf/communication/communicator_grid.h"
#include "dlaf/communication/executor.h"
#include "dlaf/communication/kernels.h"
#include "dlaf/executors.h"
#include "dlaf/lapack/tile.h"
#include "dlaf/matrix/distribution.h"
#include "dlaf/matrix/extra_buffers.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf/matrix/panel.h"
#include "dlaf/matrix/tile.h"
#include "dlaf/solver/triangular/api.h"
#include "dlaf/util_matrix.h"

namespace dlaf {
namespace solver {
namespace internal {

namespace triangular_lln {
template <class Executor, class T, Device D>
void trsm_B_panel_tile(Executor&& ex, blas::Diag diag, T alpha,
                       hpx::shared_future<matrix::Tile<const T, D>> in_tile,
                       hpx::future<matrix::Tile<T, D>> out_tile) {
  hpx::dataflow(std::forward<Executor>(ex), matrix::unwrapExtendTiles(tile::trsm_o), blas::Side::Left,
                blas::Uplo::Lower, blas::Op::NoTrans, diag, alpha, std::move(in_tile),
                std::move(out_tile));
}

template <class Executor, class T, Device D>
void gemm_trailing_matrix_tile(Executor&& ex, T beta,
                               hpx::shared_future<matrix::Tile<const T, D>> a_tile,
                               hpx::shared_future<matrix::Tile<const T, D>> b_tile,
                               hpx::future<matrix::Tile<T, D>> c_tile) {
  hpx::dataflow(std::forward<Executor>(ex), matrix::unwrapExtendTiles(tile::gemm_o), blas::Op::NoTrans,
                blas::Op::NoTrans, beta, std::move(a_tile), std::move(b_tile), T(1.0),
                std::move(c_tile));
}
}

template <Backend backend, Device device, class T>
void Triangular<backend, device, T>::call_LLN(blas::Diag diag, T alpha, Matrix<const T, device>& mat_a,
                                              Matrix<T, device>& mat_b) {
  using namespace triangular_lln;
  auto executor_hp = dlaf::getHpExecutor<backend>();
  auto executor_np = dlaf::getNpExecutor<backend>();

  SizeType m = mat_b.nrTiles().rows();
  SizeType n = mat_b.nrTiles().cols();

  for (SizeType k = 0; k < m; ++k) {
    for (SizeType j = 0; j < n; ++j) {
      auto kj = LocalTileIndex{k, j};

      // Triangular solve of k-th row Panel of B
      trsm_B_panel_tile(executor_hp, diag, alpha, mat_a.read(LocalTileIndex{k, k}), mat_b(kj));

      for (SizeType i = k + 1; i < m; ++i) {
        // Choose queue priority
        auto& trailing_executor = (i == k + 1) ? executor_hp : executor_np;
        auto beta = static_cast<T>(-1.0) / alpha;
        // Update trailing matrix
        gemm_trailing_matrix_tile(trailing_executor, beta, mat_a.read(LocalTileIndex{i, k}),
                                  mat_b.read(kj), mat_b(LocalTileIndex{i, j}));
      }
    }
  }
}

template <Backend backend, Device device, class T>
void Triangular<backend, device, T>::call_LLT(blas::Op op, blas::Diag diag, T alpha,
                                              Matrix<const T, device>& mat_a, Matrix<T, device>& mat_b) {
  constexpr auto Left = blas::Side::Left;
  constexpr auto Lower = blas::Uplo::Lower;
  constexpr auto NoTrans = blas::Op::NoTrans;

  auto executor_hp = dlaf::getHpExecutor<backend>();
  auto executor_np = dlaf::getNpExecutor<backend>();

  SizeType m = mat_b.nrTiles().rows();
  SizeType n = mat_b.nrTiles().cols();

  for (SizeType k = m - 1; k > -1; --k) {
    for (SizeType j = n - 1; j > -1; --j) {
      auto kj = LocalTileIndex{k, j};
      // Triangular solve of k-th row Panel of B
      hpx::dataflow(executor_hp, matrix::unwrapExtendTiles(tile::trsm_o), Left, Lower, op, diag, alpha,
                    mat_a.read(LocalTileIndex{k, k}), mat_b(kj));

      for (SizeType i = k - 1; i > -1; --i) {
        // Choose queue priority
        auto& trailing_executor = (i == k - 1) ? executor_hp : executor_np;

        auto beta = static_cast<T>(-1.0) / alpha;
        // Update trailing matrix
        hpx::dataflow(trailing_executor, matrix::unwrapExtendTiles(tile::gemm_o), op, NoTrans, beta,
                      mat_a.read(LocalTileIndex{k, i}), mat_b.read(kj), T(1.0),
                      mat_b(LocalTileIndex{i, j}));
      }
    }
  }
}

template <Backend backend, Device device, class T>
void Triangular<backend, device, T>::call_LUN(blas::Diag diag, T alpha, Matrix<const T, device>& mat_a,
                                              Matrix<T, device>& mat_b) {
  constexpr auto Left = blas::Side::Left;
  constexpr auto Upper = blas::Uplo::Upper;
  constexpr auto NoTrans = blas::Op::NoTrans;

  auto executor_hp = dlaf::getHpExecutor<backend>();
  auto executor_np = dlaf::getNpExecutor<backend>();

  SizeType m = mat_b.nrTiles().rows();
  SizeType n = mat_b.nrTiles().cols();

  for (SizeType k = m - 1; k > -1; --k) {
    for (SizeType j = n - 1; j > -1; --j) {
      auto kj = LocalTileIndex{k, j};
      // Triangular solve of k-th row Panel of B
      hpx::dataflow(executor_hp, matrix::unwrapExtendTiles(tile::trsm_o), Left, Upper, NoTrans, diag,
                    alpha, mat_a.read(LocalTileIndex{k, k}), mat_b(kj));

      for (SizeType i = k - 1; i > -1; --i) {
        // Choose queue priority
        auto& trailing_executor = (i == k - 1) ? executor_hp : executor_np;

        auto beta = static_cast<T>(-1.0) / alpha;
        // Update trailing matrix
        hpx::dataflow(trailing_executor, matrix::unwrapExtendTiles(tile::gemm_o), NoTrans, NoTrans, beta,
                      mat_a.read(LocalTileIndex{i, k}), mat_b.read(kj), T(1.0),
                      mat_b(LocalTileIndex{i, j}));
      }
    }
  }
}

template <Backend backend, Device device, class T>
void Triangular<backend, device, T>::call_LUT(blas::Op op, blas::Diag diag, T alpha,
                                              Matrix<const T, device>& mat_a, Matrix<T, device>& mat_b) {
  constexpr auto Left = blas::Side::Left;
  constexpr auto Upper = blas::Uplo::Upper;
  constexpr auto NoTrans = blas::Op::NoTrans;

  auto executor_hp = dlaf::getHpExecutor<backend>();
  auto executor_np = dlaf::getNpExecutor<backend>();

  SizeType m = mat_b.nrTiles().rows();
  SizeType n = mat_b.nrTiles().cols();

  for (SizeType k = 0; k < m; ++k) {
    for (SizeType j = 0; j < n; ++j) {
      auto kj = LocalTileIndex{k, j};

      // Triangular solve of k-th row Panel of B
      hpx::dataflow(executor_hp, matrix::unwrapExtendTiles(tile::trsm_o), Left, Upper, op, diag, alpha,
                    mat_a.read(LocalTileIndex{k, k}), mat_b(kj));

      for (SizeType i = k + 1; i < m; ++i) {
        // Choose queue priority
        auto& trailing_executor = (i == k + 1) ? executor_hp : executor_np;

        auto beta = static_cast<T>(-1.0) / alpha;
        // Update trailing matrix
        hpx::dataflow(trailing_executor, matrix::unwrapExtendTiles(tile::gemm_o), op, NoTrans, beta,
                      mat_a.read(LocalTileIndex{k, i}), mat_b.read(kj), T(1.0),
                      mat_b(LocalTileIndex{i, j}));
      }
    }
  }
}

template <Backend backend, Device device, class T>
void Triangular<backend, device, T>::call_RLN(blas::Diag diag, T alpha, Matrix<const T, device>& mat_a,
                                              Matrix<T, device>& mat_b) {
  constexpr auto Right = blas::Side::Right;
  constexpr auto Lower = blas::Uplo::Lower;
  constexpr auto NoTrans = blas::Op::NoTrans;

  auto executor_hp = dlaf::getHpExecutor<backend>();
  auto executor_np = dlaf::getNpExecutor<backend>();

  SizeType m = mat_b.nrTiles().rows();
  SizeType n = mat_b.nrTiles().cols();

  for (SizeType k = n - 1; k > -1; --k) {
    for (SizeType i = m - 1; i > -1; --i) {
      auto ik = LocalTileIndex{i, k};

      // Triangular solve of k-th col Panel of B
      hpx::dataflow(executor_hp, matrix::unwrapExtendTiles(tile::trsm_o), Right, Lower, NoTrans, diag,
                    alpha, mat_a.read(LocalTileIndex{k, k}), mat_b(ik));

      for (SizeType j = k - 1; j > -1; --j) {
        // Choose queue priority
        auto& trailing_executor = (j == k - 1) ? executor_hp : executor_np;

        auto beta = static_cast<T>(-1.0) / alpha;
        // Update trailing matrix
        hpx::dataflow(trailing_executor, matrix::unwrapExtendTiles(tile::gemm_o), NoTrans, NoTrans, beta,
                      mat_b.read(ik), mat_a.read(LocalTileIndex{k, j}), T(1.0),
                      mat_b(LocalTileIndex{i, j}));
      }
    }
  }
}

template <Backend backend, Device device, class T>
void Triangular<backend, device, T>::call_RLT(blas::Op op, blas::Diag diag, T alpha,
                                              Matrix<const T, device>& mat_a, Matrix<T, device>& mat_b) {
  constexpr auto Right = blas::Side::Right;
  constexpr auto Lower = blas::Uplo::Lower;
  constexpr auto NoTrans = blas::Op::NoTrans;

  auto executor_hp = dlaf::getHpExecutor<backend>();
  auto executor_np = dlaf::getNpExecutor<backend>();

  SizeType m = mat_b.nrTiles().rows();
  SizeType n = mat_b.nrTiles().cols();

  for (SizeType k = 0; k < n; ++k) {
    for (SizeType i = 0; i < m; ++i) {
      auto ik = LocalTileIndex{i, k};

      // Triangular solve of k-th col Panel of B
      hpx::dataflow(executor_hp, matrix::unwrapExtendTiles(tile::trsm_o), Right, Lower, op, diag, alpha,
                    mat_a.read(LocalTileIndex{k, k}), mat_b(ik));

      for (SizeType j = k + 1; j < n; ++j) {
        // Choose queue priority
        auto& trailing_executor = (j == k + 1) ? executor_hp : executor_np;

        auto beta = static_cast<T>(-1.0) / alpha;
        // Update trailing matrix
        hpx::dataflow(trailing_executor, matrix::unwrapExtendTiles(tile::gemm_o), NoTrans, op, beta,
                      mat_b.read(ik), mat_a.read(LocalTileIndex{j, k}), T(1.0),
                      mat_b(LocalTileIndex{i, j}));
      }
    }
  }
}

template <Backend backend, Device device, class T>
void Triangular<backend, device, T>::call_RUN(blas::Diag diag, T alpha, Matrix<const T, device>& mat_a,
                                              Matrix<T, device>& mat_b) {
  constexpr auto Right = blas::Side::Right;
  constexpr auto Upper = blas::Uplo::Upper;
  constexpr auto NoTrans = blas::Op::NoTrans;

  auto executor_hp = dlaf::getHpExecutor<backend>();
  auto executor_np = dlaf::getNpExecutor<backend>();

  SizeType m = mat_b.nrTiles().rows();
  SizeType n = mat_b.nrTiles().cols();

  for (SizeType k = 0; k < n; ++k) {
    for (SizeType i = 0; i < m; ++i) {
      auto ik = LocalTileIndex{i, k};

      // Triangular solve of k-th col Panel of B
      hpx::dataflow(executor_hp, matrix::unwrapExtendTiles(tile::trsm_o), Right, Upper, NoTrans, diag,
                    alpha, mat_a.read(LocalTileIndex{k, k}), mat_b(ik));

      for (SizeType j = k + 1; j < n; ++j) {
        // Choose queue priority
        auto& trailing_executor = (j == k + 1) ? executor_hp : executor_np;

        auto beta = static_cast<T>(-1.0) / alpha;
        // Update trailing matrix
        hpx::dataflow(trailing_executor, matrix::unwrapExtendTiles(tile::gemm_o), NoTrans, NoTrans, beta,
                      mat_b.read(ik), mat_a.read(LocalTileIndex{k, j}), T(1.0),
                      mat_b(LocalTileIndex{i, j}));
      }
    }
  }
}

template <Backend backend, Device device, class T>
void Triangular<backend, device, T>::call_RUT(blas::Op op, blas::Diag diag, T alpha,
                                              Matrix<const T, device>& mat_a, Matrix<T, device>& mat_b) {
  constexpr auto Right = blas::Side::Right;
  constexpr auto Upper = blas::Uplo::Upper;
  constexpr auto NoTrans = blas::Op::NoTrans;

  auto executor_hp = dlaf::getHpExecutor<backend>();
  auto executor_np = dlaf::getNpExecutor<backend>();

  SizeType m = mat_b.nrTiles().rows();
  SizeType n = mat_b.nrTiles().cols();

  for (SizeType k = n - 1; k > -1; --k) {
    for (SizeType i = m - 1; i > -1; --i) {
      auto ik = LocalTileIndex{i, k};

      // Triangular solve of k-th col Panel of B
      hpx::dataflow(executor_hp, matrix::unwrapExtendTiles(tile::trsm_o), Right, Upper, op, diag, alpha,
                    mat_a.read(LocalTileIndex{k, k}), mat_b(ik));

      for (SizeType j = k - 1; j > -1; --j) {
        // Choose queue priority
        auto& trailing_executor = (j == k - 1) ? executor_hp : executor_np;

        auto beta = static_cast<T>(-1.0) / alpha;
        // Update trailing matrix
        hpx::dataflow(trailing_executor, matrix::unwrapExtendTiles(tile::gemm_o), NoTrans, op, beta,
                      mat_b.read(ik), mat_a.read(LocalTileIndex{j, k}), T(1.0),
                      mat_b(LocalTileIndex{i, j}));
      }
    }
  }
}

template <Backend backend, Device device, class T>
void Triangular<backend, device, T>::call_LLN(comm::CommunicatorGrid grid, blas::Diag diag, T alpha,
                                              Matrix<const T, device>& mat_a, Matrix<T, device>& mat_b) {
  using namespace triangular_lln;

  auto executor_hp = dlaf::getHpExecutor<backend>();
  auto executor_np = dlaf::getNpExecutor<backend>();
  auto executor_mpi = dlaf::getMPIExecutor<backend>();

  // Set up MPI executor pipelines
  common::Pipeline<comm::Communicator> mpi_row_task_chain(grid.rowCommunicator());
  common::Pipeline<comm::Communicator> mpi_col_task_chain(grid.colCommunicator());

  const comm::Index2D this_rank = grid.rank();

  const matrix::Distribution& distr_a = mat_a.distribution();
  const matrix::Distribution& distr_b = mat_b.distribution();
  auto a_rows = mat_a.nrTiles().rows();
  auto local_rows = distr_a.localNrTiles().rows();
  auto b_local_cols = distr_b.localNrTiles().cols();

  // If mat_b is empty return immediately
  if (mat_b.size().isEmpty())
    return;

  constexpr std::size_t n_workspaces = 2;
  common::RoundRobin<matrix::Panel<Coord::Col, T, device>> a_panels(n_workspaces, distr_a);
  common::RoundRobin<matrix::Panel<Coord::Row, T, device>> b_panels(n_workspaces, distr_b);

  for (SizeType k = 0; k < a_rows; ++k) {
    const GlobalTileIndex kk(k, k);
    auto kk_rank = distr_a.rankGlobalTile(kk);

    const LocalTileIndex kk_offset{
        distr_a.nextLocalTileFromGlobalTile<Coord::Row>(k),
        distr_a.nextLocalTileFromGlobalTile<Coord::Col>(k),
    };

    const LocalTileIndex bt_offset{distr_b.nextLocalTileFromGlobalTile<Coord::Row>(k + 1), 0};

    auto& a_panel = a_panels.nextResource();
    auto& b_panel = b_panels.nextResource();
    a_panel.setRangeStart(kk);
    if (k == a_rows - 1)
      a_panel.setWidth(mat_a.tileSize(kk).rows());
    b_panel.setHeight(mat_a.tileSize(kk).cols());

    if (kk_rank.col() == this_rank.col()) {
      for (SizeType i_local = kk_offset.row(); i_local < local_rows; ++i_local) {
        const LocalTileIndex ik_panel(Coord::Row, i_local);
        const LocalTileIndex ik(i_local, kk_offset.col());

        a_panel.setTile(ik_panel, mat_a.read(ik));
      }
    }
    broadcast(executor_mpi, kk_rank.col(), a_panel, mpi_row_task_chain);

    for (SizeType j_local = 0; j_local < b_local_cols; ++j_local) {
      // Triangular solve B's k-th row panel and broadcast B(kj) column-wise
      if (kk_rank.row() == this_rank.row()) {
        auto k_local_row = distr_b.localTileFromGlobalTile<Coord::Row>(k);
        const LocalTileIndex kk_panel(Coord::Row, k_local_row);
        const LocalTileIndex kj(k_local_row, j_local);
        const LocalTileIndex kj_panel(Coord::Col, j_local);

        trsm_B_panel_tile(executor_hp, diag, alpha, a_panel.read(kk_panel), mat_b(kj));
        b_panel.setTile(kj_panel, mat_b.read(kj));
      }
    }
    // Nothing else to do if the trailing matrix is empty.
    if (k == a_rows - 1)
      continue;

    broadcast(executor_mpi, kk_rank.row(), b_panel, mpi_col_task_chain);

    for (SizeType i_local = bt_offset.row(); i_local < local_rows; ++i_local) {
      // Choose queue priority
      auto i = distr_a.globalTileFromLocalTile<Coord::Row>(i_local);
      auto& trailing_executor = (i == k + 1) ? executor_hp : executor_np;

      const LocalTileIndex ik_panel(Coord::Row, i_local);

      // Update trailing matrix
      for (SizeType j_local = 0; j_local < b_local_cols; ++j_local) {
        const LocalTileIndex kj_panel(Coord::Col, j_local);
        const LocalTileIndex ij(i_local, j_local);
        const T beta = T(-1.0) / alpha;

        gemm_trailing_matrix_tile(trailing_executor, beta, a_panel.read(ik_panel),
                                  b_panel.read(kj_panel), mat_b(ij));
      }
    }
    a_panel.reset();
    b_panel.reset();
  }
}

/// Note:
/// This is a workaround because the Np/Hp CUDA executors are able to execute
/// cuSolver and cuBlas calls delegating to the respective custom executors, but they
/// do not have as fallback the basic CUDA executor, who is is needed by the set0
/// call.
/// Moreover, since the algorithm is generic for both CPU and GPU, this helper allows to
/// hide the different backends needs.
template <Backend B>
struct getGenericExecutor {
  static auto call() {
    return dlaf::getNpExecutor<Backend::MC>();
  }
};

#ifdef DLAF_WITH_CUDA
template <>
struct getGenericExecutor<Backend::GPU> {
  static auto call() {
    return dlaf::cuda::Executor{dlaf::internal::getNpCudaStreamPool()};
  }
};
#endif

template <class T>
void tileSubtractInPlace(const matrix::Tile<T, Device::CPU>& tile_a,
                         const matrix::Tile<const T, Device::CPU>& tile_b) {
  DLAF_ASSERT(equal_size(tile_a, tile_b), tile_a, tile_b);
  for (auto j = 0; j < tile_a.size().cols(); ++j)
    blas::axpy(tile_a.size().rows(), T(-1), tile_b.ptr({0, j}), 1, tile_a.ptr({0, j}), 1);
}

#ifdef DLAF_WITH_CUDA
template <class T>
void tileSubtractInPlace(cublasHandle_t handle, const matrix::Tile<T, Device::GPU>& tile_a,
                         const matrix::Tile<const T, Device::GPU>& tile_b) {
  DLAF_ASSERT(equal_size(tile_a, tile_b), tile_a, tile_b);
  const T alpha = -1;
  for (auto j = 0; j < tile_a.size().cols(); ++j)
    tile::internal::CublasAxpy<T>::call(handle, tile_a.size().rows(), util::blasToCublasCast(&alpha),
                                        util::blasToCublasCast(tile_b.ptr({0, j})), 1,
                                        util::blasToCublasCast(tile_a.ptr({0, j})), 1);
}
#endif

DLAF_MAKE_CALLABLE_OBJECT(tileSubtractInPlace);

template <Backend backend, Device D, class T>
void Triangular<backend, D, T>::call_LLT(comm::CommunicatorGrid grid, blas::Op op, blas::Diag diag,
                                         T alpha, Matrix<const T, D>& mat_a, Matrix<T, D>& mat_b) {
  auto executor_hp = dlaf::getHpExecutor<backend>();
  auto executor_np = dlaf::getNpExecutor<backend>();
  auto executor_mpi = dlaf::getMPIExecutor<backend>();

  common::Pipeline<comm::Communicator> mpi_row_task_chain(grid.rowCommunicator());
  common::Pipeline<comm::Communicator> mpi_col_task_chain(grid.colCommunicator());

  const comm::Index2D this_rank = grid.rank();

  const matrix::Distribution& distr_a = mat_a.distribution();
  const matrix::Distribution& distr_b = mat_b.distribution();
  // auto a_rows = mat_a.nrTiles().rows();
  // auto local_rows = distr_a.localNrTiles().rows();
  // auto b_local_cols = distr_b.localNrTiles().cols();

  // If mat_b is empty return immediately
  if (mat_b.size().isEmpty())
    return;

  constexpr std::size_t n_workspaces = 2;
  common::RoundRobin<matrix::Panel<Coord::Col, T, D>> a_panels(n_workspaces, distr_a);
  common::RoundRobin<matrix::Panel<Coord::Row, T, D>> b_panels(n_workspaces, distr_b);

  constexpr std::size_t num_extra_buffers = 4;
  std::vector<dlaf::matrix::ExtraBuffers<T>> b_buffers;
  b_buffers.reserve(to_sizet(distr_b.localNrTiles().cols()));
  for (auto j = 0; j < distr_b.localNrTiles().cols(); ++j)
    b_buffers.emplace_back(num_extra_buffers, distr_b.blockSize());

  for (SizeType k = mat_a.nrTiles().cols() - 1; k >= 0; --k) {
    const GlobalTileIndex kk{k, k};

    const LocalTileIndex kk_offset{distr_a.nextLocalTileFromGlobalTile<Coord::Row>(kk.row()),
                                   distr_a.nextLocalTileFromGlobalTile<Coord::Col>(kk.col())};

    auto& a_panel = a_panels.nextResource();
    a_panel.setRangeStart(kk);
    if (kk.row() == mat_a.nrTiles().rows() - 1)
      a_panel.setWidth(mat_a.tileSize(kk).rows());

    const auto rank_kk = distr_a.rankGlobalTile(kk);
    if (this_rank.col() == rank_kk.col()) {
      for (SizeType i_loc = kk_offset.row(); i_loc < distr_a.localNrTiles().rows(); ++i_loc) {
        const LocalTileIndex ik{i_loc, kk_offset.col()};
        a_panel.setTile(ik, mat_a.read(ik));
      }
    }
    comm::broadcast(executor_mpi, rank_kk.col(), a_panel, mpi_row_task_chain);

    auto& b_panel = b_panels.nextResource();
    b_panel.setHeight(mat_a.tileSize(kk).cols());  // TODO just if?

    const LocalTileIndex bt_offset(distr_b.nextLocalTileFromGlobalTile<Coord::Row>(kk.row() + 1), 0);

    // TODO np/hp executor selection in getGenericExecutor
    matrix::util::set0(getGenericExecutor<backend>::call(), b_panel);

    for (auto j = 0; j < distr_b.localNrTiles().cols(); ++j) {
      auto& b_extra = b_buffers.at(to_sizet(j));
      b_extra.setup_base(b_panel(LocalTileIndex(0, j)));
      b_extra.clear();
    }

    for (SizeType j = bt_offset.col(); j < distr_b.localNrTiles().cols(); ++j) {
      for (SizeType i = bt_offset.row(); i < distr_b.localNrTiles().rows(); ++i) {
        const LocalTileIndex ij(i, j);

        const T beta = T(1) / alpha;
        // TODO executor

        auto& b_extra = b_buffers.at(to_sizet(j));
        hpx::dataflow(executor_np, matrix::unwrapExtendTiles(tile::gemm_o), op, blas::Op::NoTrans, beta,
                      a_panel.read(ij), mat_b.read(ij), T(1), b_extra.get_buffer(i));
      }
    }

    for (auto j = 0; j < distr_b.localNrTiles().cols(); ++j) {
      auto& b_extra = b_buffers.at(to_sizet(j));
      b_extra.reduce();
    }

    for (const auto& idx : b_panel.iteratorLocal()) {
      if (this_rank.row() == rank_kk.row()) {
        comm::scheduleReduceRecvInPlace(executor_mpi, mpi_col_task_chain(), MPI_SUM, b_panel(idx));
      }
      else {
        comm::scheduleReduceSend(executor_mpi, rank_kk.row(), mpi_col_task_chain(), MPI_SUM,
                                 b_panel.read(idx));
      }
    }

    if (this_rank.row() == rank_kk.row()) {
      for (SizeType j_loc = 0; j_loc < distr_b.localNrTiles().cols(); ++j_loc) {
        const LocalTileIndex kj(kk_offset.row(), j_loc);
        // TODO executor
        hpx::dataflow(executor_hp, matrix::unwrapExtendTiles(tileSubtractInPlace_o), mat_b(kj),
                      b_panel.read(kj));

        // TODO executor
        hpx::dataflow(executor_hp, matrix::unwrapExtendTiles(tile::trsm_o), blas::Side::Left,
                      blas::Uplo::Lower, op, diag, alpha, a_panel.read(kk_offset), mat_b(kj));
      }
    }

    b_panel.reset();
    a_panel.reset();
  }
}
}
}
}
