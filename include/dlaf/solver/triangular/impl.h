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
#include <pika/thread.hpp>

#include "dlaf/blas/tile.h"
#include "dlaf/blas/tile_extensions.h"
#include "dlaf/common/index2d.h"
#include "dlaf/common/pipeline.h"
#include "dlaf/common/range2d.h"
#include "dlaf/common/round_robin.h"
#include "dlaf/communication/broadcast_panel.h"
#include "dlaf/communication/communicator.h"
#include "dlaf/communication/communicator_grid.h"
#include "dlaf/communication/kernels.h"
#include "dlaf/lapack/tile.h"
#include "dlaf/matrix/distribution.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf/matrix/panel.h"
#include "dlaf/matrix/tile.h"
#include "dlaf/sender/when_all_lift.h"
#include "dlaf/solver/triangular/api.h"
#include "dlaf/util_matrix.h"

namespace dlaf {
namespace solver {
namespace internal {
namespace triangular_lln {
template <Backend backend, class T, typename InSender, typename OutSender>
void trsmBPanelTile(pika::threads::thread_priority priority, blas::Diag diag, T alpha,
                    InSender&& in_tile, OutSender&& out_tile) {
  dlaf::internal::whenAllLift(blas::Side::Left, blas::Uplo::Lower, blas::Op::NoTrans, diag, alpha,
                              std::forward<InSender>(in_tile), std::forward<OutSender>(out_tile)) |
      tile::trsm(dlaf::internal::Policy<backend>(priority)) |
      pika::execution::experimental::start_detached();
}

template <Backend backend, class T, typename ASender, typename BSender, typename CSender>
void gemmTrailingMatrixTile(pika::threads::thread_priority priority, T beta, ASender&& a_tile,
                            BSender&& b_tile, CSender&& c_tile) {
  dlaf::internal::whenAllLift(blas::Op::NoTrans, blas::Op::NoTrans, beta, std::forward<ASender>(a_tile),
                              std::forward<BSender>(b_tile), T(1.0), std::forward<CSender>(c_tile)) |
      tile::gemm(dlaf::internal::Policy<backend>(priority)) |
      pika::execution::experimental::start_detached();
}
}

namespace triangular_llt {
template <Backend backend, class T, typename InSender, typename OutSender>
void trsmBPanelTile(pika::threads::thread_priority priority, blas::Op op, blas::Diag diag, T alpha,
                    InSender&& in_tile, OutSender&& out_tile) {
  dlaf::internal::whenAllLift(blas::Side::Left, blas::Uplo::Lower, op, diag, alpha,
                              std::forward<InSender>(in_tile), std::forward<OutSender>(out_tile)) |
      tile::trsm(dlaf::internal::Policy<backend>(priority)) |
      pika::execution::experimental::start_detached();
}

template <Backend backend, class T, typename ASender, typename BSender, typename CSender>
void gemmTrailingMatrixTile(pika::threads::thread_priority priority, blas::Op op, T beta,
                            ASender&& a_tile, BSender&& b_tile, CSender&& c_tile) {
  dlaf::internal::whenAllLift(op, blas::Op::NoTrans, beta, std::forward<ASender>(a_tile),
                              std::forward<BSender>(b_tile), T(1.0), std::forward<CSender>(c_tile)) |
      tile::gemm(dlaf::internal::Policy<backend>(priority)) |
      pika::execution::experimental::start_detached();
}
}

namespace triangular_lun {
template <Backend backend, class T, typename InSender, typename OutSender>
void trsmBPanelTile(pika::threads::thread_priority priority, blas::Diag diag, T alpha,
                    InSender&& in_tile, OutSender&& out_tile) {
  dlaf::internal::whenAllLift(blas::Side::Left, blas::Uplo::Upper, blas::Op::NoTrans, diag, alpha,
                              std::forward<InSender>(in_tile), std::forward<OutSender>(out_tile)) |
      tile::trsm(dlaf::internal::Policy<backend>(priority)) |
      pika::execution::experimental::start_detached();
}

template <Backend backend, class T, typename ASender, typename BSender, typename CSender>
void gemmTrailingMatrixTile(pika::threads::thread_priority priority, T beta, ASender&& a_tile,
                            BSender&& b_tile, CSender&& c_tile) {
  dlaf::internal::whenAllLift(blas::Op::NoTrans, blas::Op::NoTrans, beta, std::forward<ASender>(a_tile),
                              std::forward<BSender>(b_tile), T(1.0), std::forward<CSender>(c_tile)) |
      tile::gemm(dlaf::internal::Policy<backend>(priority)) |
      pika::execution::experimental::start_detached();
}
}

namespace triangular_lut {
template <Backend backend, class T, typename InSender, typename OutSender>
void trsmBPanelTile(pika::threads::thread_priority priority, blas::Op op, blas::Diag diag, T alpha,
                    InSender&& in_tile, OutSender&& out_tile) {
  dlaf::internal::whenAllLift(blas::Side::Left, blas::Uplo::Upper, op, diag, alpha,
                              std::forward<InSender>(in_tile), std::forward<OutSender>(out_tile)) |
      tile::trsm(dlaf::internal::Policy<backend>(priority)) |
      pika::execution::experimental::start_detached();
}

template <Backend backend, class T, typename ASender, typename BSender, typename CSender>
void gemmTrailingMatrixTile(pika::threads::thread_priority priority, blas::Op op, T beta,
                            ASender&& a_tile, BSender&& b_tile, CSender&& c_tile) {
  dlaf::internal::whenAllLift(op, blas::Op::NoTrans, beta, std::forward<ASender>(a_tile),
                              std::forward<BSender>(b_tile), T(1.0), std::forward<CSender>(c_tile)) |
      tile::gemm(dlaf::internal::Policy<backend>(priority)) |
      pika::execution::experimental::start_detached();
}
}

namespace triangular_rln {
template <Backend backend, class T, typename InSender, typename OutSender>
void trsmBPanelTile(pika::threads::thread_priority priority, blas::Diag diag, T alpha,
                    InSender&& in_tile, OutSender&& out_tile) {
  dlaf::internal::whenAllLift(blas::Side::Right, blas::Uplo::Lower, blas::Op::NoTrans, diag, alpha,
                              std::forward<InSender>(in_tile), std::forward<OutSender>(out_tile)) |
      tile::trsm(dlaf::internal::Policy<backend>(priority)) |
      pika::execution::experimental::start_detached();
}

template <Backend backend, class T, typename ASender, typename BSender, typename CSender>
void gemmTrailingMatrixTile(pika::threads::thread_priority priority, T beta, ASender&& a_tile,
                            BSender&& b_tile, CSender&& c_tile) {
  dlaf::internal::whenAllLift(blas::Op::NoTrans, blas::Op::NoTrans, beta, std::forward<ASender>(a_tile),
                              std::forward<BSender>(b_tile), T(1.0), std::forward<CSender>(c_tile)) |
      tile::gemm(dlaf::internal::Policy<backend>(priority)) |
      pika::execution::experimental::start_detached();
}
}

namespace triangular_rlt {
template <Backend backend, class T, typename InSender, typename OutSender>
void trsmBPanelTile(pika::threads::thread_priority priority, blas::Op op, blas::Diag diag, T alpha,
                    InSender&& in_tile, OutSender&& out_tile) {
  dlaf::internal::whenAllLift(blas::Side::Right, blas::Uplo::Lower, op, diag, alpha,
                              std::forward<InSender>(in_tile), std::forward<OutSender>(out_tile)) |
      tile::trsm(dlaf::internal::Policy<backend>(priority)) |
      pika::execution::experimental::start_detached();
}

template <Backend backend, class T, typename ASender, typename BSender, typename CSender>
void gemmTrailingMatrixTile(pika::threads::thread_priority priority, blas::Op op, T beta,
                            ASender&& a_tile, BSender&& b_tile, CSender&& c_tile) {
  dlaf::internal::whenAllLift(blas::Op::NoTrans, op, beta, std::forward<ASender>(a_tile),
                              std::forward<BSender>(b_tile), T(1.0), std::forward<CSender>(c_tile)) |
      tile::gemm(dlaf::internal::Policy<backend>(priority)) |
      pika::execution::experimental::start_detached();
}
}

namespace triangular_run {
template <Backend backend, class T, typename InSender, typename OutSender>
void trsmBPanelTile(pika::threads::thread_priority priority, blas::Diag diag, T alpha,
                    InSender&& in_tile, OutSender&& out_tile) {
  dlaf::internal::whenAllLift(blas::Side::Right, blas::Uplo::Upper, blas::Op::NoTrans, diag, alpha,
                              std::forward<InSender>(in_tile), std::forward<OutSender>(out_tile)) |
      tile::trsm(dlaf::internal::Policy<backend>(priority)) |
      pika::execution::experimental::start_detached();
}

template <Backend backend, class T, typename ASender, typename BSender, typename CSender>
void gemmTrailingMatrixTile(pika::threads::thread_priority priority, T beta, ASender&& a_tile,
                            BSender&& b_tile, CSender&& c_tile) {
  dlaf::internal::whenAllLift(blas::Op::NoTrans, blas::Op::NoTrans, beta, std::forward<ASender>(a_tile),
                              std::forward<BSender>(b_tile), T(1.0), std::forward<CSender>(c_tile)) |
      tile::gemm(dlaf::internal::Policy<backend>(priority)) |
      pika::execution::experimental::start_detached();
}
}

namespace triangular_rut {
template <Backend backend, class T, typename InSender, typename OutSender>
void trsmBPanelTile(pika::threads::thread_priority priority, blas::Op op, blas::Diag diag, T alpha,
                    InSender&& in_tile, OutSender&& out_tile) {
  dlaf::internal::whenAllLift(blas::Side::Right, blas::Uplo::Upper, op, diag, alpha,
                              std::forward<InSender>(in_tile), std::forward<OutSender>(out_tile)) |
      tile::trsm(dlaf::internal::Policy<backend>(priority)) |
      pika::execution::experimental::start_detached();
}

template <Backend backend, class T, typename ASender, typename BSender, typename CSender>
void gemmTrailingMatrixTile(pika::threads::thread_priority priority, blas::Op op, T beta,
                            ASender&& a_tile, BSender&& b_tile, CSender&& c_tile) {
  dlaf::internal::whenAllLift(blas::Op::NoTrans, op, beta, std::forward<ASender>(a_tile),
                              std::forward<BSender>(b_tile), T(1.0), std::forward<CSender>(c_tile)) |
      tile::gemm(dlaf::internal::Policy<backend>(priority)) |
      pika::execution::experimental::start_detached();
}
}

template <Backend backend, Device device, class T>
void Triangular<backend, device, T>::call_LLN(blas::Diag diag, T alpha, Matrix<const T, device>& mat_a,
                                              Matrix<T, device>& mat_b) {
  using namespace triangular_lln;
  using pika::threads::thread_priority;

  SizeType m = mat_b.nrTiles().rows();
  SizeType n = mat_b.nrTiles().cols();

  for (SizeType k = 0; k < m; ++k) {
    for (SizeType j = 0; j < n; ++j) {
      auto kj = LocalTileIndex{k, j};

      // Triangular solve of k-th row Panel of B
      trsmBPanelTile<backend>(thread_priority::high, diag, alpha,
                              mat_a.read_sender(LocalTileIndex{k, k}), mat_b.readwrite_sender(kj));

      for (SizeType i = k + 1; i < m; ++i) {
        // Choose queue priority
        const auto trailing_priority = (i == k + 1) ? thread_priority::high : thread_priority::normal;

        auto beta = static_cast<T>(-1.0) / alpha;
        // Update trailing matrix
        gemmTrailingMatrixTile<backend>(trailing_priority, beta, mat_a.read_sender(LocalTileIndex{i, k}),
                                        mat_b.read_sender(kj),
                                        mat_b.readwrite_sender(LocalTileIndex{i, j}));
      }
    }
  }
}

template <Backend backend, Device device, class T>
void Triangular<backend, device, T>::call_LLT(blas::Op op, blas::Diag diag, T alpha,
                                              Matrix<const T, device>& mat_a, Matrix<T, device>& mat_b) {
  using namespace triangular_llt;
  using pika::threads::thread_priority;

  SizeType m = mat_b.nrTiles().rows();
  SizeType n = mat_b.nrTiles().cols();

  for (SizeType k = m - 1; k >= 0; --k) {
    for (SizeType j = n - 1; j >= 0; --j) {
      auto kj = LocalTileIndex{k, j};
      // Triangular solve of k-th row Panel of B
      trsmBPanelTile<backend>(thread_priority::high, op, diag, alpha,
                              mat_a.read_sender(LocalTileIndex{k, k}), mat_b.readwrite_sender(kj));

      for (SizeType i = k - 1; i >= 0; --i) {
        // Choose queue priority
        const auto trailing_priority = (i == k - 1) ? thread_priority::high : thread_priority::normal;

        auto beta = static_cast<T>(-1.0) / alpha;

        // Update trailing matrix
        gemmTrailingMatrixTile<backend>(trailing_priority, op, beta,
                                        mat_a.read_sender(LocalTileIndex{k, i}), mat_b.read_sender(kj),
                                        mat_b.readwrite_sender(LocalTileIndex{i, j}));
      }
    }
  }
}

template <Backend backend, Device device, class T>
void Triangular<backend, device, T>::call_LUN(blas::Diag diag, T alpha, Matrix<const T, device>& mat_a,
                                              Matrix<T, device>& mat_b) {
  using namespace triangular_lun;
  using pika::threads::thread_priority;

  SizeType m = mat_b.nrTiles().rows();
  SizeType n = mat_b.nrTiles().cols();

  for (SizeType k = m - 1; k >= 0; --k) {
    for (SizeType j = n - 1; j >= 0; --j) {
      auto kj = LocalTileIndex{k, j};
      // Triangular solve of k-th row Panel of B
      trsmBPanelTile<backend>(thread_priority::high, diag, alpha,
                              mat_a.read_sender(LocalTileIndex{k, k}), mat_b.readwrite_sender(kj));

      for (SizeType i = k - 1; i >= 0; --i) {
        // Choose queue priority
        const auto trailing_priority = (i == k - 1) ? thread_priority::high : thread_priority::normal;
        auto beta = static_cast<T>(-1.0) / alpha;
        // Update trailing matrix
        gemmTrailingMatrixTile<backend>(trailing_priority, beta, mat_a.read_sender(LocalTileIndex{i, k}),
                                        mat_b.read_sender(kj),
                                        mat_b.readwrite_sender(LocalTileIndex{i, j}));
      }
    }
  }
}

template <Backend backend, Device device, class T>
void Triangular<backend, device, T>::call_LUT(blas::Op op, blas::Diag diag, T alpha,
                                              Matrix<const T, device>& mat_a, Matrix<T, device>& mat_b) {
  using namespace triangular_lut;
  using pika::threads::thread_priority;

  SizeType m = mat_b.nrTiles().rows();
  SizeType n = mat_b.nrTiles().cols();

  for (SizeType k = 0; k < m; ++k) {
    for (SizeType j = 0; j < n; ++j) {
      auto kj = LocalTileIndex{k, j};

      // Triangular solve of k-th row Panel of B
      trsmBPanelTile<backend>(thread_priority::high, op, diag, alpha,
                              mat_a.read_sender(LocalTileIndex{k, k}), mat_b.readwrite_sender(kj));

      for (SizeType i = k + 1; i < m; ++i) {
        // Choose queue priority
        const auto trailing_priority = (i == k + 1) ? thread_priority::high : thread_priority::normal;

        auto beta = static_cast<T>(-1.0) / alpha;
        // Update trailing matrix
        gemmTrailingMatrixTile<backend>(trailing_priority, op, beta,
                                        mat_a.read_sender(LocalTileIndex{k, i}), mat_b.read_sender(kj),
                                        mat_b.readwrite_sender(LocalTileIndex{i, j}));
      }
    }
  }
}

template <Backend backend, Device device, class T>
void Triangular<backend, device, T>::call_RLN(blas::Diag diag, T alpha, Matrix<const T, device>& mat_a,
                                              Matrix<T, device>& mat_b) {
  using namespace triangular_rln;
  using pika::threads::thread_priority;

  SizeType m = mat_b.nrTiles().rows();
  SizeType n = mat_b.nrTiles().cols();

  for (SizeType k = n - 1; k >= 0; --k) {
    for (SizeType i = m - 1; i >= 0; --i) {
      auto ik = LocalTileIndex{i, k};

      // Triangular solve of k-th col Panel of B
      trsmBPanelTile<backend>(thread_priority::high, diag, alpha,
                              mat_a.read_sender(LocalTileIndex{k, k}), mat_b.readwrite_sender(ik));

      for (SizeType j = k - 1; j >= 0; --j) {
        // Choose queue priority
        const auto trailing_priority = (j == k - 1) ? thread_priority::high : thread_priority::normal;
        auto beta = static_cast<T>(-1.0) / alpha;
        // Update trailing matrix
        gemmTrailingMatrixTile<backend>(trailing_priority, beta, mat_b.read_sender(ik),
                                        mat_a.read_sender(LocalTileIndex{k, j}),
                                        mat_b.readwrite_sender(LocalTileIndex{i, j}));
      }
    }
  }
}

template <Backend backend, Device device, class T>
void Triangular<backend, device, T>::call_RLT(blas::Op op, blas::Diag diag, T alpha,
                                              Matrix<const T, device>& mat_a, Matrix<T, device>& mat_b) {
  using namespace triangular_rlt;
  using pika::threads::thread_priority;

  SizeType m = mat_b.nrTiles().rows();
  SizeType n = mat_b.nrTiles().cols();

  for (SizeType k = 0; k < n; ++k) {
    for (SizeType i = 0; i < m; ++i) {
      auto ik = LocalTileIndex{i, k};

      // Triangular solve of k-th col Panel of B
      trsmBPanelTile<backend>(thread_priority::high, op, diag, alpha,
                              mat_a.read_sender(LocalTileIndex{k, k}), mat_b.readwrite_sender(ik));

      for (SizeType j = k + 1; j < n; ++j) {
        // Choose queue priority
        const auto trailing_priority = (j == k + 1) ? thread_priority::high : thread_priority::normal;

        auto beta = static_cast<T>(-1.0) / alpha;
        // Update trailing matrix
        gemmTrailingMatrixTile<backend>(trailing_priority, op, beta, mat_b.read_sender(ik),
                                        mat_a.read_sender(LocalTileIndex{j, k}),
                                        mat_b.readwrite_sender(LocalTileIndex{i, j}));
      }
    }
  }
}

template <Backend backend, Device device, class T>
void Triangular<backend, device, T>::call_RUN(blas::Diag diag, T alpha, Matrix<const T, device>& mat_a,
                                              Matrix<T, device>& mat_b) {
  using namespace triangular_run;
  using pika::threads::thread_priority;

  SizeType m = mat_b.nrTiles().rows();
  SizeType n = mat_b.nrTiles().cols();

  for (SizeType k = 0; k < n; ++k) {
    for (SizeType i = 0; i < m; ++i) {
      auto ik = LocalTileIndex{i, k};

      // Triangular solve of k-th col Panel of B
      trsmBPanelTile<backend>(thread_priority::high, diag, alpha,
                              mat_a.read_sender(LocalTileIndex{k, k}), mat_b.readwrite_sender(ik));

      for (SizeType j = k + 1; j < n; ++j) {
        // Choose queue priority
        const auto trailing_priority = (j == k + 1) ? thread_priority::high : thread_priority::normal;
        auto beta = static_cast<T>(-1.0) / alpha;
        // Update trailing matrix
        gemmTrailingMatrixTile<backend>(trailing_priority, beta, mat_b.read_sender(ik),
                                        mat_a.read_sender(LocalTileIndex{k, j}),
                                        mat_b.readwrite_sender(LocalTileIndex{i, j}));
      }
    }
  }
}

template <Backend backend, Device device, class T>
void Triangular<backend, device, T>::call_RUT(blas::Op op, blas::Diag diag, T alpha,
                                              Matrix<const T, device>& mat_a, Matrix<T, device>& mat_b) {
  using namespace triangular_rut;
  using pika::threads::thread_priority;

  SizeType m = mat_b.nrTiles().rows();
  SizeType n = mat_b.nrTiles().cols();

  for (SizeType k = n - 1; k >= 0; --k) {
    for (SizeType i = m - 1; i >= 0; --i) {
      auto ik = LocalTileIndex{i, k};

      // Triangular solve of k-th col Panel of B
      trsmBPanelTile<backend>(thread_priority::high, op, diag, alpha,
                              mat_a.read_sender(LocalTileIndex{k, k}), mat_b.readwrite_sender(ik));

      for (SizeType j = k - 1; j >= 0; --j) {
        // Choose queue priority
        const auto trailing_priority = (j == k - 1) ? thread_priority::high : thread_priority::normal;

        auto beta = static_cast<T>(-1.0) / alpha;
        // Update trailing matrix
        gemmTrailingMatrixTile<backend>(trailing_priority, op, beta, mat_b.read_sender(ik),
                                        mat_a.read_sender(LocalTileIndex{j, k}),
                                        mat_b.readwrite_sender(LocalTileIndex{i, j}));
      }
    }
  }
}

template <Backend backend, Device device, class T>
void Triangular<backend, device, T>::call_LLN(comm::CommunicatorGrid grid, blas::Diag diag, T alpha,
                                              Matrix<const T, device>& mat_a, Matrix<T, device>& mat_b) {
  using namespace triangular_lln;
  using pika::threads::thread_priority;

  using common::internal::vector;

  // Set up MPI executor pipelines
  common::Pipeline<comm::Communicator> mpi_row_task_chain(grid.rowCommunicator().clone());
  common::Pipeline<comm::Communicator> mpi_col_task_chain(grid.colCommunicator().clone());

  const comm::Index2D this_rank = grid.rank();

  const matrix::Distribution& distr_a = mat_a.distribution();
  const matrix::Distribution& distr_b = mat_b.distribution();

  // If mat_b is empty return immediately
  if (mat_b.size().isEmpty())
    return;

  constexpr std::size_t n_workspaces = 2;
  common::RoundRobin<matrix::Panel<Coord::Col, T, device>> a_panels(n_workspaces, distr_a);
  common::RoundRobin<matrix::Panel<Coord::Row, T, device>> b_panels(n_workspaces, distr_b);

  for (SizeType k = 0; k < mat_a.nrTiles().rows(); ++k) {
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
    if (k == mat_a.nrTiles().rows() - 1) {
      a_panel.setWidth(mat_a.tileSize(kk).rows());
      b_panel.setHeight(mat_a.tileSize(kk).cols());
    }

    if (kk_rank.col() == this_rank.col()) {
      for (SizeType i_local = kk_offset.row(); i_local < distr_a.localNrTiles().rows(); ++i_local) {
        const LocalTileIndex ik_panel(Coord::Row, i_local);
        const LocalTileIndex ik(i_local, kk_offset.col());
        a_panel.setTile(ik_panel, mat_a.read(ik));
      }
    }
    broadcast(kk_rank.col(), a_panel, mpi_row_task_chain);

    for (SizeType j_local = 0; j_local < distr_b.localNrTiles().cols(); ++j_local) {
      // Triangular solve B's k-th row panel and broadcast B(kj) column-wise
      if (kk_rank.row() == this_rank.row()) {
        auto k_local_row = distr_b.localTileFromGlobalTile<Coord::Row>(k);
        const LocalTileIndex kk_panel(Coord::Row, k_local_row);
        const LocalTileIndex kj(k_local_row, j_local);
        const LocalTileIndex kj_panel(Coord::Col, j_local);

        trsmBPanelTile<backend>(thread_priority::high, diag, alpha, a_panel.read_sender(kk_panel),
                                mat_b.readwrite_sender(kj));
        b_panel.setTile(kj_panel, mat_b.read(kj));
      }
    }
    // Nothing else to do if the trailing matrix is empty.
    if (k == mat_a.nrTiles().rows() - 1)
      continue;

    broadcast(kk_rank.row(), b_panel, mpi_col_task_chain);

    for (SizeType i_local = bt_offset.row(); i_local < distr_a.localNrTiles().rows(); ++i_local) {
      // Choose queue priority
      auto i = distr_a.globalTileFromLocalTile<Coord::Row>(i_local);
      const auto trailing_priority = (i == k + 1) ? thread_priority::high : thread_priority::normal;

      const LocalTileIndex ik_panel(Coord::Row, i_local);

      // Update trailing matrix
      for (SizeType j_local = 0; j_local < distr_b.localNrTiles().cols(); ++j_local) {
        const LocalTileIndex kj_panel(Coord::Col, j_local);
        const LocalTileIndex ij(i_local, j_local);
        const T beta = T(-1.0) / alpha;

        gemmTrailingMatrixTile<backend>(trailing_priority, beta, a_panel.read_sender(ik_panel),
                                        b_panel.read_sender(kj_panel), mat_b.readwrite_sender(ij));
      }
    }
    a_panel.reset();
    b_panel.reset();
  }
}

template <Backend backend, Device D, class T>
void Triangular<backend, D, T>::call_LLT(comm::CommunicatorGrid grid, blas::Op op, blas::Diag diag,
                                         T alpha, Matrix<const T, D>& mat_a, Matrix<T, D>& mat_b) {
  using namespace triangular_llt;
  using pika::threads::thread_priority;

  common::Pipeline<comm::Communicator> mpi_row_task_chain(grid.rowCommunicator().clone());
  common::Pipeline<comm::Communicator> mpi_col_task_chain(grid.colCommunicator().clone());

  const comm::Index2D this_rank = grid.rank();

  const matrix::Distribution& distr_a = mat_a.distribution();
  const matrix::Distribution& distr_b = mat_b.distribution();

  if (mat_b.size().isEmpty())
    return;

  constexpr std::size_t n_workspaces = 2;
  common::RoundRobin<matrix::Panel<Coord::Col, T, D>> a_panels(n_workspaces, distr_a);
  common::RoundRobin<matrix::Panel<Coord::Row, T, D>> b_panels(n_workspaces, distr_b);

  for (SizeType k = mat_a.nrTiles().cols() - 1; k >= 0; --k) {
    const GlobalTileIndex kk{k, k};

    const LocalTileIndex kk_offset{distr_a.nextLocalTileFromGlobalTile<Coord::Row>(kk.row()),
                                   distr_a.nextLocalTileFromGlobalTile<Coord::Col>(kk.col())};
    const LocalTileIndex bt_offset(distr_b.nextLocalTileFromGlobalTile<Coord::Row>(kk.row() + 1), 0);

    auto& a_panel = a_panels.nextResource();
    auto& b_panel = b_panels.nextResource();

    a_panel.setRangeStart(kk);

    if (kk.row() == mat_a.nrTiles().rows() - 1) {
      a_panel.setWidth(mat_a.tileSize(kk).cols());
      b_panel.setHeight(mat_a.tileSize(kk).rows());
    }

    const auto rank_kk = distr_a.rankGlobalTile(kk);
    if (this_rank.col() == rank_kk.col()) {
      for (SizeType i_loc = kk_offset.row(); i_loc < distr_a.localNrTiles().rows(); ++i_loc) {
        const LocalTileIndex ik{i_loc, kk_offset.col()};
        a_panel.setTile(ik, mat_a.read(ik));
      }
    }
    comm::broadcast(rank_kk.col(), a_panel, mpi_row_task_chain);

    matrix::util::set0<backend>(thread_priority::normal, b_panel);

    for (const auto& ij : common::iterate_range2d(bt_offset, indexFromOrigin(distr_b.localNrTiles())))
      gemmTrailingMatrixTile<backend>(ij.row() == bt_offset.row() ? thread_priority::high
                                                                  : thread_priority::normal,
                                      op, T(1) / alpha, a_panel.read_sender(ij), mat_b.read_sender(ij),
                                      b_panel.readwrite_sender(ij));

    for (const auto& idx : b_panel.iteratorLocal()) {
      if (this_rank.row() == rank_kk.row())
        comm::scheduleReduceRecvInPlace(mpi_col_task_chain(), MPI_SUM, b_panel(idx));
      else
        comm::scheduleReduceSend(rank_kk.row(), mpi_col_task_chain(), MPI_SUM, b_panel.read(idx));
    }

    if (this_rank.row() == rank_kk.row()) {
      for (SizeType j_loc = 0; j_loc < distr_b.localNrTiles().cols(); ++j_loc) {
        const LocalTileIndex kj(kk_offset.row(), j_loc);
        const auto& priority = thread_priority::high;

        dlaf::internal::whenAllLift(T(-1), b_panel.read_sender(kj), mat_b.readwrite_sender(kj)) |
            tile::add(dlaf::internal::Policy<backend>(priority)) |
            pika::execution::experimental::start_detached();

        trsmBPanelTile<backend>(priority, op, diag, alpha, a_panel.read_sender(kk_offset),
                                mat_b.readwrite_sender(kj));
      }
    }

    b_panel.reset();
    a_panel.reset();
  }
}

template <Backend backend, Device device, class T>
void Triangular<backend, device, T>::call_LUN(comm::CommunicatorGrid grid, blas::Diag diag, T alpha,
                                              Matrix<const T, device>& mat_a, Matrix<T, device>& mat_b) {
  using namespace triangular_lun;
  using pika::threads::thread_priority;

  // Set up MPI executor pipelines
  common::Pipeline<comm::Communicator> mpi_row_task_chain(grid.rowCommunicator().clone());
  common::Pipeline<comm::Communicator> mpi_col_task_chain(grid.colCommunicator().clone());

  const comm::Index2D this_rank = grid.rank();

  const matrix::Distribution& distr_a = mat_a.distribution();
  const matrix::Distribution& distr_b = mat_b.distribution();

  // If mat_b is empty return immediately
  if (mat_b.size().isEmpty())
    return;

  constexpr std::size_t n_workspaces = 2;
  common::RoundRobin<matrix::Panel<Coord::Col, T, device>> a_panels(n_workspaces, distr_a);
  common::RoundRobin<matrix::Panel<Coord::Row, T, device>> b_panels(n_workspaces, distr_b);

  for (SizeType k = mat_a.nrTiles().rows() - 1; k >= 0; --k) {
    const GlobalTileIndex kk(k, k);
    auto kk_rank = distr_a.rankGlobalTile(kk);

    const LocalTileIndex kk_offset{
        distr_a.nextLocalTileFromGlobalTile<Coord::Row>(k + 1),
        distr_a.nextLocalTileFromGlobalTile<Coord::Col>(k),
    };

    const LocalTileIndex bt_offset{distr_b.nextLocalTileFromGlobalTile<Coord::Row>(k), 0};

    auto& a_panel = a_panels.nextResource();
    auto& b_panel = b_panels.nextResource();
    if (k == mat_a.nrTiles().rows() - 1) {
      a_panel.setWidth(mat_a.tileSize(kk).rows());
      b_panel.setHeight(mat_a.tileSize(kk).cols());
    }

    if (kk_rank.col() == this_rank.col()) {
      for (SizeType i_local = kk_offset.row() - 1; i_local >= 0; --i_local) {
        const LocalTileIndex ik_panel(Coord::Row, i_local);
        const LocalTileIndex ik(i_local, kk_offset.col());
        a_panel.setTile(ik_panel, mat_a.read(ik));
      }
    }
    broadcast(kk_rank.col(), a_panel, mpi_row_task_chain);

    for (SizeType j_local = distr_b.localNrTiles().cols() - 1; j_local >= 0; --j_local) {
      // Triangular solve B's k-th row panel and broadcast B(kj) column-wise
      if (kk_rank.row() == this_rank.row()) {
        auto k_local_row = distr_b.localTileFromGlobalTile<Coord::Row>(k);
        const LocalTileIndex kk_panel(Coord::Row, k_local_row);
        const LocalTileIndex kj(k_local_row, j_local);
        const LocalTileIndex kj_panel(Coord::Col, j_local);

        trsmBPanelTile<backend>(thread_priority::high, diag, alpha, a_panel.read_sender(kk_panel),
                                mat_b.readwrite_sender(kj));
        b_panel.setTile(kj_panel, mat_b.read(kj));
      }
    }
    // Nothing else to do if the trailing matrix is empty.
    if (k == 0)
      continue;

    broadcast(kk_rank.row(), b_panel, mpi_col_task_chain);

    for (SizeType i_local = bt_offset.row() - 1; i_local >= 0; --i_local) {
      // Choose queue priority
      auto i = distr_a.globalTileFromLocalTile<Coord::Row>(i_local);
      const auto trailing_priority = (i == k - 1) ? thread_priority::high : thread_priority::normal;

      const LocalTileIndex ik_panel(Coord::Row, i_local);

      // Update trailing matrix
      for (SizeType j_local = distr_b.localNrTiles().cols() - 1; j_local >= 0; --j_local) {
        const LocalTileIndex kj_panel(Coord::Col, j_local);
        const LocalTileIndex ij(i_local, j_local);
        const T beta = T(-1.0) / alpha;

        gemmTrailingMatrixTile<backend>(trailing_priority, beta, a_panel.read_sender(ik_panel),
                                        b_panel.read_sender(kj_panel), mat_b.readwrite_sender(ij));
      }
    }
    a_panel.reset();
    b_panel.reset();
  }
}

template <Backend backend, Device device, class T>
void Triangular<backend, device, T>::call_RLN(comm::CommunicatorGrid grid, blas::Diag diag, T alpha,
                                              Matrix<const T, device>& mat_a, Matrix<T, device>& mat_b) {
  using namespace triangular_rln;
  using pika::threads::thread_priority;

  // Set up MPI executor pipelines
  common::Pipeline<comm::Communicator> mpi_row_task_chain(grid.rowCommunicator().clone());
  common::Pipeline<comm::Communicator> mpi_col_task_chain(grid.colCommunicator().clone());

  const comm::Index2D this_rank = grid.rank();

  const matrix::Distribution& distr_a = mat_a.distribution();
  const matrix::Distribution& distr_b = mat_b.distribution();

  // If mat_b is empty return immediately
  if (mat_b.size().isEmpty())
    return;

  constexpr std::size_t n_workspaces = 2;
  common::RoundRobin<matrix::Panel<Coord::Row, T, device>> a_panels(n_workspaces, distr_a);
  common::RoundRobin<matrix::Panel<Coord::Col, T, device>> b_panels(n_workspaces, distr_b);

  for (SizeType k = mat_a.nrTiles().cols() - 1; k >= 0; --k) {
    const GlobalTileIndex kk(k, k);
    auto kk_rank = distr_a.rankGlobalTile(kk);

    const LocalTileIndex kk_offset{
        distr_a.nextLocalTileFromGlobalTile<Coord::Row>(k),
        distr_a.nextLocalTileFromGlobalTile<Coord::Col>(k + 1),
    };

    const LocalTileIndex bt_offset{0, distr_b.nextLocalTileFromGlobalTile<Coord::Col>(k)};

    auto& a_panel = a_panels.nextResource();
    auto& b_panel = b_panels.nextResource();
    if (k == mat_a.nrTiles().cols() - 1) {
      a_panel.setHeight(mat_a.tileSize(kk).cols());
      b_panel.setWidth(mat_a.tileSize(kk).rows());
    }

    if (kk_rank.row() == this_rank.row()) {
      for (SizeType j_local = kk_offset.col() - 1; j_local >= 0; --j_local) {
        const LocalTileIndex kj_panel(Coord::Col, j_local);
        const LocalTileIndex kj(kk_offset.row(), j_local);
        a_panel.setTile(kj_panel, mat_a.read(kj));
      }
    }
    broadcast(kk_rank.row(), a_panel, mpi_col_task_chain);

    for (SizeType i_local = distr_b.localNrTiles().rows() - 1; i_local >= 0; --i_local) {
      // Triangular solve B's k-th col panel and broadcast B(ik) row-wise
      if (kk_rank.col() == this_rank.col()) {
        auto k_local_col = distr_b.localTileFromGlobalTile<Coord::Col>(k);
        const LocalTileIndex kk_panel(Coord::Col, k_local_col);
        const LocalTileIndex ik(i_local, k_local_col);
        const LocalTileIndex ik_panel(Coord::Row, i_local);

        trsmBPanelTile<backend>(thread_priority::high, diag, alpha, a_panel.read_sender(kk_panel),
                                mat_b.readwrite_sender(ik));
        b_panel.setTile(ik_panel, mat_b.read(ik));
      }
    }
    // Nothing else to do if the trailing matrix is empty.
    if (k == 0)
      continue;

    broadcast(kk_rank.col(), b_panel, mpi_row_task_chain);

    for (SizeType j_local = bt_offset.col() - 1; j_local >= 0; --j_local) {
      // Choose queue priority
      auto j = distr_a.globalTileFromLocalTile<Coord::Col>(j_local);
      const auto trailing_priority = (j == k - 1) ? thread_priority::high : thread_priority::normal;

      const LocalTileIndex kj_panel(Coord::Col, j_local);

      // Update trailing matrix
      for (SizeType i_local = distr_b.localNrTiles().rows() - 1; i_local >= 0; --i_local) {
        const LocalTileIndex ik_panel(Coord::Row, i_local);
        const LocalTileIndex ij(i_local, j_local);
        const T beta = T(-1.0) / alpha;

        gemmTrailingMatrixTile<backend>(trailing_priority, beta, b_panel.read_sender(ik_panel),
                                        a_panel.read_sender(kj_panel), mat_b.readwrite_sender(ij));
      }
    }
    a_panel.reset();
    b_panel.reset();
  }
}

template <Backend backend, Device device, class T>
void Triangular<backend, device, T>::call_RUN(comm::CommunicatorGrid grid, blas::Diag diag, T alpha,
                                              Matrix<const T, device>& mat_a, Matrix<T, device>& mat_b) {
  using namespace triangular_run;
  using pika::threads::thread_priority;

  // Set up MPI executor pipelines
  common::Pipeline<comm::Communicator> mpi_row_task_chain(grid.rowCommunicator().clone());
  common::Pipeline<comm::Communicator> mpi_col_task_chain(grid.colCommunicator().clone());

  const comm::Index2D this_rank = grid.rank();

  const matrix::Distribution& distr_a = mat_a.distribution();
  const matrix::Distribution& distr_b = mat_b.distribution();

  // If mat_b is empty return immediately
  if (mat_b.size().isEmpty())
    return;

  constexpr std::size_t n_workspaces = 2;
  common::RoundRobin<matrix::Panel<Coord::Row, T, device>> a_panels(n_workspaces, distr_a);
  common::RoundRobin<matrix::Panel<Coord::Col, T, device>> b_panels(n_workspaces, distr_b);

  for (SizeType k = 0; k < mat_a.nrTiles().cols(); ++k) {
    const GlobalTileIndex kk(k, k);
    auto kk_rank = distr_a.rankGlobalTile(kk);

    const LocalTileIndex kk_offset{
        distr_a.nextLocalTileFromGlobalTile<Coord::Row>(k),
        distr_a.nextLocalTileFromGlobalTile<Coord::Col>(k),
    };

    const LocalTileIndex bt_offset{0, distr_b.nextLocalTileFromGlobalTile<Coord::Col>(k + 1)};

    auto& a_panel = a_panels.nextResource();
    auto& b_panel = b_panels.nextResource();
    a_panel.setRangeStart(kk);
    if (k == mat_a.nrTiles().cols() - 1) {
      a_panel.setHeight(mat_a.tileSize(kk).rows());
      b_panel.setWidth(mat_a.tileSize(kk).cols());
    }

    if (kk_rank.row() == this_rank.row()) {
      for (SizeType j_local = kk_offset.col(); j_local < distr_a.localNrTiles().cols(); ++j_local) {
        const LocalTileIndex kj_panel(Coord::Col, j_local);
        const LocalTileIndex kj(kk_offset.row(), j_local);
        a_panel.setTile(kj_panel, mat_a.read(kj));
      }
    }
    broadcast(kk_rank.row(), a_panel, mpi_col_task_chain);

    for (SizeType i_local = 0; i_local < distr_b.localNrTiles().rows(); ++i_local) {
      // Triangular solve B's k-th row panel and broadcast B(kj) column-wise
      if (kk_rank.col() == this_rank.col()) {
        auto k_local_col = distr_b.localTileFromGlobalTile<Coord::Col>(k);
        const LocalTileIndex kk_panel(Coord::Col, k_local_col);
        const LocalTileIndex ik(i_local, k_local_col);
        const LocalTileIndex ik_panel(Coord::Row, i_local);

        trsmBPanelTile<backend>(thread_priority::high, diag, alpha, a_panel.read_sender(kk_panel),
                                mat_b.readwrite_sender(ik));
        b_panel.setTile(ik_panel, mat_b.read(ik));
      }
    }
    // Nothing else to do if the trailing matrix is empty.
    if (k == mat_a.nrTiles().cols() - 1)
      continue;

    broadcast(kk_rank.col(), b_panel, mpi_row_task_chain);

    for (SizeType j_local = bt_offset.col(); j_local < distr_a.localNrTiles().cols(); ++j_local) {
      // Choose queue priority
      auto j = distr_a.globalTileFromLocalTile<Coord::Col>(j_local);
      const auto trailing_priority = (j == k + 1) ? thread_priority::high : thread_priority::normal;

      const LocalTileIndex kj_panel(Coord::Col, j_local);

      // Update trailing matrix
      for (SizeType i_local = 0; i_local < distr_b.localNrTiles().rows(); ++i_local) {
        const LocalTileIndex ik_panel(Coord::Row, i_local);
        const LocalTileIndex ij(i_local, j_local);
        const T beta = T(-1.0) / alpha;

        gemmTrailingMatrixTile<backend>(trailing_priority, beta, b_panel.read_sender(ik_panel),
                                        a_panel.read_sender(kj_panel), mat_b.readwrite_sender(ij));
      }
    }
    a_panel.reset();
    b_panel.reset();
  }
}
}
}
}
