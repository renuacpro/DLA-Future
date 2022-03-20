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

#include "dlaf/communication/communicator_grid.h"
#include "dlaf/eigensolver/tridiag_solver/mc.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf/types.h"
#include "dlaf/util_matrix.h"

namespace dlaf {
namespace eigensolver {

/// Finds the eigenvalues and eigenvectors of a symmetric tridiagonal matrix.
///
/// Notation:
///
/// nb - the block/tile size of all matrices and vectors
/// n1 - the size of the top subproblem
/// n2 - the size of the bottom subproblem
/// Q1 - (n1 x n1) the orthogonal matrix of the top subproblem
/// Q2 - (n2 x n2) the orthogonal matrix of the bottom subproblem
/// n := n1 + n2, the size of the merged problem
///
///      ┌───┬───┐
///      │Q1 │   │
/// Q := ├───┼───┤ , (n x n) orthogonal matrix composed of the top and bottom subproblems
///      │   │Q2 │
///      └───┴───┘
/// D                 := diag(Q), (n x 1) the diagonal of Q
/// z                 := (n x 1) rank 1 update vector
/// rho               := rank 1 update scaling factor
/// D + rho*z*z^T     := rank 1 update problem
/// U                 := (n x n) matrix of eigenvectors of the rank 1 update problem:
///
/// k                 := the size of the deflated rank 1 update problem (k <= n)
/// D'                := (k x 1), deflated D
/// z'                := (k x 1), deflated z
/// D' + rho*z'*z'^T  := deflated rank 1 update problem
/// U'                := (k x k) matrix of eigenvectors of the deflated rank 1 update problem
///
/// l1  := number of columns of the top subproblem after deflation
/// l2  := number of columns of the bottom subproblem after deflation
/// Q1' := (n1 x l1) the non-deflated part of Q1 (l1 < n1)
/// Q2' := (n2 x l2) the non-deflated part of Q2 (l2 < n2)
/// Qd  := (n-k x n) the deflated parts of Q1 and Q2
/// U1' := (k x l1) is the first l1 columns of U'
/// U2' := (k x l2) is the last l2 columns of U'
/// I   := (n-k x n-k) identity matrix
/// P   := (n x n) permutation matrix used to bring Q and U into multiplication form
///
/// Q-U multiplication form to arrive at the eigenvectors of the merged problem:
///
///          ┌────┬───┬──┐   ┌────┬───┬──┐T  ┌────────┬──┐
///          │ Q1'│   │  │   │ U1'│   │  │   │Q1'U1'^T│  │
/// QPP^TU = │    │   │  │   │        │  │   │        │  │
///          ├───┬┴───┤Qd│ X │   │ U2'│  │ = ├────────┤Qd│
///          │   │Q2' │  │   ├───┴────┼──┤   │Q2'U2'^T│  │
///          │   │    │  │   │        │I │   │        │  │
///          └───┴────┴──┘   └────────┴──┘   └────────┴──┘
///
/// Note:
/// 1. U1' and U2' may overlap (in practice they almost always do)
/// 2. The second matrix is transposed
/// 3. The overlap between U1' and U2' matches the number of shared columns between Q1' and Q2'
/// 4. The overlap region is due to deflation via Givens rotations of a column vector from Q1 with a
///    column vector of Q2.
///
/// The following statements hold:
///
/// 1. either n2 = n1 or n2 = n1 + nb
/// 2. l1 + l2 <= n
///
/// @param mat_a  [in/out] (n x 2) local matrix with the diagonal and off-diagonal of the symmetric
/// tridiagonal matrix in the first column and second columns respectively. The last entry of the second
/// column is not used. On exit the eigenvalues are saved in the first column.
/// @param mat_ev [out]    (n x n) local matrix holding the eigenvectors of the the symmetric tridiagonal
/// matrix on exit.
///
/// @pre mat_a and mat_ev are local matrices
/// @pre mat_a has 2 columns
/// @pre mat_ev is a square matrix
/// @pre mat_ev has a square block size
template <Backend backend, Device device, class T>
void tridiagSolver(Matrix<T, device>& mat_trd, Matrix<T, device>& mat_ev) {
  static_assert(std::is_same<T, float>::value || std::is_same<T, double>::value,
                "tridagSolver accepts only real values (float, double)!");

  DLAF_ASSERT(matrix::local_matrix(mat_trd), mat_trd);
  DLAF_ASSERT(mat_trd.distribution().size().cols() == 2, mat_trd);

  DLAF_ASSERT(matrix::local_matrix(mat_ev), mat_ev);
  DLAF_ASSERT(matrix::square_size(mat_ev), mat_ev);
  DLAF_ASSERT(matrix::square_blocksize(mat_ev), mat_ev);
  // TODO: ASSERT that `mat_trd` and `mat_ev` have column-major layout

  // Auxiliary matrix used for the D&C algorithm
  const matrix::Distribution& distr = mat_ev.distribution();
  // Extra workspace for Q1', Q2' and U1' when l2 > l1 or U2' when l1 > l2
  Matrix<T, device> mat_qws(distr);
  // Extra workspace for U
  Matrix<T, device> mat_uws(distr);

  // Auxialiary vectors used for the D&C algorithm
  LocalElementSize vec_size(distr.size().rows(), 1);
  TileElementSize vec_tile_size(distr.blockSize().rows(), 1);
  // Holds the diagonal elements of the tridiagonal matrix
  Matrix<T, device> d(vec_size, vec_tile_size);
  // Holds the values of the deflated diagonal sorted in ascending order
  Matrix<T, device> d_defl(vec_size, vec_tile_size);
  // Holds the values of Cuppen's rank-1 vector
  Matrix<T, device> z(vec_size, vec_tile_size);
  // Holds the values of the rank-1 update vector sorted corresponding to `d_defl`
  Matrix<T, device> z_defl(vec_size, vec_tile_size);
  // Holds indices/permutations of elements of the diagonal sorted in ascending order.
  Matrix<SizeType, Device::CPU> perm_d(vec_size, vec_tile_size);
  // Holds indices/permutations of the rows of U that bring it in Q-U matrix multiplication form
  Matrix<SizeType, Device::CPU> perm_u(vec_size, vec_tile_size);
  // Holds indices/permutations of the columns of Q that bring it in Q-U matrix multiplication form
  Matrix<SizeType, Device::CPU> perm_q(vec_size, vec_tile_size);
  // Assigns a type to each column of Q which is used to calculate the permutation indices for Q and U
  // that bring them in matrix multiplication form.
  Matrix<internal::ColType, Device::CPU> coltypes(vec_size, vec_tile_size);

  // Tile indices of the first and last diagonal tiles
  SizeType i_begin = 0;
  SizeType i_end = SizeType(distr.nrTiles().rows() - 1);

  internal::TridiagSolver<backend, device, T>::call(i_begin, i_end, coltypes, d, d_defl, z, z_defl,
                                                    perm_d, perm_q, perm_u, mat_qws, mat_uws, mat_trd,
                                                    mat_ev);
}

/// TODO: more info on the distributed version
/// Finds the eigenvalues and eigenvectors of a symmetric tridiagonal matrix.
///
/// @param mat_a  [in/out] `n x 2` matrix with the diagonal and off-diagonal of the symmetric tridiagonal
/// matrix in the first column and second columns respectively. The last entry of the second column is
/// not used. On exit the eigenvalues are saved in the first column.
/// @param mat_ev [out]    `n x n` matrix holding the eigenvectors of the the symmetric tridiagonal
/// matrix on exit.
///
/// @pre mat_a and mat_ev are local matrices
/// @pre mat_a has 2 columns
/// @pre mat_ev is a square matrix
/// @pre mat_ev has a square block size
template <Backend backend, Device device, class T>
void tridiagSolver(comm::CommunicatorGrid grid, Matrix<T, device>& mat_a, Matrix<T, device>& mat_ev) {
  DLAF_ASSERT(matrix::local_matrix(mat_a), mat_a);
  DLAF_ASSERT(mat_a.distribution().size().cols() == 2, mat_a);

  DLAF_ASSERT(matrix::local_matrix(mat_ev), mat_ev);
  DLAF_ASSERT(matrix::square_size(mat_ev), mat_ev);
  DLAF_ASSERT(matrix::square_blocksize(mat_ev), mat_ev);
  DLAF_ASSERT(matrix::equal_process_grid(mat_a, grid), mat_a, grid);
  DLAF_ASSERT(matrix::equal_process_grid(mat_ev, grid), mat_ev, grid);

  internal::TridiagSolver<backend, device, T>::call(std::move(grid), mat_a, mat_ev);
}

}
}
