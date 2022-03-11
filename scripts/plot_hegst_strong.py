#!/usr/bin/env python3
# coding: utf-8

#
# Distributed Linear Algebra with Future (DLAF)
#
# Copyright (c) 2018-2022, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
#

import postprocess as pp

df = pp.parse_jobs_cmdargs(description="Plot gen2std strong scaling benchmarks.")

df_grp = pp.calc_gen2std_metrics(df)
pp.gen_gen2std_plots_strong(df_grp)
pp.gen_gen2std_plots_strong(df_grp, combine_mb=True)
