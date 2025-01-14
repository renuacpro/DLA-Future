#!/bin/bash

#
# Distributed Linear Algebra with Future (DLAF)
#
# Copyright (c) 2018-2022, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
#

function submit() {
    local commit_status=${1}
    local commit_sha=${2}

    curl --verbose \
        --url "https://api.github.com/repos/eth-cscs/DLA-Future/statuses/${commit_sha}" \
        --header 'Content-Type: application/json' \
        --header "authorization: Bearer ${GITHUB_TOKEN}" \
        --data "{ \"state\": \"${commit_status}\", \"target_url\": \"${CI_PIPELINE_URL}\", \"description\": \"All Gitlab pipelines\", \"context\": \"ci/gitlab/full-pipeline\" }"
}

commit_status=${1}

# always submit for the current commit
submit "${commit_status}" "$CI_COMMIT_SHA"

# For Bors: get the latest commit before the merge to set the status.
if [[ $CI_COMMIT_REF_NAME =~ ^(trying|staging)$ ]]; then
    parent_sha=`git rev-parse --verify -q "$CI_COMMIT_SHA"^2`
    if [[ $? -eq 0 ]]; then
        submit "${commit_status}" "${parent_sha}"
    fi
fi

exit 0
