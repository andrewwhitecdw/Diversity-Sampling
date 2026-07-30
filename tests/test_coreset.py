#
# SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import numpy as np
from src.coreset import CoresetSampler

N_ROWS = 1024
N_COLS = 2
N_SAMPLES = N_ROWS // 8
RNG_SEED = 0


def _make_data(rng_seed=RNG_SEED):
    rng = np.random.default_rng(seed=rng_seed)
    return rng.standard_normal((N_ROWS, N_COLS)), rng


def test_coreset_outliers():
    """Test if coreset always includes diverse points (outliers)"""
    cs = CoresetSampler(n_samples=N_SAMPLES, random_seed=0)
    x, _ = _make_data()
    x[14, 0] = -100
    x[77, 1] = 100

    cs.initialize(x)
    indices = cs.sample(x)

    assert (14 in indices) and (77 in indices)


def test_coreset_determinism():
    """Test if coreset sampling output is always the same given random seed."""
    x, _ = _make_data()

    cs1 = CoresetSampler(n_samples=N_SAMPLES, random_seed=4)
    cs2 = CoresetSampler(n_samples=N_SAMPLES, random_seed=4)

    cs1.initialize(x)
    indices1 = cs1.sample(x)

    cs2.initialize(x)
    indices2 = cs2.sample(x)

    assert np.array_equal(indices1, indices2)


def test_coreset_std():
    """Test if coreset sampling is more diverse than random sampling."""
    cs = CoresetSampler(n_samples=N_SAMPLES, random_seed=0)
    x, rng = _make_data()

    cs.initialize(x)
    indices = cs.sample(x)

    random_indices = rng.choice(N_ROWS, N_SAMPLES)

    assert x[indices].std() > x[random_indices].std()


def test_coreset_sample_count():
    """Test if coreset returns exactly n_samples indices."""
    cs = CoresetSampler(n_samples=N_SAMPLES, random_seed=0)
    x, _ = _make_data()
    cs.initialize(x)
    indices = cs.sample(x)
