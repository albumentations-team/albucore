import warnings

import cv2
import numpy as np
import pytest
import torch

from albucore import matmul, pairwise_distances_squared


class TestMatmul:
    @pytest.mark.parametrize(
        "m,k,n",
        [
            (2, 2, 2),  # Tiny (stain normalization)
            (3, 3, 3),  # Tiny (affine transforms)
            (10, 10, 10),  # Small (TPS typical)
            (100, 100, 100),  # Medium
            pytest.param(512, 512, 512, marks=pytest.mark.slow),  # Large
            pytest.param(262144, 2, 10, marks=pytest.mark.slow),  # TPS: pairwise dots
            pytest.param(262144, 10, 2, marks=pytest.mark.slow),  # TPS: nonlinear
            pytest.param(262144, 3, 2, marks=pytest.mark.slow),  # TPS: affine
        ],
    )
    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_matmul_matches_cv2_gemm(self, m: int, k: int, n: int, dtype: type) -> None:
        a = np.random.randn(m, k).astype(dtype)
        b = np.random.randn(k, n).astype(dtype)

        result_cv2 = cv2.gemm(a, b, 1, None, 0)
        result_matmul = matmul(a, b)

        # Tolerance accounts for different BLAS / OpenCV GEMM implementations across
        # platforms (e.g., OpenBLAS on Linux CI vs local); large (512³) float32 can
        # land ~4e-5 apart on a few elements.
        if dtype == np.float32:
            np.testing.assert_allclose(result_matmul, result_cv2, rtol=1e-4, atol=5e-5)
        else:  # float64
            np.testing.assert_allclose(result_matmul, result_cv2, rtol=1e-6, atol=1e-8)

    @pytest.mark.parametrize(
        "m,k,n",
        [
            (2, 2, 2),
            (10, 10, 10),
            (100, 100, 100),
        ],
    )
    def test_matmul_uint8(self, m: int, k: int, n: int) -> None:
        """Verify matmul works with uint8 (cv2.gemm doesn't support this)."""
        a = np.random.randint(0, 256, size=(m, k), dtype=np.uint8)
        b = np.random.randint(0, 256, size=(k, n), dtype=np.uint8)

        result = matmul(a, b)

        expected = a @ b
        np.testing.assert_array_equal(result, expected)

        assert result.shape == (m, n)
        assert result.dtype == expected.dtype

    @pytest.mark.parametrize("as_tensor", [False, True])
    def test_matmul_uint8_wraps_in_output_dtype(self, as_tensor: bool) -> None:
        a = np.array([[255]], dtype=np.uint8)
        b = np.array([[255]], dtype=np.uint8)
        if as_tensor:
            a = torch.from_numpy(a)
            b = torch.from_numpy(b)

        result = matmul(a, b)

        expected = torch.tensor([[1]], dtype=torch.uint8) if as_tensor else np.array([[1]], dtype=np.uint8)
        assert result.dtype == expected.dtype
        if isinstance(result, torch.Tensor):
            torch.testing.assert_close(result, expected, rtol=0, atol=0)
        else:
            np.testing.assert_array_equal(result, expected)

    def test_matmul_tensor_float16_matches_wide_accumulation(self) -> None:
        rng = np.random.default_rng(0)
        a = torch.from_numpy(rng.random((128, 64), dtype=np.float32).astype(np.float16))
        b = torch.from_numpy(rng.random((64, 32), dtype=np.float32).astype(np.float16))

        result = matmul(a, b)
        expected = torch.from_numpy((a.numpy().astype(np.float32) @ b.numpy().astype(np.float32)).astype(np.float16))

        assert isinstance(result, torch.Tensor)
        assert result.dtype == torch.float16
        torch.testing.assert_close(result, expected, rtol=0, atol=0)

    def test_matmul_output_dtype(self) -> None:
        a_f32 = np.random.randn(5, 3).astype(np.float32)
        b_f32 = np.random.randn(3, 4).astype(np.float32)
        result = matmul(a_f32, b_f32)
        assert result.dtype == np.float32

        a_f64 = np.random.randn(5, 3).astype(np.float64)
        b_f64 = np.random.randn(3, 4).astype(np.float64)
        result = matmul(a_f64, b_f64)
        assert result.dtype == np.float64

    def test_matmul_float16_uses_wide_accumulation_and_preserves_dtype(self) -> None:
        rng = np.random.default_rng(0)
        a = rng.random((128, 64), dtype=np.float32).astype(np.float16)
        b = rng.random((64, 32), dtype=np.float32).astype(np.float16)

        result = matmul(a, b)
        expected = (a.astype(np.float32) @ b.astype(np.float32)).astype(np.float16)

        assert result.dtype == np.float16
        np.testing.assert_array_equal(result, expected)

    def test_matmul_shape(self) -> None:
        a = np.random.randn(7, 5).astype(np.float32)
        b = np.random.randn(5, 11).astype(np.float32)
        result = matmul(a, b)
        assert result.shape == (7, 11)

    def test_matmul_identity(self) -> None:
        a = np.random.randn(5, 5).astype(np.float32)
        identity = np.eye(5, dtype=np.float32)

        result = matmul(a, identity)
        np.testing.assert_allclose(result, a, rtol=1e-5)

        result = matmul(identity, a)
        np.testing.assert_allclose(result, a, rtol=1e-5)

    def test_matmul_zeros(self) -> None:
        a = np.random.randn(3, 4).astype(np.float32)
        zeros = np.zeros((4, 5), dtype=np.float32)

        result = matmul(a, zeros)
        expected = np.zeros((3, 5), dtype=np.float32)
        np.testing.assert_array_equal(result, expected)

    def test_tall_float32_inputs_do_not_emit_runtime_warnings(self) -> None:
        points = np.linspace(0, 1, 4096 * 2, dtype=np.float32).reshape(4096, 2)
        control_points = np.array(
            [[0, 0], [0, 1], [1, 0], [1, 1], [0.5, 0.5]],
            dtype=np.float32,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            result = matmul(points, control_points.T)

        expected = cv2.gemm(points, control_points.T, 1.0, None, 0.0)
        assert result.dtype == np.float32
        assert np.isfinite(result).all()
        np.testing.assert_allclose(result, expected, rtol=1e-4, atol=5e-5)

    @pytest.mark.parametrize(
        ("a_shape", "b_shape"),
        [
            ((32,), (32,)),
            ((32,), (32, 2)),
            ((2, 32), (32,)),
        ],
    )
    def test_macos_warning_suppression_supports_vector_inputs(
        self,
        a_shape: tuple[int, ...],
        b_shape: tuple[int, ...],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr("albucore.ops_misc._IS_MACOS_ARM64", True)
        a = np.arange(np.prod(a_shape), dtype=np.float32).reshape(a_shape)
        b = np.arange(np.prod(b_shape), dtype=np.float32).reshape(b_shape)

        result = matmul(a, b)

        np.testing.assert_allclose(result, a @ b, rtol=1e-6, atol=1e-6)


class TestPairwiseDistancesSquared:
    @pytest.mark.parametrize("n,m", [(1, 999), (1, 1000), (25, 40)])
    @pytest.mark.parametrize("as_tensor", [False, True], ids=["numpy", "tensor"])
    def test_zero_coordinates(self, n: int, m: int, as_tensor: bool) -> None:
        points1 = np.empty((n, 0), dtype=np.float32)
        points2 = np.empty((m, 0), dtype=np.float32)
        result = pairwise_distances_squared(
            torch.from_numpy(points1) if as_tensor else points1,
            torch.from_numpy(points2) if as_tensor else points2,
        )
        assert isinstance(result, torch.Tensor if as_tensor else np.ndarray)
        if as_tensor:
            result = result.numpy()
        assert result.dtype == np.float32
        np.testing.assert_array_equal(result, np.zeros((n, m), dtype=np.float32))

    @pytest.mark.parametrize(
        "n,m,dimensions",
        [(31, 31, 2), (32, 32, 2), (25, 40, 2), (1, 999, 2), (1, 1000, 2), (1, 1001, 2), (32, 32, 9)],
    )
    @pytest.mark.parametrize(
        "first,second,squared_distance",
        [
            ([1e20, 1e20], [1e20, 1e20], 0.0),
            ([1e20, 0], [1e20, 1], 1.0),
            ([10000, 10000], [10000.125, 10000.125], 0.03125),
            ([0, 0], [1e20, 0], np.inf),
        ],
        ids=["identical-large", "large-shared-coordinate", "common-offset", "unrepresentable-distance"],
    )
    @pytest.mark.parametrize("as_tensor", [False, True], ids=["numpy", "tensor"])
    @pytest.mark.parametrize("strided", [False, True], ids=["contiguous", "strided"])
    def test_coordinate_offsets_and_overflow(
        self, n: int, m: int, first: list[float], second: list[float], squared_distance: float,
        as_tensor: bool, strided: bool, dimensions: int,
    ) -> None:
        points1 = np.zeros((n, dimensions * (2 if strided else 1)), dtype=np.float32)
        points2 = np.zeros((m, dimensions * (2 if strided else 1)), dtype=np.float32)
        if strided:
            points1 = points1[:, ::2]
            points2 = points2[:, ::2]
        points1[:, :2] = first
        points2[:, :2] = second
        original1, original2 = points1.copy(), points2.copy()
        expected = np.full((n, m), squared_distance, dtype=np.float32)

        with np.errstate(over="ignore" if np.isinf(squared_distance) else "raise", invalid="raise"):
            result = pairwise_distances_squared(
                torch.from_numpy(points1) if as_tensor else points1,
                torch.from_numpy(points2) if as_tensor else points2,
            )

        assert isinstance(result, torch.Tensor if as_tensor else np.ndarray)
        if as_tensor:
            assert result.device.type == "cpu"
            result = result.numpy()
        assert result.dtype == np.float32
        np.testing.assert_array_equal(result, expected)
        np.testing.assert_array_equal(points1, original1)
        np.testing.assert_array_equal(points2, original2)

    @pytest.mark.parametrize("dimensions", [1, 2, 3, 5, 8, 9, 64])
    @pytest.mark.parametrize("storage", ["contiguous", "strided", "reversed", "read-only"])
    def test_mixed_points_at_common_offset(self, dimensions: int, storage: str) -> None:
        rng = np.random.default_rng(197)
        points1 = 10000 + rng.integers(-8, 9, size=(32, dimensions)).astype(np.float32) * 0.125
        points2 = 10000 + rng.integers(-8, 9, size=(40, dimensions)).astype(np.float32) * 0.125
        differences = points1.astype(np.float64)[:, None, :] - points2.astype(np.float64)[None, :, :]
        expected = np.sum(differences**2, axis=2).astype(np.float32)
        if storage == "strided":
            points1, points2 = points1.T.copy().T, points2.T.copy().T
        elif storage == "reversed":
            points1, points2 = points1[::-1], points2[::-1]
            expected = expected[::-1, ::-1]
        elif storage == "read-only":
            points1.flags.writeable = False
            points2.flags.writeable = False

        result = pairwise_distances_squared(points1, points2)

        np.testing.assert_array_equal(result, expected)

    def test_matches_manual_computation(self) -> None:
        points1 = np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float32)
        points2 = np.array([[0, 0], [1, 1]], dtype=np.float32)

        result = pairwise_distances_squared(points1, points2)

        expected = np.array(
            [
                [0.0, 2.0],  # (0,0) to [(0,0), (1,1)]
                [1.0, 1.0],  # (1,0) to [(0,0), (1,1)]
                [1.0, 1.0],  # (0,1) to [(0,0), (1,1)]
            ],
            dtype=np.float32,
        )

        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-7)

    def test_matches_old_cv2_implementation(self) -> None:
        # Former AlbumentationsX implementation.
        def old_implementation(
            points1: np.ndarray,
            points2: np.ndarray,
        ) -> np.ndarray:
            points1 = np.ascontiguousarray(points1, dtype=np.float32)
            points2 = np.ascontiguousarray(points2, dtype=np.float32)
            p1_squared = cv2.multiply(points1, points1).sum(axis=1, keepdims=True)
            p2_squared = cv2.multiply(points2, points2).sum(axis=1)[None, :]
            dot_product = cv2.gemm(points1, points2.T, 1, None, 0)
            return p1_squared + p2_squared - 2 * dot_product

        points1 = np.random.randn(50, 2).astype(np.float32)
        points2 = np.random.randn(30, 2).astype(np.float32)

        result_old = old_implementation(points1, points2)
        result_new = pairwise_distances_squared(points1, points2)

        np.testing.assert_allclose(result_new, result_old, rtol=1e-4, atol=1e-6)

    @pytest.mark.parametrize(
        "n,m,d",
        [
            (10, 10, 2),  # Typical TPS control points (NumKong cdist backend)
            (100, 100, 2),  # Larger point sets (numpy backend)
            (1000, 100, 2),  # Dense grid (numpy backend)
            (5, 10, 3),  # 3D points
            (50, 20, 5),  # Higher dimensions
        ],
    )
    def test_various_sizes(self, n: int, m: int, d: int) -> None:
        points1 = np.random.randn(n, d).astype(np.float32)
        points2 = np.random.randn(m, d).astype(np.float32)

        result = pairwise_distances_squared(points1, points2)

        assert result.shape == (n, m)
        assert result.dtype == np.float32
        assert np.all(result >= 0)

    @pytest.mark.parametrize("n,dimensions", [(10, 2), (32, 9), (32, 64)])
    def test_self_distance_is_zero(self, n: int, dimensions: int) -> None:
        points = np.random.randn(n, dimensions).astype(np.float32)
        result = pairwise_distances_squared(points, points)

        diagonal = np.diag(result)
        np.testing.assert_array_equal(diagonal, np.zeros(n, dtype=np.float32))
        assert np.all(result >= 0)

    def test_symmetry(self) -> None:
        points = np.random.randn(20, 3).astype(np.float32)
        result = pairwise_distances_squared(points, points)

        np.testing.assert_allclose(result, result.T, rtol=1e-5, atol=1e-7)

    def test_triangle_inequality(self) -> None:
        points = np.array([[0, 0], [1, 0], [0.5, 0.5]], dtype=np.float32)

        distances_sq = pairwise_distances_squared(points, points)
        distances = np.sqrt(distances_sq)

        for i in range(3):
            for j in range(3):
                for k in range(3):
                    assert distances[i, k] <= distances[i, j] + distances[j, k] + 1e-5

    def test_output_dtype_float32(self) -> None:
        # Input as float64 should be converted
        points1 = np.random.randn(10, 2).astype(np.float64)
        points2 = np.random.randn(5, 2).astype(np.float64)

        result = pairwise_distances_squared(points1, points2)
        assert result.dtype == np.float32

    def test_non_contiguous_input(self) -> None:
        points1 = np.random.randn(2, 20).astype(np.float32).T  # (20, 2) non-contiguous
        points2 = np.random.randn(2, 10).astype(np.float32).T  # (10, 2) non-contiguous

        assert not points1.flags["C_CONTIGUOUS"]
        assert not points2.flags["C_CONTIGUOUS"]

        result = pairwise_distances_squared(points1, points2)

        points1_c = np.ascontiguousarray(points1)
        points2_c = np.ascontiguousarray(points2)
        expected = pairwise_distances_squared(points1_c, points2_c)

        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_numkong_backend_small_points(self) -> None:
        # Small point set (10*10 = 100 < 1000) uses nk.cdist
        points1 = np.random.randn(10, 2).astype(np.float32)
        points2 = np.random.randn(10, 2).astype(np.float32)

        result = pairwise_distances_squared(points1, points2)

        def numpy_impl(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
            p1 = np.ascontiguousarray(p1, dtype=np.float32)
            p2 = np.ascontiguousarray(p2, dtype=np.float32)
            p1_sq = (p1**2).sum(axis=1, keepdims=True)
            p2_sq = (p2**2).sum(axis=1)[None, :]
            dot = p1 @ p2.T
            return p1_sq + p2_sq - 2 * dot

        expected = numpy_impl(points1, points2)
        np.testing.assert_allclose(result, expected, rtol=1e-4, atol=1e-5)

    def test_numpy_backend_large_points(self) -> None:
        # Large point set (100*100 = 10000 >= 1000) should use numpy
        points1 = np.random.randn(100, 2).astype(np.float32)
        points2 = np.random.randn(100, 2).astype(np.float32)

        result = pairwise_distances_squared(points1, points2)

        for _ in range(5):
            i = np.random.randint(0, 100)
            j = np.random.randint(0, 100)
            expected_dist = np.sum((points1[i] - points2[j]) ** 2)
            np.testing.assert_allclose(result[i, j], expected_dist, rtol=1e-5, atol=1e-6)

    def test_tall_float32_inputs_do_not_emit_runtime_warnings(self) -> None:
        points1 = np.linspace(0, 1, 4096 * 2, dtype=np.float32).reshape(4096, 2)
        points2 = np.array(
            [[0, 0], [0, 1], [1, 0], [1, 1], [0.5, 0.5]],
            dtype=np.float32,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            result = pairwise_distances_squared(points1, points2)

        expected = ((points1[:, None, :] - points2[None, :, :]) ** 2).sum(axis=2)
        assert result.dtype == np.float32
        assert np.isfinite(result).all()
        assert np.all(result >= 0)
        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=2e-7)

    def test_edge_case_single_point(self) -> None:
        point1 = np.array([[1.0, 2.0]], dtype=np.float32)
        point2 = np.array([[4.0, 6.0]], dtype=np.float32)

        result = pairwise_distances_squared(point1, point2)

        # Distance from (1,2) to (4,6): sqrt((4-1)^2 + (6-2)^2) = sqrt(9+16) = 5
        # Squared distance: 25
        expected = np.array([[25.0]], dtype=np.float32)
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_known_distances(self) -> None:
        # Unit square corners
        points1 = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
        points2 = np.array([[0, 0]], dtype=np.float32)

        result = pairwise_distances_squared(points1, points2)

        expected = np.array([[0.0], [1.0], [2.0], [1.0]], dtype=np.float32)
        np.testing.assert_allclose(result, expected, rtol=1e-5)
