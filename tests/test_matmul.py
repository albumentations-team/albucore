import cv2
import numpy as np
import pytest

from albucore import matmul, pairwise_distances_squared


class TestMatmul:
    """Test matmul function."""

    @pytest.mark.parametrize(
        "m,k,n",
        [
            (2, 2, 2),  # Tiny (stain normalization)
            (3, 3, 3),  # Tiny (affine transforms)
            (10, 10, 10),  # Small (TPS typical)
            (100, 100, 100),  # Medium
            (512, 512, 512),  # Large
            (262144, 2, 10),  # TPS: pairwise dots
            (262144, 10, 2),  # TPS: nonlinear
            (262144, 3, 2),  # TPS: affine
        ],
    )
    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_matmul_matches_cv2_gemm(self, m: int, k: int, n: int, dtype: type) -> None:
        """Verify matmul gives identical results to cv2.gemm."""
        a = np.random.randn(m, k).astype(dtype)
        b = np.random.randn(k, n).astype(dtype)

        result_cv2 = cv2.gemm(a, b, 1, None, 0)
        result_matmul = matmul(a, b)

        if dtype == np.float32:
            np.testing.assert_allclose(result_matmul, result_cv2, rtol=1e-4, atol=1e-5)
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

        # Verify against NumPy @ operator
        expected = a @ b
        np.testing.assert_array_equal(result, expected)

        # Verify shape and dtype match NumPy behavior
        assert result.shape == (m, n)
        assert result.dtype == expected.dtype

    def test_matmul_output_dtype(self) -> None:
        """Verify output dtype matches input dtype for float types."""
        a_f32 = np.random.randn(5, 3).astype(np.float32)
        b_f32 = np.random.randn(3, 4).astype(np.float32)
        result = matmul(a_f32, b_f32)
        assert result.dtype == np.float32

        a_f64 = np.random.randn(5, 3).astype(np.float64)
        b_f64 = np.random.randn(3, 4).astype(np.float64)
        result = matmul(a_f64, b_f64)
        assert result.dtype == np.float64

    def test_matmul_shape(self) -> None:
        """Verify output shape is correct."""
        a = np.random.randn(7, 5).astype(np.float32)
        b = np.random.randn(5, 11).astype(np.float32)
        result = matmul(a, b)
        assert result.shape == (7, 11)

    def test_matmul_identity(self) -> None:
        """Test multiplication by identity matrix."""
        a = np.random.randn(5, 5).astype(np.float32)
        identity = np.eye(5, dtype=np.float32)

        result = matmul(a, identity)
        np.testing.assert_allclose(result, a, rtol=1e-5)

        result = matmul(identity, a)
        np.testing.assert_allclose(result, a, rtol=1e-5)

    def test_matmul_zeros(self) -> None:
        """Test multiplication with zero matrix."""
        a = np.random.randn(3, 4).astype(np.float32)
        zeros = np.zeros((4, 5), dtype=np.float32)

        result = matmul(a, zeros)
        expected = np.zeros((3, 5), dtype=np.float32)
        np.testing.assert_array_equal(result, expected)


class TestPairwiseDistancesSquared:
    """Test pairwise_distances_squared function."""

    def test_matches_manual_computation(self) -> None:
        """Verify against manual distance computation."""
        points1 = np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float32)
        points2 = np.array([[0, 0], [1, 1]], dtype=np.float32)

        result = pairwise_distances_squared(points1, points2)

        # Manual computation
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
        """Verify against old cv2.multiply + cv2.gemm implementation."""

        # This is what's currently in AlbumentationsX
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

        # Test with random points
        points1 = np.random.randn(50, 2).astype(np.float32)
        points2 = np.random.randn(30, 2).astype(np.float32)

        result_old = old_implementation(points1, points2)
        result_new = pairwise_distances_squared(points1, points2)

        np.testing.assert_allclose(result_new, result_old, rtol=1e-4, atol=1e-6)

    @pytest.mark.parametrize(
        "n,m,d",
        [
            (10, 10, 2),  # Typical TPS control points (simsimd backend)
            (100, 100, 2),  # Larger point sets (numpy backend)
            (1000, 100, 2),  # Dense grid (numpy backend)
            (5, 10, 3),  # 3D points
            (50, 20, 5),  # Higher dimensions
        ],
    )
    def test_various_sizes(self, n: int, m: int, d: int) -> None:
        """Test with various point set sizes and dimensions."""
        points1 = np.random.randn(n, d).astype(np.float32)
        points2 = np.random.randn(m, d).astype(np.float32)

        result = pairwise_distances_squared(points1, points2)

        assert result.shape == (n, m)
        assert result.dtype == np.float32
        # Distances are non-negative
        assert np.all(result >= -1e-5)  # Allow small numerical errors

    def test_self_distance_is_zero(self) -> None:
        """Verify that distance from point to itself is zero."""
        points = np.random.randn(10, 2).astype(np.float32)
        result = pairwise_distances_squared(points, points)

        # Diagonal should be all zeros (distance to self)
        diagonal = np.diag(result)
        np.testing.assert_allclose(diagonal, 0, atol=1e-5)

    def test_symmetry(self) -> None:
        """Verify that distance matrix is symmetric when comparing same points."""
        points = np.random.randn(20, 3).astype(np.float32)
        result = pairwise_distances_squared(points, points)

        # Result should be symmetric
        np.testing.assert_allclose(result, result.T, rtol=1e-5, atol=1e-7)

    def test_triangle_inequality(self) -> None:
        """Verify triangle inequality: sqrt(d(a,c)) <= sqrt(d(a,b)) + sqrt(d(b,c))."""
        # Create three points that form a triangle
        points = np.array([[0, 0], [1, 0], [0.5, 0.5]], dtype=np.float32)

        # Compute all pairwise distances
        distances_sq = pairwise_distances_squared(points, points)
        distances = np.sqrt(distances_sq)

        # Triangle inequality for all combinations
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    # d(i,k) <= d(i,j) + d(j,k)
                    assert distances[i, k] <= distances[i, j] + distances[j, k] + 1e-5

    def test_output_dtype_float32(self) -> None:
        """Verify output is always float32."""
        # Input as float64 should be converted
        points1 = np.random.randn(10, 2).astype(np.float64)
        points2 = np.random.randn(5, 2).astype(np.float64)

        result = pairwise_distances_squared(points1, points2)
        assert result.dtype == np.float32

    def test_non_contiguous_input(self) -> None:
        """Test that non-contiguous arrays are handled correctly."""
        # Create non-contiguous arrays via transpose
        points1 = np.random.randn(2, 20).astype(np.float32).T  # (20, 2) non-contiguous
        points2 = np.random.randn(2, 10).astype(np.float32).T  # (10, 2) non-contiguous

        assert not points1.flags["C_CONTIGUOUS"]
        assert not points2.flags["C_CONTIGUOUS"]

        result = pairwise_distances_squared(points1, points2)

        # Compare with contiguous version
        points1_c = np.ascontiguousarray(points1)
        points2_c = np.ascontiguousarray(points2)
        expected = pairwise_distances_squared(points1_c, points2_c)

        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_simsimd_backend_small_points(self) -> None:
        """Test that simsimd backend is used for small point sets."""
        # Small point set (10*10 = 100 < 1000) should use simsimd
        points1 = np.random.randn(10, 2).astype(np.float32)
        points2 = np.random.randn(10, 2).astype(np.float32)

        result = pairwise_distances_squared(points1, points2)

        # Verify correctness against numpy implementation
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
        """Test that numpy backend is used for large point sets."""
        # Large point set (100*100 = 10000 >= 1000) should use numpy
        points1 = np.random.randn(100, 2).astype(np.float32)
        points2 = np.random.randn(100, 2).astype(np.float32)

        result = pairwise_distances_squared(points1, points2)

        # Verify correctness by checking a few random distances
        for _ in range(5):
            i = np.random.randint(0, 100)
            j = np.random.randint(0, 100)
            expected_dist = np.sum((points1[i] - points2[j]) ** 2)
            np.testing.assert_allclose(result[i, j], expected_dist, rtol=1e-5)

    def test_edge_case_single_point(self) -> None:
        """Test with single points."""
        point1 = np.array([[1.0, 2.0]], dtype=np.float32)
        point2 = np.array([[4.0, 6.0]], dtype=np.float32)

        result = pairwise_distances_squared(point1, point2)

        # Distance from (1,2) to (4,6): sqrt((4-1)^2 + (6-2)^2) = sqrt(9+16) = 5
        # Squared distance: 25
        expected = np.array([[25.0]], dtype=np.float32)
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_known_distances(self) -> None:
        """Test with points of known distances."""
        # Unit square corners
        points1 = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
        points2 = np.array([[0, 0]], dtype=np.float32)

        result = pairwise_distances_squared(points1, points2)

        expected = np.array([[0.0], [1.0], [2.0], [1.0]], dtype=np.float32)
        np.testing.assert_allclose(result, expected, rtol=1e-5)
