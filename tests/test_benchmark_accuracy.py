import unittest
from src.benchmark import run_benchmark


class TestBenchmarkAccuracy(unittest.TestCase):
    def test_benchmark_suite_accuracy(self):
        report = run_benchmark(verbose=False)
        self.assertEqual(report.failed, 0, f"Benchmark had {report.failed} failures.")
        self.assertGreaterEqual(report.accuracy, 95.0, f"Measured accuracy below 95%: {report.accuracy}%")
        self.assertEqual(report.total, 30)


if __name__ == "__main__":
    unittest.main()
