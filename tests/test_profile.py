
import unittest

from src.ingestion.profile import profile_dataset


class TestDatasetProfile(unittest.TestCase):

    def test_csv_profile(self):
        result = profile_dataset("data/sample.csv")

        self.assertEqual(result["row_count"], 5)
        self.assertEqual(result["column_count"], 3)
        self.assertEqual(result["missing_values"]["salary"], 1)
        self.assertEqual(result["duplicate_rows"], 1)

    def test_json_profile(self):
        result = profile_dataset("data/sample.json")

        self.assertEqual(result["row_count"], 3)
        self.assertEqual(result["column_count"], 3)
        self.assertEqual(result["missing_values"]["stock"], 1)

    def test_unsupported_format(self):
        with self.assertRaises(ValueError):
            profile_dataset("data/sample.txt")


if __name__ == "__main__":
    unittest.main()
