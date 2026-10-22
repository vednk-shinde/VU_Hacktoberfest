import tempfile
import unittest
from pathlib import Path
from src.engine import QAEngine
from src.registry.models import DatasetStatus
from src.registry.service import DatasetRegistryService
from src.registry.store import DatasetRegistryStore
from src.storage.manager import StorageManager, compute_sha256, ensure_within_directory, sanitize_filename
from src.security.permissions import UserContext


class TestDatasetLifecycle(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.storage_root = Path(self.temp_dir.name) / "storage"
        self.storage_mgr = StorageManager(base_dir=self.storage_root)
        self.store = DatasetRegistryStore(db_path=self.storage_root / "registry.db")
        self.service = DatasetRegistryService(store=self.store, storage_manager=self.storage_mgr)

        self.sample_csv = "data/sample.csv"
        self.sample_json = "data/sample.json"
        self.synth_csv = "data/synthetic_employees.csv"

    def tearDown(self):
        self.temp_dir.cleanup()

    # --- 1. Operation 1: Add Dataset ---

    def test_add_dataset_success(self):
        ver = self.service.add_dataset(
            file_path=self.sample_csv,
            name="Employee Data",
            description="Initial employee sample",
            dataset_id="test_emp_01",
        )

        self.assertEqual(ver.dataset_id, "test_emp_01")
        self.assertEqual(ver.version, 1)
        self.assertEqual(ver.status, DatasetStatus.ACTIVE)
        self.assertEqual(ver.format, "csv")
        self.assertTrue(Path(ver.storage_path).exists())

        # Checksum validation
        actual_checksum = compute_sha256(ver.storage_path)
        self.assertEqual(ver.checksum_sha256, actual_checksum)

        # Timezone-aware ISO timestamp validation
        self.assertIn("+00:00", ver.upload_timestamp)

        # Schema profile integrity
        self.assertEqual(ver.schema_profile["row_count"], 5)
        self.assertEqual(sorted(ver.schema_profile["columns"]), sorted(["employee", "department", "salary"]))

        # Registry retrieval check
        active = self.service.get_active_dataset("test_emp_01")
        self.assertEqual(active.version, 1)
        self.assertEqual(active.status, DatasetStatus.ACTIVE)

    def test_duplicate_dataset_id_rejected(self):
        self.service.add_dataset(self.sample_csv, name="First", dataset_id="dup_id")
        with self.assertRaises(ValueError) as ctx:
            self.service.add_dataset(self.sample_csv, name="Duplicate", dataset_id="dup_id")
        self.assertIn("already exists", str(ctx.exception))

    # --- 2. Path Traversal & Filename Sanitization ---

    def test_path_traversal_sanitization(self):
        # Traversal names should be stripped to simple basenames
        self.assertEqual(sanitize_filename("../../etc/passwd"), "passwd")
        self.assertEqual(sanitize_filename("..\\..\\Windows\\System32\\cmd.exe"), "cmd.exe")
        self.assertEqual(sanitize_filename("foo/bar/dataset.csv"), "dataset.csv")

    def test_path_boundary_enforcement(self):
        base = self.storage_root / "safe_dir"
        base.mkdir(parents=True, exist_ok=True)

        inside = base / "valid.csv"
        self.assertEqual(ensure_within_directory(base, inside), inside.resolve())

        outside = self.storage_root / "outside.csv"
        with self.assertRaises(ValueError) as ctx:
            ensure_within_directory(base, outside)
        self.assertIn("Path traversal detected", str(ctx.exception))

    # --- 3. Operation 2: Staged Replace Dataset ---

    def test_replace_dataset_success_and_superseding(self):
        # Add version 1
        v1 = self.service.add_dataset(self.sample_csv, name="Products", dataset_id="products_ds")
        self.assertEqual(v1.version, 1)
        self.assertEqual(v1.status, DatasetStatus.ACTIVE)
        path_v1 = Path(v1.storage_path)

        # Replace with synthetic employee dataset as version 2
        v2 = self.service.replace_dataset("products_ds", self.synth_csv)
        self.assertEqual(v2.version, 2)
        self.assertEqual(v2.status, DatasetStatus.ACTIVE)
        path_v2 = Path(v2.storage_path)

        # Both files must coexist on disk in separate version directories
        self.assertTrue(path_v1.exists(), "Previous version file must be preserved")
        self.assertTrue(path_v2.exists(), "New version file must exist")
        self.assertNotEqual(path_v1, path_v2)

        # Active version is now version 2
        active = self.service.get_active_dataset("products_ds")
        self.assertEqual(active.version, 2)

        # Version 1 is marked as SUPERSEDED in history
        history = self.service.get_version_history("products_ds")
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0].version, 1)
        self.assertEqual(history[0].status, DatasetStatus.SUPERSEDED)
        self.assertEqual(history[1].version, 2)
        self.assertEqual(history[1].status, DatasetStatus.ACTIVE)

        # Schema drift was recorded in provenance
        self.assertIn("schema_diff", history[1].provenance)
        self.assertIn("added_columns", history[1].provenance["schema_diff"])

    # --- 4. Failed Replacement & Rollback Mechanism ---

    def test_failed_replacement_rollback_preserves_active_version(self):
        # 1. Add valid initial version
        v1 = self.service.add_dataset(self.sample_csv, name="Employees", dataset_id="emp_rollback")
        self.assertEqual(v1.version, 1)
        self.assertEqual(v1.status, DatasetStatus.ACTIVE)

        # 2. Create an invalid/malformed candidate file
        bad_file = Path(self.temp_dir.name) / "corrupt.json"
        bad_file.write_text("{this is not valid json")

        # 3. Attempt replacement
        with self.assertRaises(ValueError) as ctx:
            self.service.replace_dataset("emp_rollback", bad_file)
        self.assertIn("Replacement validation failed", str(ctx.exception))

        # 4. CRUCIAL ROLLBACK CHECKS:
        # Version 1 must STILL be active
        active = self.service.get_active_dataset("emp_rollback")
        self.assertEqual(active.version, 1)
        self.assertEqual(active.status, DatasetStatus.ACTIVE)
        self.assertTrue(Path(active.storage_path).exists())

        # Version 2 must be recorded as FAILED
        history = self.service.get_version_history("emp_rollback")
        self.assertEqual(len(history), 2)
        v2 = history[1]
        self.assertEqual(v2.version, 2)
        self.assertEqual(v2.status, DatasetStatus.FAILED)
        self.assertIsNotNone(v2.error_message)

        # Staging directory must be cleaned up
        stage_dir = self.storage_mgr.get_staging_dir("emp_rollback", 2)
        self.assertFalse(stage_dir.exists())

    # --- 5. Dataset Isolation ---

    def test_dataset_isolation(self):
        self.service.add_dataset(self.sample_csv, name="Dataset A", dataset_id="ds_a")
        self.service.add_dataset(self.sample_json, name="Dataset B", dataset_id="ds_b")

        # Replace Dataset A
        self.service.replace_dataset("ds_a", self.synth_csv)

        # Dataset B must be completely unaffected
        b_active = self.service.get_active_dataset("ds_b")
        self.assertEqual(b_active.version, 1)
        self.assertEqual(b_active.status, DatasetStatus.ACTIVE)
        self.assertEqual(b_active.schema_profile["row_count"], 3)

    # --- 6. QA Engine Integration with Registered Datasets ---

    def test_qa_engine_uses_active_version_and_updates_on_replacement(self):
        # Register dataset
        self.service.add_dataset(self.sample_csv, name="Emp", dataset_id="emp_qa")

        from src.security.permissions import UserContext
        engine = QAEngine(registry_service=self.service)

        # Query v1
        user = UserContext(user_id="qa", allowed_datasets=["emp_qa"])
        denied = engine.ask("How many rows are in the dataset?", dataset_path="emp_qa")
        self.assertEqual(denied.status, "abstained")
        self.assertIn("Access denied", denied.reason)
        res_v1 = engine.ask("How many rows are in the dataset?", dataset_path="emp_qa", user_context=user)
        self.assertEqual(res_v1.status, "answered")
        self.assertEqual(res_v1.answer, 5)
        self.assertEqual(res_v1.evidence["dataset_id"], "emp_qa")
        self.assertEqual(res_v1.evidence["dataset_version"], 1)

        # Replace with 10-row synthetic dataset
        self.service.replace_dataset("emp_qa", self.synth_csv)

        # Query again without changing caller code: automatically hits active v2
        res_v2 = engine.ask("How many rows are in the dataset?", dataset_path="emp_qa", user_context=user)
        self.assertEqual(res_v2.status, "answered")
        self.assertEqual(res_v2.answer, 10)
        self.assertEqual(res_v2.evidence["dataset_id"], "emp_qa")
        self.assertEqual(res_v2.evidence["dataset_version"], 2)

    def test_registry_errors_do_not_fall_back_to_file_paths(self):
        engine = QAEngine(registry_service=self.service)
        with self.assertRaises(KeyError):
            engine.load("not_registered")

        self.store.get_version = lambda *_args: (_ for _ in ()).throw(RuntimeError("registry corruption"))
        with self.assertRaisesRegex(RuntimeError, "registry corruption"):
            engine.load("emp_qa", version=1)

    def test_registered_file_path_cannot_bypass_dataset_id_authorization(self):
        ver = self.service.add_dataset(self.sample_csv, name="Managed", dataset_id="managed")
        engine = QAEngine(registry_service=self.service, approved_file_roots=[self.storage_root])
        user = UserContext(user_id="qa", allowed_datasets=["managed"])
        result = engine.ask("How many rows?", dataset_path=ver.storage_path, user_context=user)
        self.assertEqual(result.status, "abstained")
        self.assertIn("registered dataset ID", result.reason)


if __name__ == "__main__":
    unittest.main()
