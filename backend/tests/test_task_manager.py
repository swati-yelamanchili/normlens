import time
import pytest
from app.services.task_manager import TaskManager, TaskStatus


class TestTaskManager:
    def setup_method(self):
        self.manager = TaskManager()

    def test_create_task(self):
        task_id = self.manager.create_task("test")
        assert task_id is not None
        task = self.manager.get_task(task_id)
        assert task["status"] == TaskStatus.PENDING
        assert task["type"] == "test"

    def test_get_status_pending(self):
        task_id = self.manager.create_task("test")
        assert self.manager.get_status(task_id) == TaskStatus.PENDING

    def test_get_status_nonexistent(self):
        assert self.manager.get_status("nonexistent") is None

    def test_run_successful_task(self):
        task_id = self.manager.create_task("test")

        def successful():
            return 42

        self.manager.start_task(task_id, successful)
        time.sleep(0.1)
        task = self.manager.get_task(task_id)
        assert task["status"] == TaskStatus.COMPLETED
        assert task["result"] == 42

    def test_run_failing_task(self):
        task_id = self.manager.create_task("test")

        def failing():
            raise ValueError("Task failed")

        self.manager.start_task(task_id, failing)
        time.sleep(0.1)
        task = self.manager.get_task(task_id)
        assert task["status"] == TaskStatus.FAILED
        assert "Task failed" in task["error"]

    def test_multiple_tasks(self):
        ids = [self.manager.create_task(f"task_{i}") for i in range(5)]
        assert len(set(ids)) == 5
        for tid in ids:
            assert self.manager.get_status(tid) == TaskStatus.PENDING

    def test_task_with_args(self):
        task_id = self.manager.create_task("test")

        def adder(a, b):
            return a + b

        self.manager.start_task(task_id, adder, args=(2, 3))
        time.sleep(0.1)
        task = self.manager.get_task(task_id)
        assert task["result"] == 5
