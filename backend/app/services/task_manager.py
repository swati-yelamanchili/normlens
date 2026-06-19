import logging
import threading
import time
import uuid
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class TaskStatus:
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskManager:
    def __init__(self):
        self._tasks: Dict[str, dict] = {}
        self._lock = threading.Lock()

    def create_task(self, task_type: str) -> str:
        task_id = str(uuid.uuid4())
        with self._lock:
            self._tasks[task_id] = {
                "id": task_id,
                "type": task_type,
                "status": TaskStatus.PENDING,
                "result": None,
                "error": None,
                "created_at": time.time(),
                "completed_at": None,
            }
        return task_id

    def start_task(self, task_id: str, target, args=None, kwargs=None):
        with self._lock:
            if task_id in self._tasks:
                self._tasks[task_id]["status"] = TaskStatus.RUNNING

        thread = threading.Thread(
            target=self._run_task,
            args=(task_id, target, args or (), kwargs or {}),
            daemon=True,
        )
        thread.start()

    def _run_task(self, task_id: str, target, args, kwargs):
        try:
            result = target(*args, **kwargs)
            with self._lock:
                self._tasks[task_id]["status"] = TaskStatus.COMPLETED
                self._tasks[task_id]["result"] = result
                self._tasks[task_id]["completed_at"] = time.time()
        except Exception as e:
            logger.exception("Task %s failed", task_id)
            with self._lock:
                self._tasks[task_id]["status"] = TaskStatus.FAILED
                self._tasks[task_id]["error"] = str(e)
                self._tasks[task_id]["completed_at"] = time.time()

    def get_task(self, task_id: str) -> Optional[dict]:
        with self._lock:
            return self._tasks.get(task_id)

    def get_status(self, task_id: str) -> Optional[str]:
        task = self.get_task(task_id)
        return task["status"] if task else None


_task_manager_instance: Optional[TaskManager] = None


def get_task_manager() -> TaskManager:
    global _task_manager_instance
    if _task_manager_instance is None:
        _task_manager_instance = TaskManager()
    return _task_manager_instance
