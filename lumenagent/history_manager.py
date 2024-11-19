import json
import time
from typing import Dict, Any, List, Optional, Tuple
from collections import deque
import pandas as pd
import torch
from tabulate import tabulate

class HistoryManager:
    def __init__(self, max_size: int = 1000, auto_save_interval: int = 300):
        self._history: deque = deque(maxlen=max_size)
        self._redo_stack: List[Dict[str, Any]] = []
        self._locked_entries: Dict[str, bool] = {}
        self._auto_save_interval: int = auto_save_interval
        self._last_auto_save: float = time.time()

    def add(self, key: str, value: Any) -> None:
        """Add a new entry to the history."""
        if key in self._locked_entries and self._locked_entries[key]:
            raise ValueError(f"Entry '{key}' is locked and cannot be modified.")
        
        timestamp = time.time()
        entry = {"key": key, "value": value, "timestamp": timestamp, "versions": []}
        
        if self._history and self._history[-1]["key"] == key:
            self._history[-1]["versions"].append(self._history[-1]["value"])
            self._history[-1]["value"] = value
            self._history[-1]["timestamp"] = timestamp
        else:
            self._history.append(entry)
        
        self._redo_stack.clear()
        self._auto_save_check()

    def undo(self) -> Optional[Dict[str, Any]]:
        """Undo the last action."""
        if not self._history:
            return None
        
        last_entry = self._history.pop()
        self._redo_stack.append(last_entry)
        return last_entry

    def redo(self) -> Optional[Dict[str, Any]]:
        """Redo the last undone action."""
        if not self._redo_stack:
            return None
        
        last_undone = self._redo_stack.pop()
        self._history.append(last_undone)
        return last_undone

    def search(self, keyword: str) -> List[Dict[str, Any]]:
        """Search for entries containing the given keyword."""
        return [entry for entry in self._history 
                if keyword in str(entry["key"]) or keyword in str(entry["value"])]

    def revert_to_version(self, key: str, version_index: int) -> None:
        """Revert an entry to a specific version."""
        for entry in reversed(self._history):
            if entry["key"] == key:
                if 0 <= version_index < len(entry["versions"]):
                    entry["value"], entry["versions"][version_index] = (
                        entry["versions"][version_index],
                        entry["value"]
                    )
                    entry["timestamp"] = time.time()
                    return
                raise IndexError("Version index out of range")
        raise KeyError(f"Key '{key}' not found in history")

    def export_history(self, filename: str) -> None:
        """Export history to a JSON file."""
        with open(filename, 'w') as f:
            json.dump(list(self._history), f)

    def import_history(self, filename: str) -> None:
        """Import history from a JSON file."""
        with open(filename, 'r') as f:
            imported_history = json.load(f)
        self._history = deque(imported_history, maxlen=self._history.maxlen)
        self._redo_stack.clear()

    def lock_entry(self, key: str) -> None:
        """Lock an entry to prevent modifications."""
        self._locked_entries[key] = True

    def unlock_entry(self, key: str) -> None:
        """Unlock a previously locked entry."""
        self._locked_entries[key] = False

    def take_snapshot(self) -> Dict[str, Any]:
        """Take a snapshot of the current history state."""
        return {
            "history": list(self._history),
            "redo_stack": self._redo_stack.copy(),
            "locked_entries": self._locked_entries.copy()
        }

    def restore_snapshot(self, snapshot: Dict[str, Any]) -> None:
        """Restore history state from a snapshot."""
        self._history = deque(snapshot["history"], maxlen=self._history.maxlen)
        self._redo_stack = snapshot["redo_stack"]
        self._locked_entries = snapshot["locked_entries"]

    def filter_by_date(self, start_date: float, end_date: float) -> List[Dict[str, Any]]:
        """Filter history entries by date range."""
        return [entry for entry in self._history 
                if start_date <= entry["timestamp"] <= end_date]

    def _auto_save_check(self) -> None:
        """Check if it's time for an auto-save and perform if necessary."""
        current_time = time.time()
        if current_time - self._last_auto_save >= self._auto_save_interval:
            self.export_history("auto_save.json")
            self._last_auto_save = current_time

    def get_history_dataframe(self) -> pd.DataFrame:
        """Return history as a pandas DataFrame."""
        return pd.DataFrame(self._history)

    def get_history_tensor(self) -> torch.Tensor:
        """Return history timestamps as a PyTorch tensor."""
        timestamps = [entry["timestamp"] for entry in self._history]
        return torch.tensor(timestamps)

    def print_history(self) -> None:
        """Print history in a tabulated format."""
        headers = ["Key", "Value", "Timestamp"]
        table_data = [(entry["key"], str(entry["value"]), entry["timestamp"]) 
                      for entry in self._history]
        print(tabulate(table_data, headers=headers))

    def __len__(self) -> int:
        return len(self._history)

    def __getitem__(self, key: str) -> Any:
        for entry in reversed(self._history):
            if entry["key"] == key:
                return entry["value"]
        raise KeyError(f"Key '{key}' not found in history")

    def __setitem__(self, key: str, value: Any) -> None:
        self.add(key, value)

    def __iter__(self):
        return iter(self._history)

    def __contains__(self, key: str) -> bool:
        return any(entry["key"] == key for entry in self._history)

# # Create an instance of HistoryManager
# manager = HistoryManager(max_size=5)
# # Add some entries
# manager.add("entry1", {"value": 10})
# manager.add("entry2", {"value": 20})
# manager.add("entry3", {"value": 30})
# # Lock an entry and try to modify it
# manager.lock_entry("entry2")
# try:
#     manager.add("entry2", {"value": 25})  # This should raise an error
# except ValueError as e:
#     print(f"Error: {e}")
# # Unlock the entry and modify it
# manager.unlock_entry("entry2")
# manager.add("entry2", {"value": 25})
# # Undo the last action
# manager.undo()
# # Redo the undone action
# manager.redo()
# # Search for an entry
# search_results = manager.search("entry2")
# print("Search Results:", search_results)
# # Take a snapshot and restore from it
# snapshot = manager.take_snapshot()
# manager.add("entry4", {"value": 40})
# manager.restore_snapshot(snapshot)
# # Export and import history
# manager.export_history("history.json")
# manager.import_history("history.json")
# # Get history as pandas DataFrame and torch tensor
# df = manager.get_history_dataframe()
# tensor = manager.get_history_tensor()
# # Print the history
# manager.print_history()
# # Output the dataframe and tensor
# print("History DataFrame:\n", df)
# print("History Timestamps as Tensor:\n", tensor)
