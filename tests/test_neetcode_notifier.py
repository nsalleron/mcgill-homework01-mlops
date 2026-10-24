"""
Tests for NeetCode Spaced-Repetition Notifier.
"""

from __future__ import annotations

import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

from scripts.neetcode_notifier.config import Config
from scripts.neetcode_notifier.catalog import ProblemCatalog, normalize_key
from scripts.neetcode_notifier.neetcode_client import NeetCodeClient
from scripts.neetcode_notifier.notifier import NtfyNotifier
from scripts.neetcode_notifier.main import select_random_problem
from scripts.neetcode_notifier.scheduler import RandomDailyScheduler


class TestNeetCodeCatalog(unittest.TestCase):
    def setUp(self):
        self.catalog = ProblemCatalog()

    def test_catalog_loaded(self):
        self.assertGreater(len(self.catalog._all_problems), 100)

    def test_normalize_key(self):
        self.assertEqual(normalize_key("Group Anagrams"), "groupanagrams")
        self.assertEqual(normalize_key("0049-group-anagrams"), "0049groupanagrams")
        self.assertEqual(normalize_key("3Sum"), "3sum")

    def test_get_problem(self):
        # By title
        p1 = self.catalog.get_problem("3Sum")
        self.assertIsNotNone(p1)
        self.assertEqual(p1["difficulty"], "Medium")

        # By slug
        p2 = self.catalog.get_problem("group-anagrams")
        self.assertIsNotNone(p2)
        self.assertEqual(p2["difficulty"], "Medium")

        # By code with number
        p3 = self.catalog.get_problem("0049-group-anagrams")
        self.assertIsNotNone(p3)
        self.assertEqual(p3["name"], "Group Anagrams")

    def test_filter_medium_hard(self):
        problems = [
            "contains-duplicate",  # Easy
            "3sum",                # Medium
            "trapping-rain-water", # Hard
            "group-anagrams",      # Medium
            "3sum",                # Duplicate Medium
            "trapping-rain-water", # Duplicate Hard
        ]
        results = self.catalog.filter_medium_hard_problems(problems)
        self.assertEqual(len(results), 3)  # 2 medium + 1 hard, duplicates deduplicated
        names = {m["name"] for m in results}
        self.assertIn("3Sum", names)
        self.assertIn("Group Anagrams", names)
        self.assertIn("Trapping Rain Water", names)
        self.assertNotIn("Contains Duplicate", names)

    def test_filter_medium(self):
        problems = [
            "contains-duplicate",  # Easy
            "3sum",                # Medium
            "trapping-rain-water", # Hard
            "group-anagrams",      # Medium
        ]
        mediums = self.catalog.filter_medium_problems(problems)
        self.assertEqual(len(mediums), 2)
        names = {m["name"] for m in mediums}
        self.assertIn("3Sum", names)
        self.assertIn("Group Anagrams", names)


class TestNeetCodeClientExtraction(unittest.TestCase):
    def test_extract_dict_of_lists(self):
        raw = {
            "arrays-and-hashing": ["contains-duplicate", "group-anagrams"],
            "two-pointers": ["two-sum-ii", "3sum"],
        }
        extracted = NeetCodeClient.extract_problem_identifiers(raw)
        self.assertEqual(
            extracted,
            ["3sum", "contains-duplicate", "group-anagrams", "two-sum-ii"],
        )

    def test_extract_dict_of_bools(self):
        raw = {
            "two-sum": True,
            "3sum": True,
            "n-queens": False,
        }
        extracted = NeetCodeClient.extract_problem_identifiers(raw)
        self.assertEqual(extracted, ["3sum", "two-sum"])

    def test_extract_flat_list(self):
        raw = ["group-anagrams", "top-k-frequent-elements"]
        extracted = NeetCodeClient.extract_problem_identifiers(raw)
        self.assertEqual(extracted, ["group-anagrams", "top-k-frequent-elements"])


class TestTimeWindow(unittest.TestCase):
    def test_is_within_time_window(self):
        # 14h is inside [9, 21)
        dt_inside = datetime(2026, 10, 22, 14, 30)
        self.assertTrue(Config.is_within_time_window(dt_inside))

        # 8h is outside [9, 21)
        dt_before = datetime(2026, 10, 22, 8, 45)
        self.assertFalse(Config.is_within_time_window(dt_before))

        # 22h is outside [9, 21)
        dt_after = datetime(2026, 10, 22, 22, 15)
        self.assertFalse(Config.is_within_time_window(dt_after))


class TestProblemSelectionAndRotation(unittest.TestCase):
    def test_select_random_problem_avoids_recent_history(self):
        problems = [
            {"name": "3Sum", "slug": "3sum", "difficulty": "Medium"},
            {"name": "Group Anagrams", "slug": "group-anagrams", "difficulty": "Medium"},
            {"name": "Sort an Array", "slug": "sort-an-array", "difficulty": "Medium"},
        ]
        history = ["3sum", "group-anagrams"]

        with patch("scripts.neetcode_notifier.main.save_history"):
            chosen = select_random_problem(problems, history)
            self.assertEqual(chosen["slug"], "sort-an-array")

    def test_rotation_resets_when_all_problems_used(self):
        problems = [
            {"name": "3Sum", "slug": "3sum", "difficulty": "Medium"},
            {"name": "Group Anagrams", "slug": "group-anagrams", "difficulty": "Medium"},
        ]
        history = ["3sum", "group-anagrams"]

        with patch("scripts.neetcode_notifier.main.save_history"):
            chosen = select_random_problem(problems, history)
            self.assertIn(chosen["slug"], ["3sum", "group-anagrams"])


class TestRandomDailyScheduler(unittest.TestCase):
    def setUp(self):
        self.tmp_path = Path("/tmp/test_neetcode_schedule.json")
        if self.tmp_path.is_file():
            self.tmp_path.unlink()
        self.scheduler = RandomDailyScheduler(start_hour=9, end_hour=21, state_path=self.tmp_path)

    def tearDown(self):
        if self.tmp_path.is_file():
            self.tmp_path.unlink()

    def test_daily_target_generation(self):
        dt = datetime(2026, 10, 23, 10, 0)
        target = self.scheduler.get_or_create_daily_target(dt)
        self.assertEqual(target["date"], "2026-10-23")
        self.assertGreaterEqual(target["target_hour"], 9)
        self.assertLessEqual(target["target_hour"], 20)
        self.assertGreaterEqual(target["target_minute"], 0)
        self.assertLessEqual(target["target_minute"], 59)

        # Calling again on the same day returns the identical target time
        target2 = self.scheduler.get_or_create_daily_target(dt)
        self.assertEqual(target["target_hour"], target2["target_hour"])
        self.assertEqual(target["target_minute"], target2["target_minute"])

    def test_should_trigger_timing(self):
        # Target set at 14h30
        self.scheduler.save_state({
            "date": "2026-10-23",
            "target_hour": 14,
            "target_minute": 30,
            "sent": False,
            "sent_at": None,
        })

        # At 12h00 -> Too early
        should, _ = self.scheduler.should_trigger_now(datetime(2026, 10, 23, 12, 0))
        self.assertFalse(should)

        # At 14h35 -> Ready to trigger
        should, _ = self.scheduler.should_trigger_now(datetime(2026, 10, 23, 14, 35))
        self.assertTrue(should)

        # Mark sent
        self.scheduler.mark_as_sent(datetime(2026, 10, 23, 14, 35))

        # At 15h00 -> Already sent today
        should, _ = self.scheduler.should_trigger_now(datetime(2026, 10, 23, 15, 0))
        self.assertFalse(should)


class TestNtfyNotifier(unittest.TestCase):
    def test_dry_run(self):
        notifier = NtfyNotifier(topic="test-topic")
        res = notifier.send_random_medium_reminder(
            problem={
                "name": "3Sum",
                "pattern": "Two Pointers",
                "neetcode_url": "https://neetcode.io/problems/3sum",
                "leetcode_url": "https://leetcode.com/problems/3sum/",
            },
            total_completed_medium=15,
            dry_run=True,
        )
        self.assertTrue(res)


if __name__ == "__main__":
    unittest.main()
