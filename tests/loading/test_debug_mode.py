import unittest

from rising.loading import get_debug_mode, set_debug_mode, switch_debug_mode


class TestDebugMode(unittest.TestCase):
    def test_debug_mode(self):
        set_debug_mode(False)
        self.assertFalse(get_debug_mode())
        set_debug_mode(True)
        self.assertTrue(get_debug_mode())
        set_debug_mode(False)
        self.assertFalse(get_debug_mode())

    def test_switch_debug_mode(self):
        set_debug_mode(False)
        self.assertFalse(get_debug_mode())
        switch_debug_mode()
        self.assertTrue(get_debug_mode())
        switch_debug_mode()
        self.assertFalse(get_debug_mode())
