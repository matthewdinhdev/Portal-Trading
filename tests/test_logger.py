import unittest
import os
import logging
from datetime import datetime
from src.logger import setup_logger


class TestLogger(unittest.TestCase):
    def setUp(self):
        """Set up test environment before each test"""
        # Create logs directory if it doesn't exist
        self.logs_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
        os.makedirs(self.logs_dir, exist_ok=True)

        # Use absolute path for test log file
        self.test_log_file = os.path.join(self.logs_dir, "test.log")
        self.logger = setup_logger(self.test_log_file)

    def tearDown(self):
        """Clean up test environment after each test"""
        # Close all handlers
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)

        # Remove test log file if it exists
        date_str = datetime.now().strftime("%Y%m%d")
        dated_log_file = os.path.join(self.logs_dir, f"test_{date_str}.log")
        if os.path.exists(dated_log_file):
            os.remove(dated_log_file)

    def test_log_file_creation(self):
        """Test that log file is created"""
        # Log a message to ensure file is created
        self.logger.info("Test message")

        # Flush handlers to ensure file is written
        for handler in self.logger.handlers:
            handler.flush()

        # Check for dated log file
        date_str = datetime.now().strftime("%Y%m%d")
        dated_log_file = os.path.join(self.logs_dir, f"test_{date_str}.log")
        self.assertTrue(os.path.exists(dated_log_file))

    def test_log_levels(self):
        """Test different logging levels"""
        # Test each logging level
        self.logger.debug("Debug message")
        self.logger.info("Info message")
        self.logger.warning("Warning message")
        self.logger.error("Error message")

        # Flush handlers
        for handler in self.logger.handlers:
            handler.flush()

        # Read log file content
        date_str = datetime.now().strftime("%Y%m%d")
        dated_log_file = os.path.join(self.logs_dir, f"test_{date_str}.log")
        with open(dated_log_file, "r") as f:
            log_content = f.read()

        # Verify each level was logged correctly
        self.assertIn("INFO - Info message", log_content)
        self.assertIn("WARNING - Warning message", log_content)
        self.assertIn("ERROR - Error message", log_content)
        # Debug messages shouldn't be logged since level is INFO
        self.assertNotIn("DEBUG - Debug message", log_content)

    def test_log_format(self):
        """Test that log entries follow the correct format"""
        # Log a test message
        test_message = "Test message"
        self.logger.info(test_message)

        # Flush handlers
        for handler in self.logger.handlers:
            handler.flush()

        # Read log file content
        date_str = datetime.now().strftime("%Y%m%d")
        dated_log_file = os.path.join(self.logs_dir, f"test_{date_str}.log")
        with open(dated_log_file, "r") as f:
            log_lines = f.readlines()

        # Verify log format
        for line in log_lines:
            # Check timestamp format
            self.assertRegex(line, r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} - ")

            # Check log level
            self.assertTrue(any(level in line for level in ["INFO", "ERROR", "WARNING"]))

            # Check message format
            self.assertIn(" - ", line)

    def test_multiple_loggers(self):
        """Test that multiple loggers write to the same file"""
        # Create a second logger
        logger2 = setup_logger(self.test_log_file)

        # Log messages from both loggers
        self.logger.info("Message from logger 1")
        logger2.info("Message from logger 2")

        # Flush handlers
        for handler in self.logger.handlers:
            handler.flush()
        for handler in logger2.handlers:
            handler.flush()

        # Read log file content
        date_str = datetime.now().strftime("%Y%m%d")
        dated_log_file = os.path.join(self.logs_dir, f"test_{date_str}.log")
        with open(dated_log_file, "r") as f:
            log_content = f.read()

        # Verify both messages were logged
        self.assertIn("Message from logger 1", log_content)
        self.assertIn("Message from logger 2", log_content)


if __name__ == "__main__":
    unittest.main()
