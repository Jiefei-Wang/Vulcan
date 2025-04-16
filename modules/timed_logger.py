import logging
import time

class TimedLogger:
    def __init__(self):
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.start_time = None
        self.last_time = None
        self.reset_timer()

    def reset_timer(self):
        """Reset the timer for the first message."""
        self.start_time = time.time()
        self.last_time = self.start_time

    def log(self, message):
        """Log a message with elapsed time since the start and since the last message."""
        current_time = time.time()
        if self.start_time is None:
            self.reset_timer()
        elapsed_since_start = current_time - self.start_time
        elapsed_since_last = current_time - self.last_time
        self.logger.info(f"{message} (Elapsed: {elapsed_since_start:.2f}s, Since last: {elapsed_since_last:.2f}s)")
        self.last_time = current_time
    
    def done(self):
        self.log("All Jobs Done!")
        self.reset_timer()


logger = TimedLogger()