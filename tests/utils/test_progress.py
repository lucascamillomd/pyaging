from pyaging.utils._utils import progress


class RecordingLogger:
    def __init__(self):
        self.events = []

    def start_progress(self, message, indent_level):
        self.events.append(("start", indent_level))

    def finish_progress(self, message, indent_level):
        self.events.append(("finish", indent_level))


@progress("work")
def _work(value, logger, indent_level=2):
    return value + 1


def test_progress_accepts_keyword_arguments_and_uses_declared_default_indent():
    logger = RecordingLogger()

    assert _work(value=4, logger=logger) == 5
    assert logger.events == [("start", 2), ("finish", 2)]


def test_progress_accepts_positional_indent():
    logger = RecordingLogger()

    assert _work(4, logger, 3) == 5
    assert logger.events == [("start", 3), ("finish", 3)]
