from PyQt5.QtCore import QObject, pyqtSlot
from PyQt5.QtCore import QTimer, QSocketNotifier
import signal
import os
import logging


class SignalHandler(QObject):

    """Handler responsible for handling OS signals (SIGINT, SIGTERM, etc.).

    Attributes:
        _activated: Whether activate() was called.
        _notifier: A QSocketNotifier used for signals on Unix.
        _timer: A QTimer used to poll for signals on Windows.
        _orig_handlers: A {signal: handler} dict of original signal handlers.
        _orig_wakeup_fd: The original wakeup filedescriptor.
    """

    def __init__(self, app, parent=None):
        super().__init__(parent)
        self._notifier = None
        self._timer = QTimer()
        self._orig_handlers = {}
        self._activated = False
        self._orig_wakeup_fd = None
        self._app = app

    def activate(self):
        """Set up signal handlers.

        On Windows this uses a QTimer to periodically hand control over to
        Python so it can handle signals.

        On Unix, it uses a QSocketNotifier with os.set_wakeup_fd to get
        notified.
        """
        self._orig_handlers[signal.SIGINT] = signal.signal(
            signal.SIGINT, self.interrupt)
        self._orig_handlers[signal.SIGTERM] = signal.signal(
            signal.SIGTERM, self.interrupt)

        if os.name == 'posix' and hasattr(signal, 'set_wakeup_fd'):
            # pylint: disable=import-error,no-member,useless-suppression
            import fcntl
            read_fd, write_fd = os.pipe()
            for fd in (read_fd, write_fd):
                flags = fcntl.fcntl(fd, fcntl.F_GETFL)
                fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
            self._notifier = QSocketNotifier(
                read_fd, QSocketNotifier.Read, self)
            self._notifier.activated.connect(self.handle_signal_wakeup)
            self._orig_wakeup_fd = signal.set_wakeup_fd(write_fd)
        else:
            self._timer.start(1000)
            self._timer.timeout.connect(lambda: None)
        self._activated = True

    def deactivate(self):
        """Deactivate all signal handlers."""
        if not self._activated:
            return
        if self._notifier is not None:
            self._notifier.setEnabled(False)
            rfd = self._notifier.socket()
            wfd = signal.set_wakeup_fd(self._orig_wakeup_fd)
            os.close(rfd)
            os.close(wfd)
        for sig, handler in self._orig_handlers.items():
            signal.signal(sig, handler)
        self._timer.stop()
        self._activated = False

    @pyqtSlot()
    def handle_signal_wakeup(self):
        """Handle a newly arrived signal.

        This gets called via self._notifier when there's a signal.

        Python will get control here, so the signal will get handled.
        """
        logging.debug("Handling signal wakeup!")
        self._notifier.setEnabled(False)
        read_fd = self._notifier.socket()
        try:
            os.read(read_fd, 1)
        except OSError:
            logging.exception("Failed to read wakeup fd.")
        self._notifier.setEnabled(True)

    def interrupt(self, signum, _frame):
        """Handler for signals to gracefully shutdown (SIGINT/SIGTERM)."""
        print("Hello interrupt here! signum = {}; \
            frame = {}".format(signum, _frame))
        self._app.signal_quit_app.emit(signum)
