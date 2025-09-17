import time
import threading
from gnuradio import gr
import pmt

class message_strobe(gr.basic_block):
    """
    Python实现的Message Strobe Block
    定时向消息端口发送字符串
    """
    def __init__(self, period=0.1, msg="REPLAY"):
        gr.basic_block.__init__(
            self,
            name="Python Message Strobe",
            in_sig=None,
            out_sig=None
        )
        self.period = period
        self.msg = msg
        self.message_port_register_out(pmt.intern("strobe"))
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._strobe_loop)
        self._thread.daemon = True
        self._thread.start()

    def _strobe_loop(self):
        while not self._stop_event.is_set():
            self.message_port_pub(pmt.intern("strobe"), pmt.intern(self.msg))
            time.sleep(self.period)

    def stop(self):
        self._stop_event.set()
        self._thread.join()
        return True