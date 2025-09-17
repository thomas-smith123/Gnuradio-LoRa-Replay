import numpy as np
from gnuradio import gr
import pmt
import zmq,time
textboxValue = ""

class my_sync_block(gr.sync_block):
    """
    reads input from a message port
    outputs text
    """
    def __init__(self,delay=10):
        gr.sync_block.__init__(self,
            name = "LoRa Source",
            in_sig = None,
            out_sig = [np.complex64],
            )
        file_name = '/home/jiangrd3/code/lora_test/dataset/'
        self.set_output_multiple(8192)
        self.label_t = np.load(file_name+'label.npy')
        self.data_t = np.load(file_name+'data.npy')
        a, self.b =  self.data_t.shape
        print('Opened')
        self.delay = delay
        self.count = 0
        self.max = self.label_t.shape[1]
        print("label shape:", self.label_t.shape, "max:", self.max)
        # 初始化ZMQ
        self.context = zmq.Context()

        self.last_send_time = time.time()
        self.ready_to_read = False
        self.reread = False
        self.replay = 0
        self.message_port_register_in(pmt.intern("trigger"))
        self.set_msg_handler(pmt.intern("trigger"), self.handle_trigger)
        self.trig_mark = False
    def work(self, input_items, output_items):
        if time.time()-self.last_send_time>self.delay:

            print(f"{time.time()-self.last_send_time} s since last send")
            if self.count is None or self.max is None:
                print("[ERROR] self.count or self.max is None!")
                return -1
            
            # 转换数据并输出
            if self.ready_to_read:
                samples = self.data_t[self.count, :self.b//2]+1j*self.data_t[self.count, self.b//2:]
                #for x in range(8192):
                #    output_items[0][x] = samples[x]
                output_items[0][:8192] = samples[:8192]
                #output_items[0][:8192] = np.array(samples)
                self.ready_to_read = False
                self.reread = False
                # print(f"已发送一个波形，长度: {len(samples)} 个样本")
                self.count = self.count+1
                self.replay = 0
                self.last_send_time = time.time()
                print(f'Next wave: {self.count}')
                self.trig_mark = False
                return 8192
            
            if self.reread:
                self.reread = False
                samples = self.data_t[self.count, :self.b//2]+1j*self.data_t[self.count, self.b//2:]

                output_items[0][:8192] = samples[:8192]
                self.ready_to_read = False
                self.last_send_time = time.time()
                print(f'Replay wave: {self.replay}')
                self.replay += 1
                # print(f"已发送一个波形，长度: {len(samples)} 个样本")
                return 8192
            
            if self.count >= self.max:
                print("已到达文件末尾")
                return -1
        else:
            output_items[0][:8192] = np.zeros(8192, dtype=np.complex64)
            # self.last_send_time = time.time()
            return 8192

    def stop(self):
        if hasattr(self, 'file') and self.file:
            self.file.close()
        if hasattr(self, 'socket'):
            self.socket.close()
        if hasattr(self, 'context'):
            self.context.term()
        return True
    
    def handle_trigger(self, msg):
        if not self.trig_mark:
            msg = pmt.symbol_to_string(msg)
            if msg == "NEXT":
                # print(msg)
                self.ready_to_read = True
                self.reread = False
                self.trig_mark = True
                # print("收到消息端口触发信号，准备读取下一个波形")
            if msg == "REPLAY":
                # print(msg)
                self.reread = True
                self.ready_to_read = False
            