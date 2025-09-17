# """
# Embedded Python Blocks:

# Each time this file is saved, GRC will instantiate the first class it finds
# to get ports and parameters of your block. The arguments to __init__  will
# be the parameters. All of them are required to have default values!
# """

# import numpy as np
# from gnuradio import gr
# import pmt
# import time

# class blk(gr.sync_block):
#     """LoRa Preamble Detector - Energy Detection Only"""

#     def __init__(self, sample_rate=1e6, bandwidth=125e3, sf=7, threshold=0.2):
#         gr.sync_block.__init__(
#             self,
#             name='LoRa Preamble Detector (Energy Only)',
#             in_sig=[np.complex64],
#             out_sig=[np.complex64]
#         )
#         self.sample_rate = sample_rate
#         self.bandwidth = bandwidth
#         self.sf = sf
#         self.threshold = threshold
        
#         # 能量检测参数
#         self.window_size = 128  # 能量计算窗口
#         self.min_high_len = 256  # 最小高能量持续时间
#         self.capture_len = 8192  # 捕获长度
#         self.energy_threshold_factor = 5  # 能量阈值倍数
        
#         # 状态变量
#         self.last_trigger_time = 0
#         self.trigger_interval = 0.1  # 最小触发间隔(秒)
#         self.detection_count = 0
        
#         # 设置输出多倍数
#         self.set_output_multiple(self.capture_len*2)
        
#         # 注册消息输出端口
#         self.message_port_register_out(pmt.intern("trigger"))
        
#         print("LoRa能量检测器初始化完成")

#     def work(self, input_items, output_items):
#         in0 = input_items[0]
#         out = output_items[0]
#         n_input = len(in0)
#         # output_items[0][:] = in0
#         # 透传数据
#         out[:n_input] = in0
        
#         # 如果输入数据太少，直接返回
#         if n_input < self.window_size + self.min_high_len:
#             return n_input

#         # 1. 计算滑动能量
#         energy = np.convolve(np.abs(in0)**2, np.ones(self.window_size), 'valid') / self.window_size
        
#         # 2. 估计噪声基底（使用中位数更鲁棒）
#         noise_floor = np.median(energy)
#         threshold = noise_floor * self.energy_threshold_factor
        
#         # 3. 检测能量超过阈值的区域
#         above_threshold = energy > threshold
        
#         # 4. 寻找连续的高能量区域
#         # 使用差分找到上升沿和下降沿
#         diff_above = np.diff(above_threshold.astype(int))
#         rise_edges = np.where(diff_above == 1)[0]  # 上升沿
#         fall_edges = np.where(diff_above == -1)[0]  # 下降沿
        
#         # 处理边界情况
#         if len(rise_edges) > len(fall_edges):
#             fall_edges = np.append(fall_edges, len(above_threshold) - 1)
        
#         # 5. 检查每个高能量区域
#         for i in range(len(rise_edges)):
#             start_idx = rise_edges[i]
#             end_idx = fall_edges[i] if i < len(fall_edges) else len(above_threshold) - 1
            
#             # 检查持续时间是否足够长
#             duration = end_idx - start_idx
#             if duration >= self.min_high_len:
#                 # 检查触发间隔
#                 current_time = time.time()
#                 if current_time - self.last_trigger_time > self.trigger_interval:
#                     # 确保有足够的数据进行捕获
#                     if start_idx + self.capture_len <= n_input:
#                         # 发送触发消息
#                         self.message_port_pub(pmt.intern("trigger"), pmt.intern("NEXT"))
#                         self.last_trigger_time = current_time
#                         self.detection_count += 1
                        
#                         print(f"检测到信号 #{self.detection_count}, 能量: {np.mean(energy[start_idx:end_idx]):.2f}, 噪声基底: {noise_floor:.2f}")
                        
#                         # 返回已处理的数据量
#                         return n_input
#                     else:
#                         # 数据不足，等待更多数据
#                         print("1")
#                 else:
#                     print("2")
#             else:
#                 print("3")
#         self.message_port_pub(pmt.intern("trigger"), pmt.intern("REPLAY"))            
        
#         return n_input

#     #     output_items[0][:] = 0
#     #     current_time = time.time()
#     #     threshold = self.threshold
#     #     step = max(1, self.symbol_length // 32)
#     #     correlations = []
#     #     positions = []
#     #     for i in range(0, n_input - self.symbol_length, step):
#     #         window = input_items[0][i:i+self.symbol_length]
#     #         corr = np.abs(np.correlate(window, self.preamble_template, mode='valid'))[0]
#     #         corr /= (np.linalg.norm(window) * np.linalg.norm(self.preamble_template) + 1e-10)
#     #         correlations.append(corr)
#     #         positions.append(i)
#     #     found = False
#     #     # print(f"Correlation values: {correlations}")  # Debug: print correlation values
#     #     for idx in range(len(correlations) - 6):
#     #         if all(c > threshold for c in correlations[idx:idx+7]):
#     #             detection_index = positions[idx]
#     #             detection_index = max(0, detection_index - self.symbol_length // 2)
#     #             # print(f"检测到连续7个前导峰，定位点: {detection_index}")

#     #             start_idx = detection_index
#     #             end_idx = min(detection_index + self.preamble_length, n_input)
#     #             if end_idx - start_idx >= self.preamble_length:
#     #                 self.detected_preamble = input_items[0][start_idx:start_idx+self.preamble_length].copy()
#     #                 output_len = min(n_output, len(self.detected_preamble))
#     #                 output_items[0][:output_len] = self.detected_preamble[:output_len]
#     #                 self.message_port_pub(pmt.intern("trigger"), pmt.intern("NEXT"))
#     #                 self.detection_time = current_time
#     #                 # print(f"已提取前导样本点，长度: {len(self.detected_preamble)}")
#     #                 found = True
#     #                 return output_len
#     #             else:
#     #                 self.message_port_pub(pmt.intern("trigger"), pmt.intern("REPLAY"))
#     #                 # print("Need replay1")
#     #                 found = False
#     #                 return n_input
#     #     # if not found:
#     #     #     # print("not find")
#     #     #     self.message_port_pub(pmt.intern("trigger"), pmt.intern("REPLAY"))
#     #     #     return 0
#     #     return n_input

#     # def stop(self):
#     #     return True

import numpy as np
from gnuradio import gr
import pmt
import time
import matplotlib.pyplot as plt

class LoRaPreambleDetector(gr.sync_block):
    """
    LoRa 前导检测器（缓冲 + 分段能量检测）
    """
    def __init__(self, sample_rate=1e6, bandwidth=125e3,
                 sf=7, preamble_symbols=8,
                 energy_window=128, energy_threshold_factor=0.5,
                 step_size=256,save_path='./'):
        gr.sync_block.__init__(
            self,
            name="LoRa Preamble Detector",
            in_sig=[np.complex64],
            out_sig=[np.complex64]
        )

        # LoRa 参数
        self.sample_rate = sample_rate
        self.bandwidth = bandwidth
        self.sf = sf
        self.symbol_len = int(2**sf * sample_rate / bandwidth)
        self.preamble_len = preamble_symbols * self.symbol_len
        self.save_path = save_path
        # 能量检测参数
        self.energy_window = energy_window
        self.energy_threshold_factor = energy_threshold_factor
        
        self.step_size = step_size

        # 输出长度控制
        self.set_output_multiple(4096*5)  # 每次处理长度，保证work快

        # 缓冲区
        self.buffer = np.zeros(0, dtype=np.complex64)

        # 消息端口
        self.message_port_register_out(pmt.intern("trigger"))

        print(f"LoRaPreambleDetector 初始化完成")
        print(f"符号长度: {self.symbol_len}, 前导长度: {self.preamble_len}")

    def work(self, input_items, output_items):
        in0 = input_items[0]
        out = output_items[0]

        # 1. 累积数据到缓存
        self.buffer = np.concatenate([self.buffer, in0])
        consumed = len(in0)   # 输入都要消费掉

        # 2. 如果缓存不够，就先返回
        if len(self.buffer) < self.preamble_len:
            return consumed

        # 3. 能量计算
        segment = self.buffer
        energy = np.convolve(np.abs(segment)**2, np.ones(self.energy_window), 'valid')
        
        noise_floor = np.mean(energy)   # 改成均值
        threshold = max(noise_floor * self.energy_threshold_factor, 1e-3)
        # print(threshold)
        # 4. 上升沿检测
        diff_energy = np.diff(energy)
        rise_edges = np.where((diff_energy > 1e-3) & (energy[1:] > threshold))[0]

        if len(rise_edges) > 0:
            # np.save(f'{time.time()}.npy', energy)
            start_idx = rise_edges[0]
            if start_idx + self.preamble_len+self.energy_window//2 <= (len(segment)) :
                # print('dddddd')
                preamble_segment = segment[start_idx+ self.energy_window//2:start_idx+self.preamble_len+self.energy_window//2].copy()
                # np.save(f'{time.time()}.npy', energy)
                # 输出 preamble
                out_len = min(len(out), len(preamble_segment))
                out[:out_len] = preamble_segment[:out_len]
                # print(len(preamble_segment))
                # 发送触发消息
                self.message_port_pub(pmt.intern("trigger"), pmt.intern("NEXT"))
                np.save(self.save_path+f'{time.time()}.npy', preamble_segment)
                # 丢掉已处理数据
                self.buffer = self.buffer[start_idx + self.preamble_len:]
                return consumed

        # 5. 如果没检测到，就保留 buffer，只消费输入
        if len(self.buffer) > 10 * self.preamble_len:
            # 防止无限堆积
            self.buffer = self.buffer[-10 * self.preamble_len:]
            self.message_port_pub(pmt.intern("trigger"), pmt.intern("REPLAY"))
        return consumed

