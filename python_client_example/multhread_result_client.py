import cv2
import numpy as np
import tritonclient.http as httpclient
import threading
import time
import queue
from collections import defaultdict
import os
# 配置参数
SERVER_URL = "localhost:8000"
MODEL_NAME = "cls"
DATA_DIR = "/home/zhichao.yang/baidu/resnet18"
IMAGE_PATHS = [
    os.path.join(DATA_DIR, "robin_224.bmp")
    # os.path.join(DATA_DIR, "kitten_224.bmp")
] * 8  # 8个任务
CONCURRENCY = min(8, len(IMAGE_PATHS))  # 不超过任务数

def detection_preprocessing(image: cv2.Mat) -> np.ndarray:
    """图像预处理函数"""
    data_batch = (
        np.flip(image / 255.0, axis=2)
        .astype("float32")
        .transpose(2, 0, 1)
    )
    data_batch = data_batch.reshape(1, *data_batch.shape)
    return np.ascontiguousarray(data_batch)

def detection_postprocessing(scores):
    """结果后处理函数"""
    from imagenet_labels import labels
    from topk import topk
    vals, idxs = topk(scores, 5, axis=1)
    idx0 = idxs[0]
    val0 = vals[0]
    result_str = ""
    for i, (val, idx) in enumerate(zip(val0, idx0)):
        result_str += f"Top {i+1}: {val:.4f} {labels[idx]}\n"
    return result_str

class InferenceWorker(threading.Thread):
    """工作线程类"""
    def __init__(self, worker_id, task_queue, result_dict):
        super().__init__()
        self.worker_id = worker_id
        self.task_queue = task_queue
        self.result_dict = result_dict
        self.daemon = True
        self.client = None
        
    def initialize_client(self):
        """初始化Triton客户端"""
        self.client = httpclient.InferenceServerClient(
            url=SERVER_URL,
            connection_timeout=10.0,
            network_timeout=30.0
        )
        
    def run(self):
        """线程主函数"""
        try:
            self.initialize_client()
            while True:
                try:
                    # 获取任务
                    img_path = self.task_queue.get_nowait()
                    task_start = time.time()
                    
                    try:
                        # 1. 读取图片
                        img = cv2.imread(img_path)
                        if img is None:
                            raise ValueError(f"无法读取图片: {img_path}")
                        
                        # 2. 预处理
                        input_data = detection_preprocessing(img)
                        
                        # 3. 准备输入
                        inputs = httpclient.InferInput(
                            "input", input_data.shape, "FP32")
                        inputs.set_data_from_numpy(input_data)
                        
                        # 4. 发送推理请求
                        inference_start = time.time()
                        response = self.client.infer(MODEL_NAME, [inputs])
                        latency = (time.time() - inference_start) * 1000  # ms
                        
                        # 5. 处理结果
                        scores = response.as_numpy("output")
                        result = detection_postprocessing(scores)
                        
                        # 6. 记录结果
                        with self.result_dict['lock']:
                            self.result_dict['completed'] += 1
                            self.result_dict['latencies'].append(latency)
                            self.result_dict['results'][img_path] = {
                                'result': result,
                                'latency': latency
                            }
                        
                        print(f"\nWorker-{self.worker_id} 完成: {img_path} (耗时: {latency:.2f}ms)")
                        
                    except Exception as e:
                        print(f"\nWorker-{self.worker_id} 处理 {img_path} 出错: {str(e)[:100]}")
                        with self.result_dict['lock']:
                            self.result_dict['errors'] += 1
                    finally:
                        self.task_queue.task_done()
                        
                except queue.Empty:
                    break
                    
        except Exception as e:
            print(f"\nWorker-{self.worker_id} 发生严重错误: {str(e)}")
        finally:
            if hasattr(self, 'client') and self.client:
                try:
                    self.client.close()
                except:
                    pass

def run_concurrent_test():
    """运行并发测试"""
    # 结果字典
    results = {
        'completed': 0,
        'errors': 0,
        'latencies': [],
        'results': {},
        'lock': threading.Lock()
    }
    
    # 创建任务队列
    task_queue = queue.Queue()
    for path in IMAGE_PATHS:
        task_queue.put(path)
    
    # 启动工作线程
    threads = []
    start_time = time.time()
    
    print(f"启动 {CONCURRENCY} 个工作线程...")
    for i in range(CONCURRENCY):
        t = InferenceWorker(i+1, task_queue, results)
        t.start()
        threads.append(t)
        time.sleep(0.1)  # 避免瞬时创建过多线程
    
    # 进度监控
    try:
        print("\n测试进度:")
        last_count = 0
        while any(t.is_alive() for t in threads):
            time.sleep(0.5)
            current = results['completed']
            elapsed = max(0.1, time.time() - start_time)
            
            # 计算瞬时速度
            instant_speed = (current - last_count) * 2  # 0.5秒间隔
            avg_speed = current / elapsed
            
            print(f"\r已完成: {current}/{len(IMAGE_PATHS)} | "
                  f"速度: {instant_speed:.1f} req/s | "
                  f"平均: {avg_speed:.1f} req/s | "
                  f"错误: {results['errors']}", end="")
            last_count = current
            
    except KeyboardInterrupt:
        print("\n用户中断测试...")
    finally:
        # 等待所有线程完成
        for t in threads:
            t.join(timeout=2.0)
        
        total_time = time.time() - start_time
        
        # 打印汇总结果
        print("\n\n" + "="*50)
        print(f"测试汇总 (并发数={CONCURRENCY})")
        print(f"总请求数: {len(IMAGE_PATHS)}")
        print(f"成功完成: {results['completed']}")
        print(f"失败请求: {results['errors']}")
        print(f"总耗时: {total_time:.2f}秒")
        print(f"平均吞吐量: {results['completed']/total_time:.2f} req/s")
        
        # 延迟统计
        if results['latencies']:
            latencies = results['latencies']
            print("\n延迟统计(ms):")
            print(f"平均值: {np.mean(latencies):.2f} ± {np.std(latencies):.2f}")
            print(f"最小值: {np.min(latencies):.2f}")
            print(f"最大值: {np.max(latencies):.2f}")
            print(f"中位数: {np.median(latencies):.2f}")
            print(f"P90: {np.percentile(latencies, 90):.2f}")
            print(f"P99: {np.percentile(latencies, 99):.2f}")
        
        # 打印详细结果
        if results['results']:
            print("\n" + "="*50 + "\n详细推理结果:")
            for img_path, data in results['results'].items():
                print(f"\n{'-'*30}")
                print(f"图片: {img_path}")
                print(f"延迟: {data['latency']:.2f}ms")
                print("\n分类结果:")
                print(data['result'].strip())

if __name__ == "__main__":
    run_concurrent_test()