# coding=utf-8
# Copyright (C) ATHENA AUTHORS
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# pylint: disable=missing-function-docstring, invalid-name
""" data queue for multi thread """
import time
import threading
import queue


class DataQueue:
    """Queue for data prefetching"""

    def __init__(self, generator, capacity=20, num_threads=4, max_index=10000, wait_time=0.0001):
        """

        Args:
            generator(generator): instance of generator which feed data
            capacity(int): maximum data to prefetch
            num_threads(int): control concurrency, only take effect when do preprocessing
            wait_time(float): time to sleep when queue is full
        """        
        self.generator = generator
        self.capacity = capacity
        self.wait_time = wait_time
        self.queue = queue.Queue()
        self.index = 0
        self.max_index = max_index

        self._stop = threading.Event()
        self._lock = threading.Lock()

        self.threads = [
            threading.Thread(target=self.generator_task) for _ in range(num_threads)
        ]

        for t in self.threads:
            t.setDaemon(True)
            t.start()

    def __del__(self):
        self.stop()

    def get(self):
        return self.queue.get()

    def stop(self):
        self._stop.set()

    def generator_task(self):
        """Enqueue batch data
        """
        while not self._stop.is_set():
            try:
                if self.index >= self.max_index:
                    continue
                batch = self.generator(self.index)
                self._lock.acquire()
                if self.queue.qsize() < self.capacity:
                    try:
                        self.index = self.index + 1
                    except ValueError as e:
                        print(e)
                        self._lock.release()
                        continue
                    self.queue.put(batch)
                    self._lock.release()
                else:
                    self._lock.release()
                    time.sleep(self.wait_time)
            except Exception as e:
                print(e)
                self._stop.set()
                raise


def test():
    
    def generator(i):
        return i

    train_queue = DataQueue(generator, capacity=8, num_threads=4)
    for _ in range(92):
        print(train_queue.get())
    train_queue.stop()


if __name__ == "__main__":
    test()
