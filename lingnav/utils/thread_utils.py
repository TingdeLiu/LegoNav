import threading
"""
线程工具模块
提供线程安全的读写锁实现，用于协调多线程对共享资源的访问。
"""

class ReadWriteLock:
    """
    读写锁实现
    允许多个读者同时读取共享资源，但写者必须独占访问。
    实现了典型的读写互斥模式：
    - 多个读者可以同时获取读锁
    - 写者获取写锁时，必须等待所有读者释放
    - 读者获取读锁时，必须等待写者释放
    """
    def __init__(self):
        """初始化读写锁，使用条件变量和锁来实现同步"""
        self._read_ready = threading.Condition(threading.Lock())
        self._readers = 0  # 记录读者数量（负数表示写者持有锁）

    def acquire_read(self):
        """
        获取读锁
        等待直到没有写者持有锁，然后增加读者计数
        """
        with self._read_ready:
            self._read_ready.wait_for(lambda: self._readers >= 0)
            self._readers += 1

    def release_read(self):
        """
        释放读锁
        减少读者计数，当最后一个读者释放时，通知所有等待的写者
        """
        with self._read_ready:
            self._readers -= 1
            if self._readers == 0:
                self._read_ready.notify_all()

    def acquire_write(self):
        """
        获取写锁
        等待直到没有读者和写者持有锁，然后将_readers设置为-1表示有写者持有锁
        """
        with self._read_ready:
            self._read_ready.wait_for(lambda: self._readers == 0)
            self._readers = -1

    def release_write(self):
        """
        释放写锁
        将_readers重置为0，并通知所有等待的读者和写者
        """
        with self._read_ready:
            self._readers = 0
            self._read_ready.notify_all()
