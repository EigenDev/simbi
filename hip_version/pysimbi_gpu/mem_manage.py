import multiprocessing as mp 
import resource 
import struct 

print("Python bit type: {} bit".format(8 * struct.calcsize("P")))
def mem_use():
    print("Memory usage: {:2.2f} MB".format(round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.0, 1)))
    
class Piper():
    def __init__(self):
        self.processes = []
        self.recv_end, self.send_end  = mp.Pipe(False)
    
    #@staticmethod
    def wrapper(self, func, args, kwargs):
        ret = func(*args, **kwargs)
        self.send_end.send(ret)
        
    def run(self, func, *args, **kwargs):
        args2 = [func, args, kwargs]
        proc = mp.Process(target=self.wrapper, args=args2)
        
        self.processes.append(proc)
        proc.start()
        self.ret = self.recv_end.recv()
        proc.join()
        
        
        
    @property
    def result(self):
        return self.ret 
    
class Multiprocessor():

    def __init__(self):
        self.processes = []
        self.queue = mp.SimpleQueue()

    @staticmethod
    def _wrapper(func, queue, args, kwargs):
        ret = func(*args, **kwargs)
        queue.put(ret)

    def run(self, func, *args, **kwargs):
        args2 = [func, self.queue, args, kwargs]
        p = mp.Process(target=self._wrapper, args=args2)
        self.processes.append(p)
        p.start()

    def result(self):
        rets = []
        for p in self.processes:
            ret = self.queue.get()
            rets.append(ret)
        for p in self.processes:
            p.join()
        return rets