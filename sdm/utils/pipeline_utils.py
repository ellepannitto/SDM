import multiprocessing as mp
from collections import namedtuple

Task = namedtuple("Task", ["done", "data"])


class SmartQueue:

    def __init__(self, nworker_in, nworker_out, ide, maxsize=0):
        self._q = mp.Queue(maxsize)
        self.ide = ide

        self._nworker_in = nworker_in
        self._nworker_out = nworker_out
        self.received_eos = mp.Value('i', 0)

    def put(self, obj: Task):
        if obj.done:
            with self.received_eos.get_lock():
                self.received_eos.value += 1
                # print ("Queue({},{}) received {} EOS so far".format(self._nworker_in, self._nworker_out, self.received_eos.value))
                if self.received_eos.value == self._nworker_in:
                    # print("Queue({},{}) Finished".format(self._nworker_in, self._nworker_out))
                    for _ in range(self._nworker_out):
                        # print("[QUEUE - ", self.ide, "]: put EOS")
                        self._q.put(Task(True, None))
        else:
            # print("[QUEUE - ", self.ide, "]: put", obj)
            self._q.put(obj)

    def get(self):
        ret = self._q.get()
        # print("[QUEUE - ", self.ide, "]: get", ret)
        return ret


class Pipeline:

    def __init__(self, list_of_functions, list_of_workers, batch_size):
        self.functions = list_of_functions
        self.workers = list_of_workers
        self.batch_size = batch_size

    def parallel_process(self, func, in_q, out_q):
        x = in_q.get()

        while not x.done:
            for y in func(x.data):
                out_q.put(Task(False, y))

            x = in_q.get()
        out_q.put(Task(True, None))


    def run(self, iterable_input):

        queues = []
        nworker_in = 1
        ide = 0
        for _, nworker_out in zip(self.functions, self.workers):
            queues.append(SmartQueue(nworker_in, nworker_out, ide=ide, maxsize=2*nworker_out))
            ide += 1
            nworker_in = nworker_out
        queues.append(SmartQueue(nworker_in, 1, ide=ide, maxsize=2*self.batch_size))

        pool_list = []
        i = 0
        for func, n_workers in zip(self.functions, self.workers):
            pool_list.append(mp.Pool(n_workers,
                                     initializer=self.parallel_process,
                                     initargs=(func, queues[i], queues[i+1])))
            i += 1

        for x in iterable_input:
            queues[0].put(Task(False, x))
        queues[0].put(Task(True, None))

        y = queues[-1].get()
        while not y.done:
            yield y.data
            y = queues[-1].get()

        for i, pool in enumerate(pool_list):
            pool.close()
            pool.join()


if __name__ == "__main__":
    import random
    import os
    import time

    def double(x):
        print(os.getpid(), time.time(), "double", x)
        for i in range(random.randint(50000000, 100000000)):
            pass
        print(os.getpid(), time.time(), "finish double", x)
        yield 2 * x


    def increment(x):
        print(os.getpid(), time.time(), "increment", x)
        for i in range(random.randint(70000000, 130000000)):
            pass
        print(os.getpid(), time.time(), "finished increment", x)
        yield x + 1

    pipeline = Pipeline([double, increment, double], [4,4,4])
    for y in pipeline.run(range(12)):
        print ("Result:", y)