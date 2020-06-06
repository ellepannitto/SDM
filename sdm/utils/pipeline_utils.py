import multiprocessing as mp
from collections import namedtuple
import logging
import tqdm
import os

logger = logging.getLogger(__name__)

Task = namedtuple("Task", ["done", "data"])


class SmartQueue:

    def __init__(self, nworker_in, nworker_out, ide, maxsize=0, debug=False):
        self._q = mp.Queue(maxsize)
        self.ide = ide

        self._nworker_in = nworker_in
        self._nworker_out = nworker_out
        self.received_eos = mp.Value('i', 0)
        self._debug = debug

    def put(self, obj: Task):
        if obj.done:
            with self.received_eos.get_lock():
                self.received_eos.value += 1

                if self._debug:
                    logger.debug ("Queue({}) received {} EOS so far".format(self.ide, self.received_eos.value))

                if self.received_eos.value == self._nworker_in:

                    for _ in range(self._nworker_out):
                        self._q.put(Task(True, None))

                    if self._debug:
                        logger.debug("Queue({}) Finished".format(self.ide))
        else:
            # if self._debug:
            #     logger.debug ("Queue({}) put {} ".format(self.ide, obj)[:50])
            self._q.put(obj)

    def get(self):

        ret = self._q.get()
        # if self._debug:
        #     logger.debug("Queue({}) get {}".format(self.ide, ret)[:50])
        return ret


class Pipeline:
    """
    This class is used for multiprocessing.
    It takes as parameter n functions (and the number of workers to assign to each function)
      and creates n+1 queues, so that each function works in parallel by taking the input data
      from one queue and putting the output in the next.
    """

    def __init__(self, list_of_functions, list_of_workers, batch_size):
        self.functions = list_of_functions
        self.workers = list_of_workers
        self.batch_size = batch_size

    def parallel_process(self, func, in_q, out_q, stage):

        logger.debug("Process(stage: {}, pid: {}) started".format(stage, os.getpid()))

        x = in_q.get()

        while not x.done:
            for y in func(x.data):
                out_q.put(Task(False, y))

            x = in_q.get()

        logger.debug("Process(stage: {}, pid: {}) finished, propagating EOS...".format(stage, os.getpid()))
        out_q.put(Task(True, None))

        logger.debug("Process(stage: {}, pid: {}) exit.".format(stage, os.getpid()))

    def run(self, iterable_input):

        queues = []
        nworker_in = 1
        ide = 0
        for _, nworker_out in zip(self.functions, self.workers):
            maxsize = 4 * nworker_out
            if ide==0:
                maxsize = 0
            queues.append(SmartQueue(nworker_in, nworker_out, ide=ide, maxsize=maxsize, debug=True))
            ide += 1
            nworker_in = nworker_out
        # queues.append(SmartQueue(nworker_in, 1, ide=ide))
        queues.append(SmartQueue(nworker_in, 1, ide=ide, maxsize=10*self.batch_size, debug=True))

        pool_list = []
        i = 0
        for func, n_workers in zip(self.functions, self.workers):
            pool_list.append(mp.Pool(n_workers,
                                     initializer=self.parallel_process,
                                     initargs=(func, queues[i], queues[i+1], i)))
            i += 1

        for x in tqdm.tqdm(iterable_input, desc="pipeline input"):
            queues[0].put(Task(False, x))
        queues[0].put(Task(True, None))

        y = queues[-1].get()
        while not y.done:
            yield y.data
            y = queues[-1].get()

        logger.debug("Pipeline.run() done, joining threads")

        for i, pool in enumerate(pool_list):
            logger.debug("terminating pool for stage {}".format(i))
            pool.terminate()
            logger.debug("joining pool for stage {}".format(i))
            pool.join()

        logger.debug("Pipeline.run() exiting")


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