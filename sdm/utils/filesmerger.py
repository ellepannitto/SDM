import heapq
import contextlib
import gzip
import glob

from enum import Enum


class Mode(Enum):
    txt = 1
    gzip = 2


def merge(filename_pattern, output_filename, mode=Mode.txt):

    openfunc = lambda fname: open(fname)
    if mode == Mode.gzip:
        openfunc = lambda fname: gzip.open(fname, "rt")

    files = glob.iglob(filename_pattern)

    with contextlib.ExitStack() as stack:
        files = [stack.enter_context(openfunc(fn)) for fn in files]
        with gzip.open(output_filename, 'wt') as f:
            f.writelines(heapq.merge(*files))


def collapse(filename, output_filename, delimiter="\t"):

    with gzip.open(filename, "rt") as fin, open(output_filename, "w") as fout:

        firstline, firstfreq = fin.readline().strip().split(delimiter)
        firstfreq = float(firstfreq)

        for line in fin:
            el, freq = line.strip().split(delimiter)
            freq = float(freq)
            if el == firstline:
                firstfreq += freq
            else:
                print("{}\t{}".format(firstline, firstfreq), file=fout)
                firstline = el
                firstfreq = freq

        print("{}\t{}".format(firstline, firstfreq), file=fout)
