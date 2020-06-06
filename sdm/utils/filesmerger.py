import heapq
import contextlib
import gzip
import glob
import os
import shutil
import tempfile

from sdm.utils import data_utils as dutils

from enum import Enum


class Mode(Enum):
    txt = 1
    gzip = 2


def merge(filename_pattern, output_filename, mode=Mode.txt, batch=20):

    openfunc = lambda fname: open(fname)
    openfunc_write = lambda fname: open(fname, "wt")
    if mode == Mode.gzip:
        openfunc = lambda fname: gzip.open(fname, "rt")
        openfunc_write = lambda fname: gzip.open(fname, "wt")

    files = glob.iglob(filename_pattern)

    first = True
    total_files_produced = 0
    tempfiles = []
    # next_file_id = 0

    while first or len(files) > 1:

        first = False
        next_iterable = []

        with contextlib.ExitStack() as stack:
            for files_group in dutils.grouper(files, batch):
                files_group = [stack.enter_context(openfunc(fn)) for fn in files_group if fn is not None]
                outfile, outfile_name = tempfile.mkstemp(text=True)
                print("PRINTING ON", outfile_name)
                next_iterable.append (outfile_name)
                tempfiles.append (outfile_name)
                total_files_produced += 1
                with open(outfile_name, "w") as f:
                    f.writelines(heapq.merge(*files_group))
                # next_file_id += 1
                for fhandler in files_group:
                    fhandler.close()

        files = next_iterable

    if not tempfiles:
        raise ValueError ("no files to merge")

    for tmpfile in tempfiles[:-1]:
        os.remove(tmpfile)
        print("REMOVING", tmpfile)
        # os.remove(_temp_out_filename+str(i))
        #print ("to be removed: {}".format(_temp_out_filename+str(i)))

    print("MOVING", tempfiles[-1])
    shutil.move (tempfiles[-1], output_filename)
    #print ("file to keep: ", total_files_produced-1)


def collapse(filename, output_filename, delimiter="\t", threshold=300, mode=Mode.txt):

    openfunc = lambda fname: open(fname)
    openfunc_write = lambda fname: open(fname, "wt")
    if mode == Mode.gzip:
        openfunc = lambda fname: gzip.open(fname, "rt")
        openfunc_write = lambda fname: gzip.open(fname, "wt")

    with openfunc(filename) as fin, openfunc_write(output_filename) as fout:

        firstline, firstfreq = fin.readline().strip().split(delimiter)
        firstfreq = float(firstfreq)

        for line in fin:
            el, freq = line.strip().split(delimiter)
            freq = float(freq)
            if el == firstline:
                firstfreq += freq
            else:
                if firstfreq > threshold:
                    print("{}\t{}".format(firstline, firstfreq), file=fout)
                firstline = el
                firstfreq = freq

        if firstfreq > threshold:
            print("{}\t{}".format(firstline, firstfreq), file=fout)
