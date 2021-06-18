import hashlib
import os
import glob
import base64
from multiprocessing import Pool, Process, Manager

OUTFILE="md5s.txt"
INPUT_FOLDER = r"J:\bypy"

def writer(q):
    while True:
        message = q.get()
        print(message)
        with open(OUTFILE,"a") as f:
            f.write(message)
            f.write("\n")


def cal_md5(path, q):
    dirname, filename = os.path.split(path) 
    if os.path.isfile(path):
        with open(path,'rb') as fp:
            data = fp.read()
            base64_str = (base64.b64encode(hashlib.md5(data).digest())).decode()
            q.put("{} : {}".format(filename, base64_str))
    else:
        q.put("{} : 文件不存在".format(filename))


def init(queue):
    global q
    q = queue

if __name__ == '__main__':
    m = Manager()
    q = m.Queue()
    filePaths = glob.glob(os.path.join(INPUT_FOLDER, "*"))
    wp = Process(target=writer, args=(q, ))
    wp.start()
    for path in filePaths:
        dirname, filename = os.path.split(path)
        print(filename)
    with Pool(24) as p:
        p.starmap(cal_md5, [(path, q) for path in filePaths])
    wp.terminate()