'''
When running the zq_prepreprocess.py.
It is too slow for the data in ../HindiEng/parallel 110M.
So I decided to use the multiprocess style code to deal with it.
'''

from zq_tokenize import word_tokenize
from autocorrect import spell
import string
import argparse
from onmt.utils.logging import init_logger, logger
import re
import os
from tqdm import tqdm
regex = re.compile('[%s]' % re.escape(string.punctuation))
import os
import datetime
from multiprocessing import Process, Queue, Array, RLock,Manager

class FileManager(object):
    def __init__(self,filename,out_dir="",worker_num=4,BLOCK_SIZE=100000000,makeDict=False):
        '''
        :param filename:
        :param worker_num:
        :param BLOCK_SIZE: byte as its unit
        '''
        self.filename = filename
        if os.path.exists(out_dir):
            self.out_dir =  out_dir + datetime.datetime.now().strftime("%Y/%d/%m %H:%M:%S")
        else:
            self.out_dir = out_dir
        os.makedirs(self.out_dir)
        self.file_size = self._getFileSize(filename)
        self.worker_num = worker_num
        self.BLOCK_SIZE = BLOCK_SIZE
        self.makeDict = makeDict


    def _getFileSize(self,filename):
        fstream = open(filename,"r",encoding="utf-8")
        fstream.seek(0,os.SEEK_END)
        file_size=fstream.tell()
        fstream.close()
        return file_size

    def process_oneline(self,line):
        line = regex.sub(' ', line)  # remove punctuations
        words = word_tokenize(line)  # tokenizing
        if self.filename.endswith(".hi"):  # if the language is Hindi
            words = [word_tokens[:-1] if ord('।') == ord(word_tokens[-1]) else word_tokens for word_tokens in words]
        elif self.filename.endswith(".en"):
            words = [spell(word_tokens.lower()) for word_tokens in words]  # auto correct wrong spelling
        else:
            raise Exception("Error in Process(%s) in zq_prepreprocess.py" % (self.filename))
        return ' '.join(words)

    def process_found(self,pid,array,result,rlock):
        '''
        :param pid:
        :param array: 进程间的共享队列，用于标记各个进程所读的文件块的结束位置
        :param rlock:
        各个进程先从array中获取当前最大的值为起始位置startposition
        结束的位置endposition = startposition+BLOCKSIZE if startposition + BLOCKSIZE < FILESIZE else FILESIZE
        if startposition==FILESIZE : 结束进程
        elif startposition ==0 : 从0开始读取
        else: 为了防止行被block截断，先读一行但不处理，从下一行开始正式处理
        if 当前位置 <= endposition 就readline()
        else:越过边界，就从array中重新查找最大值
        :return:
        '''
        f=open(self.filename,"r",encoding="utf-8")
        while True:
            rlock.acquire()
            print("pid:%s "%(pid),','.join([str(v) for v in array]))
            startposition = max(array)
            endposition=array[pid] = (startposition+self.BLOCK_SIZE) if (startposition + self.BLOCK_SIZE) < self.file_size else self.file_size
            rlock.release()
            if startposition == self.file_size:
                print('pid:%s end'%(pid))
                break
            elif startposition !=0:
                f.seek(startposition)
                f.readline() #读入一行，但是不处理
            pos = ss = f.tell()
            fout = open(os.path.join(self.out_dir,str(pid)+"_jobs_"+str(endposition)),"w",encoding="utf-8")
            while pos < endposition:
                line = f.readline()
                line = self.process_oneline(line)
                # fout.write(line)
                fout.write(line+"\n")
                if result is not None:
                    rlock.acquire()
                    line = line.strip().split()
                    for token in line:
                        if result.get(token):
                            result[token]+=1
                        else:
                            result[token]=1
                    rlock.release()
                pos = f.tell()
            print("pid:%s , startposition:%s, endposition:%s"%(pid,ss,pos))
            fout.flush()
            fout.close()
            ee = f.tell()
        f.close()
    def Integrating(self):
        filenames = os.listdir(self.out_dir)
        os.chdir(self.out_dir)
        myDict= {}
        for filename in filenames:
            endposiion = int(filename.split("jobs_")[1])
            myDict[endposiion]=filename

        myDictList = sorted(myDict.items())
        f = open(myDictList[0][1],"a",encoding="utf-8")
        f.seek(0,os.SEEK_END)
        for endposiion,filename in myDictList[1:]:
            ftemp = open(filename,"r",encoding="utf-8")
            content = ftemp.read()
            ftemp.close()
            f.write(content)
        f.close()


    def main(self):
        start_time = datetime.datetime.now().strftime("%Y/%d/%m %H:%M:%S")
        print("start_time = ",start_time)
        print("File size = ",self.file_size)
        rlock = RLock()
        if self.makeDict:
            manager = Manager()
            result = manager.dict()
        else:
            result = None
        array = Array('l',self.worker_num,lock=rlock) #shared variable among all the processes
        processes = []
        for i in range(self.worker_num):
            p = Process(target=self.process_found,args=[i,array,result,rlock])
            processes.append(p)

        for i in range(self.worker_num):
            processes[i].start()

        for i in range(self.worker_num):
            processes[i].join()
        print("end_time = ",datetime.datetime.now().strftime("%Y/%d/%m %H:%M:%S"))

        print("Start Integrating")
        self.Integrating()

def main():
    parser = argparse.ArgumentParser(description='zh_preprocess.py')
    parser.add_argument('-files', nargs='+',default=['../HindiEng/dev_test/dev.en','../HindiEng/dev_test/dev.hi',
                                                     '../HindiEng/dev_test/test.en','../HindiEng/dev_test/test.hi',
                                                     '../HindiEng/parallel/IITB.en-hi.en','../HindiEng/parallel/IITB.en-hi.hi'],
                        help="files that need to be processed.")
    parser.add_argument('-make_vocab',action='store_true')
    opt = parser.parse_args()
    Outer_dict = None
    if opt.make_vocab:
        print("Test")
        logger.info("Making vocab option is True")
        Outer_dict = {}
    print("files = ",opt.files)
    for file in opt.files:
        logger.info("Processing %s"%(file))
        out_dir = os.path.dirname(os.path.abspath(file))
        ob = FileManager(filename=file,out_dir=os.path.join(out_dir,"TTT"),worker_num=4,BLOCK_SIZE=50000000)
        ob.main()
    # if opt.make_vocab:
    #     sortedWords= sorted(Outer_dict.items(),key=lambda a:a[1],reverse=True)
    #     logger.info("Saving vocab for glove")
    #     with open(opt.files[0]+".vocab","w",encoding="utf-8") as fout:
    #         for word, freq in sortedWords:
    #             fout.write(word+"\n")
    #     logger.info("Saved in "+opt.files[0]+".vocab")
    logger.info("\nDone.")

if __name__ == "__main__":
    init_logger('zq_MultiprocessReadWriteData.log')
    main()
    # ob = FileManager(filename="a.txt",out_dir="TTT",worker_num=3,BLOCK_SIZE=10000000)
    # ob.main()
    # del ob
    # ob = FileManager()

