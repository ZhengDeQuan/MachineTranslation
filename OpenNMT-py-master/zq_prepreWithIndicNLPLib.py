'''
like zq_prepreprocess.py
This file doing the prepreprocess work.
But when it comes to the tokenization procedure, this script would use the Indic_NLP_Library
to tokenize the sentence into words.
this script reads in the HindiEng/
outputs the WordSegementHindiEng/
'''
INDIC_NLP_LIB_HOME = r"/home/nemo/zhengquan/indic_nlp_library"
INDIC_NLP_RESOURCES = r"/home/nemo/zhengquan/indic_nlp_resources"
import sys
sys.path.append(r'{}/src'.format(INDIC_NLP_LIB_HOME))
from indicnlp import common
common.set_resources_path(INDIC_NLP_RESOURCES)
from indicnlp import loader
loader.load()
from indicnlp.tokenize import indic_tokenize
from indicnlp.morph import unsupervised_morph
from indicnlp import common
import os
import argparse
from tqdm import tqdm



def make_out_file_name(in_dir,in_file_path,out_dir_name):
    in_dir_name = in_dir.split('/')[-1]
    abs_in_dir = os.path.abspath(in_dir)
    abs_out_dir= abs_in_dir.split('/')
    abs_out_dir[-1] = out_dir_name

    abs_in_file_path = os.path.abspath(os.path.join(abs_in_dir,in_file_path)).split('/')
    abs_out_file_path = [file if file != in_dir_name else out_dir_name for file in abs_in_file_path]
    abs_out_file_path = '/'.join(abs_out_file_path)
    return abs_out_file_path,'/'.join(abs_in_file_path)

def Process(infile,outfile,column=0,max_row=1000000,interval = 100000):
    print("infile = ",infile)
    with open(infile,"r",encoding="utf-8") as fin:
        analyzer = unsupervised_morph.UnsupervisedMorphAnalyzer('hi')
        result = []
        read_line_num = 0
        lines = fin.readlines()
        tot_line = len(lines)
        for line in lines:
            read_line_num += 1
            if read_line_num % interval == 0:
                print("processed %d lines " % read_line_num)
            line = line.split('\t')
            indic_string = line[column]
            indic_string=indic_string.strip()
            indic_res1 = indic_tokenize.trivial_tokenize(indic_string)
            analyzes_tokens = analyzer.morph_analyze_document(indic_res1)
            result.append(' '.join(analyzes_tokens))
            # if read_line_num % max_row==0:
            #     if os.path.exists(outfile):
            #         fout = open(outfile,"a",encoding="utf-8")
            #         fout.seek(0,2)
            #     else:
            #         fout = open(outfile,"w",encoding="utf-8")
            #     for line in result:
            #         # fout.write(line)
            #         fout.write(line+"\n")
            #     fout.close()
            #     result = []
        print("len_result = ",len(result))
        fout = open(outfile, "w", encoding="utf-8")
        for line in result:
            fout.write(line+"\n")
        # if result:
        #     if os.path.exists(outfile):
        #         fout = open(outfile, "a", encoding="utf-8")
        #         fout.seek(0, 2)
        #     else:
        #         fout = open(outfile, "w", encoding="utf-8")
        #     for line in result:
        #         fout.write(line)
        #     fout.close()
        # print("tot_line = ",tot_line)

def main(args):
    in_files = args.data_files
    out_dir_name = args.out_dir_name
    data_dir = args.data_dir
    for in_file in in_files:
        abs_out_file, abs_in_file = make_out_file_name(data_dir,in_file,out_dir_name)
        os.makedirs('/'.join(abs_out_file.split('/')[:-1]),exist_ok=True)
        Process(abs_in_file,abs_out_file)






if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    # parser.add_argument("-data_files","--data_files",
    #                     help="data files need to be processed, with the relative dir",
    #                     default=['dev_test/test.hi','dev_test/dev.hi','monolingual/monolingual.hi','parallel/IITB.en-hi.hi'],nargs='+',type=str)
    parser.add_argument("-data_files", "--data_files",
                        help="data files need to be processed, with the relative dir",
                        default=[ 'parallel/IITB.en-hi.hi'], nargs='+', type=str)
    parser.add_argument("-data_dir","--data_dir",default="../HindiEng")
    parser.add_argument('-out_dir_name','--out_dir_name',default='WordSegmentHindiEng2',
                        help="the processed file will be stored in the out_dir and with the same file name. If it doesn't actually exists, we will make it.")
    args = parser.parse_args()

    assert len(args.data_files) != 0, print("len(args.data_files)==0, no input to process")
    main(args)


