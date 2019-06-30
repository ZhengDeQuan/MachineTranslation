#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
'''
the preprocess.py of OpenNMT requires the input_file with the format:
one sentence per line with tokens separated by a space.
For example :
The story of Libya &apos;s liberation , or rebellion , already has its defeated .
Even the comma and the full stop have been separated by a space with the others.

While Our data doesn't share the attribute naturally.
For example:
A black box in your car?
For English data, we need to preprocess the data in the way like this.
For Hindi data, we further need the kill the symbol "|" appended in the last position of the last word in a sentence.
So this script come into being.
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

def Process(file,Outer_dict=None):
    if os.path.exists(file+'.zq'):
        logger.info('File '+file+'.zq'+' already exists')
    with open(file,'r',encoding="utf-8") as fin , open(file+".zq" ,"w",encoding="utf-8") as fout:
        for index, line in enumerate( fin ):
            line = regex.sub(' ',line) # remove punctuations
            words = word_tokenize(line) # tokenizing
            if file.endswith(".hi"): # if the language is Hindi
                words = [word_tokens[:-1] if ord('।') == ord(word_tokens[-1]) else word_tokens for word_tokens in words ]
            elif file.endswith(".en"):
                words = [spell(word_tokens.lower()) for word_tokens in words] # auto correct wrong spelling
                '''
                >>> from autocorrect import spell
                >>> spell('TGhe')
                'The'
                '''
            else:
                raise Exception("Error in Process(%s) in zq_prepreprocess.py"%(file))

            if Outer_dict is not None:
                for word in words:
                    if word not in Outer_dict:
                        Outer_dict[word] = 1
                    else:
                        Outer_dict[word] += 1

            fout.write(' '.join(words)+'\n')
        logger.info("Processed data saved in "+ file+".zq")

def Process2(file,Outer_dict=None):
    if os.path.exists(file+'.zq'):
        logger.info('File '+file+'.zq'+' already exists')
        return
    new_lines = []
    with open(file,'r',encoding="utf-8") as fin:
        for index, line in enumerate( tqdm(fin.readlines()) ):
            line = regex.sub(' ',line) # remove punctuations
            words = word_tokenize(line) # tokenizing
            if file.endswith(".hi"): # if the language is Hindi
                words = [word_tokens[:-1] if ord('।') == ord(word_tokens[-1]) else word_tokens for word_tokens in words ]
            elif file.endswith(".en"):
                words = [spell(word_tokens.lower()) for word_tokens in words] # auto correct wrong spelling
            else:
                raise Exception("Error in Process(%s) in zq_prepreprocess.py"%(file))
            if Outer_dict is not None:
                for word in words:
                    if word not in Outer_dict:
                        Outer_dict[word] = 1
                    else:
                        Outer_dict[word] += 1
            new_lines.append(' '.join(words))
    print("process finish....")
    with open(file + ".zq", "w", encoding="utf-8") as fout:
        for line in new_lines:
            fout.write(line+'\n')
        logger.info("Processed data saved in "+ file+".zq")

def Process3(file,Outer_dict=None):
    with open(file,'r',encoding="utf-8") as fin:
        for index, line in enumerate( tqdm(fin.readlines()) ):

            words = line.split() #因为之前用IndicNLPLib处理了，所以不用再用别的复杂的手段处理了
            if Outer_dict is not None:
                for word in words:
                    if word not in Outer_dict:
                        Outer_dict[word] = 1
                    else:
                        Outer_dict[word] += 1
    print("process finish....")

def main():
    parser = argparse.ArgumentParser(description='zq_preprocess.py')
    # parser.add_argument('-files', nargs='+',default=[
    #                                                  '../HindiEng_Comment/dev_test/test_and_2500Comment.hi.tokByIndicNLPLib',
    #                                                  '../HindiEng_Comment/dev_test/dev_and_500Comment.hi.tokByIndicNLPLib',
    #                                                  '../HindiEng_Comment/parallel/IITB_and_13wComment.en-hi.hi.tokByIndicNLPLib',
    #                                                  '../HindiEng_Comment/monolingual/monolingualByIndicNLPLib.hi'],
    #                     help="files that need to be processed.")
    # parser.add_argument('-save_file', default="../HindiEng_Comment/Processed/IndicNLPLib_Comment.vocab")
    parser.add_argument('-files', nargs='+', default=['../HindiEng_Comment/dev_test/test_and_2500Comment.en.zq',
                                                      '../HindiEng_Comment/dev_test/dev_and_500Comment.en.zq',
                                                      '../HindiEng_Comment/parallel/IITB_and_13wComment.en-hi.en.zq',
                                                      ],
                        help="files that need to be processed.")
    parser.add_argument('-save_file', default="../HindiEng_Comment/Processed/English.vocab")
    parser.add_argument('-make_vocab',action='store_true')
    opt = parser.parse_args()

    print("opt = ",opt)
    Outer_dict = None
    if opt.make_vocab:
        print("Test")
        logger.info("Making vocab option is True")
        Outer_dict = {}
    print("files = ",opt.files)
    for file in opt.files:
        logger.info("Processing %s"%(file))
        Process3(file,Outer_dict)
        print("len(Outer_dict) = ",len(Outer_dict))
    if opt.make_vocab:
        sortedWords= sorted(Outer_dict.items(),key=lambda a:a[1],reverse=True)
        logger.info("Saving vocab for glove")
        with open(opt.save_file,"w",encoding="utf-8") as fout:
            for word, freq in sortedWords:
                fout.write(word+" "+str(freq)+"\n")
        logger.info("Saved in "+opt.save_file)
    logger.info("\nDone.")


if __name__ == "__main__":
    init_logger('zq_prepreprocess_monlingual2.log')
    main()
