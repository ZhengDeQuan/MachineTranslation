#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from itertools import repeat

from onmt.utils.logging import init_logger
from onmt.utils.misc import split_corpus
from onmt.translate.translator import build_translator

import onmt.opts as opts
from onmt.utils.parse import ArgumentParser


def main(opt):
    ArgumentParser.validate_translate_opts(opt)
    logger = init_logger(opt.log_file)

    translator = build_translator(opt, report_score=True)
    src_shards = split_corpus(opt.src, opt.shard_size)
    tgt_shards = split_corpus(opt.tgt, opt.shard_size) \
        if opt.tgt is not None else repeat(None)
    shard_pairs = zip(src_shards, tgt_shards)


    for i, (src_shard, tgt_shard) in enumerate(shard_pairs):
        '''
        src_shard type = list
        len(src_shard) = 2507
        src_shard[0].decode("utf-8")
        'आपकी कार में ब्लैक बॉक्स\n'
        '''
        logger.info("Translating shard %d." % i)
        print("in translate")
        import os
        print("in translate.py pwd = ", os.getcwd())
        translator.translate(
            src=src_shard, #src_shard:type=list,len=2507,src_shard[0]='आपकी कार में ब्लैक बॉक्स\n'
            tgt=tgt_shard,#tgt_shard[0]='a black box in your car\n'
            tgt_path = opt.tgt,
            src_dir=opt.src_dir,
            batch_size=opt.batch_size,
            attn_debug=opt.attn_debug
            )


def _get_parser():
    parser = ArgumentParser(description='translate.py')

    opts.config_opts(parser)
    opts.translate_opts(parser)
    return parser


if __name__ == "__main__":
    parser = _get_parser()

    opt = parser.parse_args()
    main(opt)
