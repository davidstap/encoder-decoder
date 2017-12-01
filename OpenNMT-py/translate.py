#!/usr/bin/env python

from __future__ import division
from builtins import bytes
import os
import argparse
import math
import codecs
import torch
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu

import onmt
import onmt.IO
import opts
from itertools import takewhile, count
try:
    from itertools import zip_longest
except ImportError:
    from itertools import izip_longest as zip_longest

parser = argparse.ArgumentParser(
    description='translate.py',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
opts.add_md_help_argument(parser)
opts.translate_opts(parser)

opt = parser.parse_args()
if opt.batch_size != 1:
    print("WARNING: -batch_size isn't supported currently, "
          "we set it to 1 for now!")
    opt.batch_size = 1


def report_score(name, score_total, words_total):
    print("%s AVG SCORE: %.4f, %s PPL: %.4f" % (
        name, score_total / words_total,
        name, math.exp(-score_total/words_total)))


def get_src_words(src_indices, index2str):
    words = []
    raw_words = (index2str[i] for i in src_indices)
    words = takewhile(lambda w: w != onmt.IO.PAD_WORD, raw_words)
    return " ".join(words)



def main():
    previous_words = None
    dummy_parser = argparse.ArgumentParser(description='train.py')
    opts.model_opts(dummy_parser)
    dummy_opt = dummy_parser.parse_known_args([])[0]
    print('dummy_opt: ',dummy_opt)

    opt.cuda = opt.gpu > -1
    if opt.cuda:
        torch.cuda.set_device(opt.gpu)
    translator = onmt.Translator(opt, dummy_opt.__dict__)
    out_file = codecs.open(opt.output, 'w', 'utf-8')
    pred_score_total, pred_words_total = 0, 0
    gold_score_total, gold_words_total = 0, 0
    if opt.dump_beam != "":
        import json
        translator.initBeamAccum()
    data = onmt.IO.ONMTDataset(
        opt.src, opt.tgt, translator.fields,
        use_filter_pred=False)

    test_data = onmt.IO.OrderedIterator(
        dataset=data, device=opt.gpu,
        batch_size=opt.batch_size, train=False, sort=False,
        shuffle=False)

    counter = count(1)
    for batch in test_data:
        pred_batch, gold_batch, pred_scores, gold_scores, attn, src \
            = translator.translate(batch, data)
        pred_score_total += sum(score[0] for score in pred_scores)
        pred_words_total += sum(len(x[0]) for x in pred_batch)
        if opt.tgt:
            gold_score_total += sum(gold_scores)
            gold_words_total += sum(len(x) for x in batch.tgt[1:])

        #davidstap
        #_, src_lengths = batch.src
        #encStates, context = translator.model.encoder(src, src_lengths)

        # z_batch: an iterator over the predictions, their scores,
        # the gold sentence, its score, and the source sentence for each
        # sentence in the batch. It has to be zip_longest instead of
        # plain-old zip because the gold_batch has length 0 if the target
        # is not included.
        z_batch = zip_longest(
                pred_batch, gold_batch,
                pred_scores, gold_scores,
                (sent.squeeze(1) for sent in src.split(1, dim=1)))



        for pred_sents, gold_sent, pred_score, gold_score, src_sent in z_batch:
            # src_sent is torch.LongTensor
            #print('type src_sent:',type(src_sent))
            n_best_preds = [" ".join(pred) for pred in pred_sents[:opt.n_best]]
            out_file.write('\n'.join(n_best_preds))
            out_file.write('\n')
            out_file.flush()

            if opt.verbose:
                sent_number = next(counter)
                words = get_src_words(
                    src_sent, translator.fields["src"].vocab.itos)




                if previous_words is not None:

                    print('BLEU: ',sentence_bleu([words], previous_words))
                    print()
                    print('S1:',words)
                    print('S2:',previous_words)

                #os.write(1, bytes('\nSENT %d: %s\n' %
            #                      (sent_number, words), 'UTF-8'))

                previous_words = words



                best_pred = n_best_preds[0]

                #TODO: calculate BLEU score reference (best_pred) and hypothesis (words)
                #TODO: calculate cosine_similarity (best_pred) and hypothesis (words)
                #bleu_score = sentence_bleu(best_pred, words)
                #print('BLEU: ',bleu_score)




                best_score = pred_score[0]
                #os.write(1, bytes('PRED %d: %s\n' %
            #                      (sent_number, best_pred), 'UTF-8'))
                #print("PRED SCORE: %.4f" % best_score)

                # 'words' = input sentence
                # 'best_pred' = prediction

                # put source sentence in translator.model.encoder to find context
                # maybe change data type src? torchtext datatype?

                #model = NMTModel(encoder, decoder) (see ModelConstructor)
                src_lengths = len(words.split())

                # src(FloatTensor): a sequence of source tensors with
                #         optional feature tensors of size (len x batch).
                # tgt(FloatTensor): a sequence of target tensors with
                #         optional feature tensors of size (len x batch).
                # lengths([int]): an array of the src length.
                # dec_state: A decoder state object




                #hidden, context = translator.model.encoder(src_sent, src_lengths)



                #euc_dist(context_r, context_pred)


                if opt.tgt:
                    tgt_sent = ' '.join(gold_sent)
                    os.write(1, bytes('GOLD %d: %s\n' %
                             (sent_number, tgt_sent), 'UTF-8'))
                    print("GOLD SCORE: %.4f" % gold_score)

                if len(n_best_preds) > 1:
                    print('\nBEST HYP:')
                    for score, sent in zip(pred_score, n_best_preds):
                        os.write(1, bytes("[%.4f] %s\n" % (score, sent),
                                 'UTF-8'))

    report_score('PRED', pred_score_total, pred_words_total)
    if opt.tgt:
        report_score('GOLD', gold_score_total, gold_words_total)

    if opt.dump_beam:
        json.dump(translator.beam_accum,
                  codecs.open(opt.dump_beam, 'w', 'utf-8'))


if __name__ == "__main__":
    main()
