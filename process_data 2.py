import argparse
from data_utils import get_dataset, get_language_names
from configure_data import make_tokenizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process Data')
    parser.add_argument('--tokenizer-model-type', type=str,
                        default=None,
                        help="Model type to use for sentencepiece tokenization \
                               (one of ['bpe', 'char', 'unigram', 'word']) or \
                               bert vocab to use for BertWordPieceTokenizer (one of \
                               ['bert-large-uncased', 'bert-large-cased', etc.])")
    parser.add_argument('--tokenizer-type', type=str,
                        default='BertWordPieceTokenizer',
                        choices=['CharacterLevelTokenizer',
                                 'SentencePieceTokenizer',
                                 'BertWordPieceTokenizer',
                                 'GPT2BPETokenizer',
                                 'ChineseSPTokenizer'],
                        help='what type of tokenizer to use')
    parser.add_argument('--train-data', nargs='+', default=None,
                        help='Whitespace separated filenames or corpora names '
                             'for training.')
    parser.add_argument('--no-pre-tokenize', action='store_true')
    args = parser.parse_args()
    args.tokenizer_type = "glm_" + args.tokenizer_type
    args.task_mask = True
    args.block_mask_prob = False
    args.make_vocab_size_divisible_by = 128

    paths = [args.train_data] if isinstance(args.train_data, str) else args.train_data
    new_paths = []
    for p in paths:
        if p == 'multilingual':
            new_paths += [f'multilingual-{lang}' for lang in get_language_names()]
        else:
            new_paths.append(p)
    tokenizer = make_tokenizer(args)
    _datasets = [get_dataset(p, tokenizer=tokenizer, pre_tokenize=not args.no_pre_tokenize, no_lazy_loader=False,
                             loader_scatter=None, data_parallel_rank=0, global_rank=0) for p in new_paths]
