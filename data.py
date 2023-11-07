# Created by Baole Fang at 10/30/23

import argparse
import os

import numpy as np
from Bio import SeqIO

from dna2vec.multi_k_model import MultiKModel
from dna2vec.generators import DisjointKmerFragmenter, SlidingKmerFragmenter, SeqFragmenter
from tqdm import tqdm

import multiprocessing

path = '../../data/SILVA_138.1_SSURef_NR99_tax_silva.fasta'


class InvalidArgException(Exception):
    pass


def dna2vec(seq):
    seq = ''.join(splitter.get_acgt_seqs(str(seq)))
    vec = np.zeros(model.vec_dim)
    fragments = fragmenter.apply(rng, seq)
    for fragment in fragments:
        vec += model.vector(fragment)
    return vec / len(fragments)


def main():
    seqs=[record.seq for record in records]
    Y=[record.description.split()[1] for record in records]
    with multiprocessing.Pool(os.cpu_count()) as p:
        X=list(tqdm(p.imap(dna2vec,seqs),total=len(seqs)))
    # X=[]
    # Y=[]
    # for record in tqdm(list(records)):
    #     X.append(dna2vec(record.seq))
    #     Y.append(record.description.split()[1])
    np.savez(path, X=X, Y=Y)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='convert dna dataset to vectors')
    parser.add_argument('-m', '--model', help='model path', type=str,
                        default='pretrained/dna2vec-20231031-0333-k3to8-100d-10c-8210Mbp-sliding-pOW.w2v')
    parser.add_argument('-l', '--low', help='lower bound of k', type=int, default=3)
    parser.add_argument('-u', '--up', help='upper bound of k', type=int, default=8)
    parser.add_argument('-i', '--input', help='path to the input dataset', type=str, required=True)
    parser.add_argument('-t', '--type', help='type of the dataset', type=str, default='fasta')
    parser.add_argument('-f', '--fragment', help='style to fragment the sequence: disjoint or sliding',
                        choices=['disjoint', 'sliding'], default='sliding')
    parser.add_argument('-o', '--output', help='output path', type=str, default='inputs')
    parser.add_argument('-s', '--seed', help='random seed', type=int, default=0)
    args = parser.parse_args()

    model = MultiKModel(args.model)
    if not model.k_low <= args.low < args.up <= model.k_high:
        raise InvalidArgException(f'Invalid relationship: {model.k_low}<={args.low}<{args.up}<={model.k_high}')

    if args.fragment == 'disjoint':
        fragmenter = DisjointKmerFragmenter(args.low, args.up)
    elif args.fragment == 'sliding':
        fragmenter = SlidingKmerFragmenter(args.low, args.up)
    else:
        raise InvalidArgException('Invalid kmer fragmenter: {}'.format(args.kmer_fragmenter))

    splitter=SeqFragmenter()

    records = list(SeqIO.parse(args.input, args.type))
    rng = np.random.RandomState(args.seed)
    name = os.path.basename(args.input).split('.')[0]
    path = os.path.join(args.output, f'{name}_{args.low}_{args.up}_{args.fragment}_{args.seed}.npz')
    main()
