import pandas as pd
import click
import subprocess as sp
from os.path import isfile


@click.group()
def main():
    pass


class CmdRunner:

    def __init__(self, mixcr):
        self.mixcr = mixcr

    def run_cmd(self, subcmd):
        flag_file = subcmd.split()[-1]
        if not isfile(flag_file):
            self._run_cmd(subcmd)

    def _run_cmd(self, subcmd):
        cmd = f'{self.mixcr} {subcmd}'
        click.echo(cmd, err=True)
        sp.check_call(cmd, shell=True)

    def process_sample(self, sample_name, r1, r2, d='.'):
        sn = f'{d}/{sample_name}'
        self.run_cmd(
            f'align -p rna-seq -s hsa -OallowPartialAlignments=true {r1} {r2} {sn}.alignments.vdjca'
        )
        self.run_cmd(
            f'assemblePartial {sn}.alignments.vdjca {sn}.alignments_rescued_1.vdjca'
        )
        self.run_cmd(
            f'assemblePartial {sn}.alignments_rescued_1.vdjca {sn}.alignments_rescued_2.vdjca'
        )
        self.run_cmd(
            f'extend {sn}.alignments_rescued_2.vdjca {sn}.alignments_rescued_2_extended.vdjca'
        )
        self.run_cmd(
            f'assemble --write-alignments --report {sn}.report.txt {sn}.alignments_rescued_2_extended.vdjca {sn}.clones.clna'
        )
        self.run_cmd(
            f'assembleContigs --report {sn}.report.txt {sn}.clones.clna {sn}.full_clones.clns'
        )
        self.run_cmd(
            f'exportClones -c IGH -p fullImputed {sn}.full_clones.clns {sn}.full_clones_IGH.txt'
        )
        click.echo(f'Finished {sample_name}', err=True)


@main.command('run')
@click.option('-d', '--out-dir', default='.')
@click.option('-m', '--mixcr', default='mixcr')
@click.argument('sample_name')
@click.argument('read_1')
@click.argument('read_2')
def run_mixcr(out_dir, mixcr, sample_name, read_1, read_2):
    runner = CmdRunner(mixcr)
    runner.process_sample(sample_name, read_1, read_2, d=out_dir)


def countit(objs, multiplier=1):
    out = {}
    for el in objs:
        out[el] = 1 + out.get(el, 0)
    out = {k: multiplier * v for k, v in out.items()}
    return out


def get_binding_motifs(seq, multiplier=1):
    out = {'type_1': [], 'type_2a': [], 'type_2b': []}
    for i in range(len(seq) - 9 + 1):
        kmer = seq[i:i + 9]
        out['type_1'].append(kmer[3:8])
    for i in range(len(seq) - 15 + 1):
        kmer = seq[i:i + 15]
        tail = kmer[5] + kmer[7] + kmer[9] + kmer[10]
        out['type_2a'].append(kmer[4] + tail)
        out['type_2b'].append(kmer[2] + tail)
    counted = {k: countit(v, multiplier=multiplier) for k, v in out.items()}
    return counted


def get_full_v_gene_aa_seq(row):
    seqs = [
        'aaSeqImputedFR1',
        'aaSeqImputedCDR1',
        'aaSeqImputedFR2',
        'aaSeqImputedCDR2',
        'aaSeqImputedFR3',
        'aaSeqImputedCDR3',
        'aaSeqImputedFR4',
    ]
    out = ''
    for seq in seqs:
        out += row[seq]
    return out


def parse_mixcr_table(filepath):
    tbl = pd.read_csv(filepath, sep='\t')
    tbl['aa_full_seq'] = tbl.apply(get_full_v_gene_aa_seq, axis=1)
    out = {'type_1': {}, 'type_2a': {}, 'type_2b': {}}
    for _, row in tbl.iterrows():
        motifs = get_binding_motifs(row['aa_full_seq'], multiplier=row['cloneCount'])
        for kind, motif_counts in motifs.items():
            for motif, count in motif_counts.items():
                out[kind][motif] = out[kind].get(motif, 0) + count
    return out


@main.command('parse')
@click.option('-o', '--outfile', type=click.File('w'), default='-')
@click.argument('tables', nargs=-1)
def parse_mixcr(outfile, tables):
    out = {}
    for table in tables:
        sample_name = table.split('/')[-1].split('.')[0]
        motif_counts = parse_mixcr_table(table)
        for kind, val in motif_counts.items():
            out[(kind, sample_name)] = val
    out = pd.DataFrame.from_dict(out, orient='index')
    out.to_csv(outfile)


if __name__ == '__main__':
    main()
