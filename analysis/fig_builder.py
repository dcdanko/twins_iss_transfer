
import pandas as pd
import os
from umap import UMAP
from cap2.capalyzer.table_builder import CAPTableBuilder
from cap2.capalyzer.pangea import PangeaFileSource
from cap2.capalyzer.pangea.utils import get_pangea_group
import pandas as pd
from umap import UMAP
import networkx as nx
from plotnine import *
from sklearn.decomposition import PCA
import community
import pysam
from gimmebio.seqs import reverseComplement


def cache_pandas(cache_name):
    def decorator(func):
        def wrapper(self, *args, **kwargs):
            if os.path.isfile(cache_name):
                return pd.read_csv(cache_name)
            tbl = func(self, *args, **kwargs)
            tbl.to_csv(cache_name)
            return tbl
        return wrapper
    return decorator


def logger(i, x):
    if i % 10 == 0:
        print(i)


def proportions(tbl):
    """Return a DataFrame with each entry divided by its row sum."""
    tbl = (tbl.T / tbl.T.sum()).T
    return tbl


def umap(mytbl, **kwargs):
    """Retrun a Pandas dataframe with UMAP, make a few basic default decisions."""
    metric = 'jaccard'
    if mytbl.shape[0] == mytbl.shape[1]:
        metric = 'precomputed'
    n_comp = kwargs.get('n_components', 2)
    umap_tbl = pd.DataFrame(UMAP(
        n_neighbors=kwargs.get('n_neighbors', min(100, int(mytbl.shape[0] / 4))),
        n_components=n_comp,
        metric=kwargs.get('metric', metric),
        random_state=kwargs.get('random_state', 42)
    ).fit_transform(mytbl))
    umap_tbl.index = mytbl.index
    umap_tbl = umap_tbl.rename(columns={i: f'C{i}' for i in range(n_comp)})
    return umap_tbl


def condense_pileups(fig_data, raw_piles, sparse):
    piles = raw_piles.copy()
    for col in ['time_label', 'subject', 'kind']:
        mycol = fig_data.metadata()[col]
        piles[col] = piles['sample_name'].map(lambda x: mycol.get(x, 'unknown'))

    seqs = piles.query('subject == "TW"')['seq'].value_counts() > (1 * 1000 // sparse)
    seqs = set(seqs[seqs].index)
    piles = piles.query('seq in @seqs')

    piles['time_label'] = pd.Categorical(
        piles['time_label'],
        categories=[
            'before',
            'flight',
            'after',
            'unknown',
        ],
        ordered=True,
    )

    def sum_pileup(tbl):
        #tbl = tbl.drop(columns=['read_results'])
        tbl = tbl.groupby(('seq', 'pos'), as_index=False).sum()
        return tbl

    piles_condensed = piles.groupby(('time_label', 'subject', 'kind')).apply(sum_pileup)
    piles_condensed = piles_condensed.reset_index(level=[0, 1, 2])
    piles_condensed = piles_condensed.query('read_count > 1')
    return piles_condensed


def time_map(times):
    if not times:
        return 'not_observed'
    if 'before' in times:
        return 'no_transfer'
    if 'flight' in times:
        if 'after' in times:
            return 'persistent'
        return 'transient'
    if 'after' in times:
        return 'after_only'
    return 'unknown'


def label_positions(position_tbl):
    def label_position(tbl):
        subjects = tbl['subject'].unique()
        if 'TW' not in subjects:
            return {'oral': 'iss_only', 'fecal': 'iss_only'}
        elif 'ISS' not in subjects:
            return {'oral': 'tw_only', 'fecal': 'tw_only'}
        tbl = tbl.query('subject == "TW"')
        times_oral = tbl.query('kind == "oral"')['time_label'].unique()
        times_fecal = tbl.query('kind == "fecal"')['time_label'].unique()
        out = {
            'oral': time_map(times_oral),
            'fecal': time_map(times_fecal),
        }
        return out

    subjects = ['TW', 'ISS']
    position_tbl = position_tbl.copy()
    position_tbl['kind'] = position_tbl['kind'].map(lambda x: 'oral' if x in ['buccal', 'saliva'] else x)
    labeled_positions = position_tbl \
        .query('subject in @subjects') \
        .query('read_count > 0') \
        .groupby(['seq', 'pos']) \
        .apply(label_position)
    labeled_positions = labeled_positions.reset_index(level=[0, 1])
    labeled_positions['oral'] = labeled_positions[0].map(lambda x: x['oral'])
    labeled_positions['fecal'] = labeled_positions[0].map(lambda x: x['fecal'])
    labeled_positions = labeled_positions.drop(columns={0: 'position_label'})
    
    def both(row):
        o, f = row['oral'], row['fecal']
        if o == f:
            return o
        if f == 'not_observed':
            return o
        if o == 'not_observed':
            return f
        no_transfers = ['unknown', 'no_transfer']
        if o in no_transfers and f in no_transfers:
            return 'mixed_non_transfer'
        elif o in no_transfers or f in no_transfers:
            return 'mixed_part_transfer'
        return 'mixed_transfer'
        
    labeled_positions['both'] = labeled_positions.apply(both, axis=1)
    transfers = ['persistent', 'transient', 'mixed_transfer', 'mixed_part_transfer']
    #labeled_positions = labeled_positions.query('both in @transfers')
    return labeled_positions


def condense_and_label_organism_positions(figs, organism, sparse=100):
    raw_piles = figs.pileup(organism, sparse=sparse)  
    piles_condensed = condense_pileups(figs, raw_piles, sparse)
    positions = label_positions(piles_condensed)
    return positions, piles_condensed


def get_mismatches(rec):
    qseq = rec.get_forward_sequence().upper()
    if rec.is_reverse:
        qseq = reverseComplement(qseq)
    rseq = rec.get_reference_sequence().upper()
    for qpos, rpos in rec.get_aligned_pairs():
        if qpos == None or rpos == None:
            continue  # no indels yet
        q = qseq[qpos]
        r = rseq[rpos - rec.reference_start]
        if q != r:
            position = (rec.reference_name, rpos)
            change = (r, q)
            yield (position, change)
            
            
def filter_to_region(node, contig=None, coords=None):
    ((seq, coord), miss) = node
    if contig and seq != contig:
        return False
    if coords and coord < coords[0]:
        return False
    if coords and coord > coords[1]:
        return False
    return True

def build_graph(rec_iter, G=nx.Graph(), contig=None, coords=None):
    for rec in rec_iter:
        misses = list(get_mismatches(rec))
        misses = [
            miss for miss in misses
            if filter_to_region(miss, contig=contig, coords=coords)
        ]
        for missA in misses:
            for missB in misses:
                if missA == missB:
                    break
                try:
                    w = G[missA][missB]['weight']
                except KeyError:
                    w = 0
                G.add_edge(missA, missB, weight=w + 1)
    return G


class TwinsFiguresData:

    def __init__(self):
        self.twins = CAPTableBuilder('twins', PangeaFileSource(get_pangea_group('Mason Lab', 'NASA Twins', 'dcdanko@gmail.com', )))
        self.iss = CAPTableBuilder('iss', PangeaFileSource(get_pangea_group('Mason Lab', 'NASA ISS', 'dcdanko@gmail.com', )))

    def species(self):
        twins = self.twins.taxa_read_counts()
        iss = self.iss.taxa_read_counts()
        tbl = pd.concat([twins, iss])
        tbl = tbl.fillna(0)
        tbl = tbl[[c for c in tbl.columns if 's__' in c and 't__' not in c]]
        tbl.columns = [c.split('__')[-1] for c in tbl.columns]
        tbl = proportions(tbl)
        return tbl

    def twins_species(self):
        twins_meta = self.metadata().query('subject in ["TW", "HR"]')
        twins_samps = list(twins_meta.index)
        species = self.species()
        species = species.loc[twins_samps]
        return species

    def twins_species_set(self, in_sample_thresh=10, proportion_of_samples=0.25):
        """Return a set of taxa with rel abund > `in_sample_thresh` (ppm) in `proportion_of_samples`."""
        t_twins = self.twins_species()
        taxa_twins = t_twins > (in_sample_thresh / 1000000)
        taxa_twins = taxa_twins.mean() > proportion_of_samples
        taxa_twins = set(taxa_twins.index[taxa_twins])
        return taxa_twins

    def iss_species(self):
        iss_meta = self.metadata().query('subject not in ["TW", "HR"]')
        iss_samps = list(iss_meta.index)
        species = self.species()
        species = species.loc[iss_samps]
        return species

    def iss_species_set(self, in_sample_thresh=10, proportion_of_samples=0.25):
        """Return a set of taxa with rel abund > `in_sample_thresh` (ppm) in `proportion_of_samples`."""
        t_iss = self.iss_species()
        taxa_iss = t_iss > (in_sample_thresh / 1000000)
        taxa_iss = taxa_iss.mean() > proportion_of_samples
        taxa_iss = set(taxa_iss.index[taxa_iss])
        return taxa_iss

    def metadata(self):
        twins = self.twins.metadata()
        twins['pma_treated'] = False
        iss = self.iss.metadata()
        iss = iss.rename(columns={
            'COLLECTION_TIMESTAMP': 'date',
            'JPL_PMA': 'pma_treated',
            'JPL project name': 'kind',
            'host_subject_id': 'subject',
        })
        iss['subject'] = iss['kind']
        iss['kind'] = iss['kind'].map(lambda x: {
            'ISS-HEPA': 'Air',
            'ISS-MO': 'Surface',
            'SpaceX': 'Surface',
        }.get(x, x))
        tbl = pd.concat([twins, iss])
        tbl = tbl[['date', 'pma_treated', 'kind', 'subject', 'flight', 'during_flight']]
        tbl['date'] = pd.to_datetime(tbl['date'])
        time_labels = tbl.apply(lambda row: (row['subject'], row['during_flight']), axis=1)

        def get_time_label(x):
            try:
                subj, time_label = time_labels[x]
                
                if subj and time_label and subj == subj and time_label == time_label and time_label != 'nan':
                    return subj, time_label
            except KeyError:
                pass
        
            if 'TW' in x:
                return 'TW', 'unknown'
            if 'HR' in x:
                return 'HR', 'unknown'
            else:
                return 'ISS', 'unknown'
            return tbl

        tbl['time_label'] = tbl.index.map(lambda x: get_time_label(x)[1])
        tbl['time_label'] = pd.Categorical(
            tbl['time_label'],
            categories=[
                'before',
                'flight',
                'after',
                'unknown',
            ],
            ordered=True,
        )
        tbl['subject'] = tbl.index.map(lambda x: get_time_label(x)[0])
        tbl['subject'] = pd.Categorical(
            tbl['subject'],
            categories=[
                'TW',
                'ISS',
                'HR',
                'unknown',
            ],
            ordered=True,
        )
        return tbl

    def tw_pca(self):
        meta = self.metadata()
        tw_meta = meta.query('subject == "TW"')
        tw_samps = list(tw_meta.index)
        tw_taxa = self.species().loc[tw_samps].fillna(0)
        pca = PCA()
        tw_pca = pd.DataFrame(pca.fit_transform(tw_taxa))
        tw_pca.index = tw_taxa.index
        tw_pca = tw_pca.rename(columns={i: f'PC{i + 1}' for i in range(len(tw_pca.columns))})
        tw_pca = pd.concat([tw_meta, tw_pca], axis=1)
        return tw_pca, pca
    
    def pileup(self, organism, sparse=1):
        twins = self.twins.strain_pileup(organism=organism, sparse=sparse)
        iss = self.iss.strain_pileup(organism=organism, sparse=sparse)
        tbl = pd.concat([twins, iss])
        return tbl
    
    def snp_graph(self, organism):
        graphs = {}
        for key in ['TW', 'ISS']:
            path = f'/home/dcdanko/Data/twins_2020_09/{organism}__{key}__graph.adjlist'
            if not os.path.isfile(path):
                break
            graphs[key] = nx.read_adjlist(path)
        if len(graphs) == 2:
            return graphs
        graphs = self._snp_graph(organism)
        for key, graph in graphs.items():
            path = f'/home/dcdanko/Data/twins_2020_09/{organism}__{key}__graph.adjlist'
            nx.write_adjlist(graph, path)
        return graphs
    
    def _snp_graph(self, organism):
        graphs = {}
        meta = self.metadata()

        def process_file(sample_name, filename):
            subject = meta.loc[sample_name, 'subject']
            graphs[subject] = graphs.get(subject, nx.Graph())
            graph = graphs[subject]
            bam = pysam.AlignmentFile(filename, 'rb')
            build_graph(bam, G=graph)
        
        for tb in [self.twins, self.iss]:
            for sample_name, filepath in tb.file_source('cap2::experimental::align_to_genome', f'bam__{organism}'):
                try:
                    process_file(sample_name, filepath)
                    print('processed', sample_name, filepath)
                except KeyError:
                    continue

        return graphs
            
    def snp_graph_time(self, organism):
        graphs = {}
        for key in ['TW-before', 'TW-flight', 'TW-after','ISS-unknown']:
            path = f'{organism}__{key}__graph.adjlist'
            if not os.path.isfile(path):
                break
            graphs[key] = nx.read_adjlist(path)
        if len(graphs) == 4:
            return graphs
        graphs = self._snp_graph_time(organism)
        for key, graph in graphs.items():
            path = f'{organism}__{key}__graph.adjlist'
            nx.write_adjlist(graph, path)
        return graphs
    
    def _snp_graph_time(self, organism):
        graphs = {}
        meta = self.metadata()

        def process_file(sample_name, filename):
            subject = meta.loc[sample_name, 'subject'] + '-' + meta.loc[sample_name, 'time_label']
            graphs[subject] = graphs.get(subject, nx.Graph())
            graph = graphs[subject]
            bam = pysam.AlignmentFile(filename, 'rb')
            build_graph(bam, G=graph)
        
        for tb in [self.twins, self.iss]:
            for sample_name, filepath in tb.file_source('cap2::experimental::align_to_genome', f'bam__{organism}'):
                try:
                    process_file(sample_name, filepath)
                    print('processed', sample_name, filepath)
                except KeyError:
                    continue

        return graphs


class TwinsFigures:

    def __init__(self, fig_data):
        self.fig_data = fig_data

    def __getattr__(self, attr):
        return getattr(self.fig_data, attr)

    def taxa_umap_scatter(self):
        taxa = self.species()
        u = umap(taxa, metric='manhattan')
        u = pd.concat([self.metadata(), u], axis=1)
        plot = (
            ggplot(u, aes(x='C0', y='C1', color='subject', shape='kind')) +
                geom_point(size=12, colour="black") +
                geom_point(size=10) +
                theme_minimal() +
                scale_color_brewer(type='qualitative', palette=6, direction=1) +
                theme_minimal() +
                labs(color='Twin', shape='Kind') +
                theme(
                    text=element_text(size=50),
                    panel_grid_major=element_blank(),
                    panel_grid_minor=element_blank(),
                    legend_position='right',
                    axis_text_x=element_blank(),
                    axis_title_x=element_blank(),
                    axis_text_y=element_blank(),
                    axis_title_y=element_blank(),
                    panel_border=element_rect(colour="black", fill='none', size=2),
                    figure_size=(16, 12),
                )
        )
        return plot

    def taxa_pca_scatter(self):
        tw_pca, _ = self.tw_pca()

        def plot(a, b):
            return (
                ggplot(tw_pca, aes(x=f'PC{a}', y=f'PC{b}', color='during_flight', shape='kind')) +
                    geom_point(size=12, colour="black") +
                    geom_point(size=10) +
                    theme_minimal() +
                    scale_color_brewer(type='qualitative', palette=6, direction=1) +
                    theme_minimal() +
                    labs(color='Time', shape='Kind') +
                    theme(
                        text=element_text(size=50),
                        panel_grid_major=element_blank(),
                        panel_grid_minor=element_blank(),
                        legend_position='right',
                        axis_text_x=element_blank(),
                        #axis_title_x=element_blank(),
                        axis_text_y=element_blank(),
                        #axis_title_y=element_blank(),
                        panel_border=element_rect(colour="black", fill='none', size=2),
                        figure_size=(16, 12),
                    )
            )
        return [plot(1, 4), plot(5, 4)]

    def plot_twins_species_set(self, species, kind='fecal'):
        tw_meta = self.metadata().query('subject in ["TW", "HR"]')

        t = self.twins_species()
        t = t[set(species) & set(t.columns)].fillna(0)
        t = 1000 * 1000 * t
        t = pd.concat([tw_meta, t], axis=1)
        t = t.query('kind == @kind')
        t = t.melt(id_vars=tw_meta.columns)
        t = t.query('value > 0')
        return (
            ggplot(t, aes(x='date', y='value', color='flight', shape='subject')) +
                geom_line(color='black', size=3) +
                geom_point(size=12, colour="black") +
                geom_point(size=10) +
                facet_grid('variable~.', scales='free') +
                theme_minimal() +
                scale_color_brewer(type='qualitative', palette=6, direction=1) +
                theme_minimal() +
                xlab(f'Date') +
                ylab(f'Rel. Abundance (PPM)') +
                labs(color='Flight', shape='Kind') +
                theme(
                    text=element_text(size=14),
                    panel_grid_major=element_blank(),
                    panel_grid_minor=element_blank(),
                    axis_text_x=element_text(angle=90),
                    legend_position='right',
                    panel_border=element_rect(colour="black", fill='none', size=2),
                    figure_size=(16, 36),
                )
        )    

