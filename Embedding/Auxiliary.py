from pykeen.triples import TriplesFactory
from pykeen.pipeline import pipeline
from pykeen.models import predict
import pykeen.nn
from typing import List
import pandas as pd
import numpy as np
import statistics
from scipy.ndimage import gaussian_filter1d
import math
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.colors as mcolors
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
from sklearn.decomposition import PCA
from rdflib import Graph
from rdflib.plugins.sparql.processor import SPARQLResult
import sys
import torch


# # Load Train data
def load_dataset(path, name):
    triple_data = open(path + name).read().strip()
    data = np.array([triple.split('\t') for triple in triple_data.split('\n')])
    tf_data = TriplesFactory.from_labeled_triples(triples=data)
    return tf_data, triple_data


def create_model(tf_training, tf_testing, embedding, n_epoch, path, fold):
    results = pipeline(
        training=tf_training,
        testing=tf_testing,
        model=embedding,  # 'TransE',  #'RotatE'
        # stopper='early',
        # stopper_kwargs=dict(frequency=5, patience=2, relative_delta=0.002),
#         training_loop='sLCWA',
#         negative_sampler='bernoulli',
        negative_sampler_kwargs=dict(
        filtered=True,
        ),
        # Training configuration
        training_kwargs=dict(
            num_epochs=n_epoch,
            use_tqdm_batch=False,
        ),
        # Runtime configuration
        random_seed=1235,
        device='gpu',
    )
    model = results.model
    results.save_to_directory(path + embedding + str(fold))
    return model, results


# # Predict links (Head prediction)
def predict_heads(model, prop, obj, tf_testing):  # triples_factory=results.training
    predicted_heads_df = predict.get_head_prediction_df(model, prop, obj, triples_factory=tf_testing)
    return predicted_heads_df


# Filter the prediction by the head 'treatment_drug:treatment'. We are not interested in predict another links
def filter_prediction(predicted_heads_df, constraint):
    predicted_heads_df = predicted_heads_df[predicted_heads_df.head_label.str.contains(constraint)]
    return predicted_heads_df


def filter_entity(kg, c1, c2):
    sub = kg[(kg[0].str.contains(c1)) | (kg[0].str.contains(c2))][0].values
    obj = kg[(kg[2].str.contains(c1)) | (kg[2].str.contains(c2))][2].values
    entity = list(sub) + list(obj)
    entity = set(entity)
    return entity


def get_config(config_file):
    config = pd.read_csv(config_file, delimiter=";")  # 'config_G1.csv'
    models = config.model.values[0].split(',')
    epochs = config.epochs.values[0]
    k = config.k_fold.values[0]
    path = config.path.values[0]
    graph_name = config.graph_name.values[0]
    return models, epochs, k, path, graph_name


def reset_index(predicted_heads):
    predicted_heads.reset_index(inplace=True)
    predicted_heads.drop(columns=['index'], inplace=True)
    return predicted_heads


def pipeline_kge(args):
    models, epochs, k, path, graph_name = get_config(args)
    # models = ['TransH','RotatE', 'TransE', 'TransD', 'HolE', 'TransR', 'ERMLP', 'QuatE', 'RESCAL', 'SE', 'UM']
    models = ['TransH']
    for m in models:
        precision = 0
        recall = 0
        f_measure = 0
        for i in range(0, k):
            training, triple_dataset = load_dataset(path, 'train_set.ttl')
            testing, triple_dataset = load_dataset(path, 'test_set.ttl')
            # training, testing = tf_dataset.split(random_state=1234)
            
            model, results = create_model(training, testing, m, epochs, path, i + 1)
            #model = torch.load(path + m + str(i + 1) + '/trained_model.pkl') # , map_location='cpu'
#            predicted_heads_eff = predict_heads(model, 'ex:belong_to', 'ex:effective', tf_testing) #tf_training


def filter_prediction(predicted_heads_df, constraint):
    predicted_heads_df = predicted_heads_df[predicted_heads_df.head_label.str.contains(constraint)]
    predicted_heads_df = reset_index(predicted_heads_df)
    return predicted_heads_df


def filter_by_type(predicted_heads, triple_data, entity_type):
    list_entity = predicted_heads.head_label
    entity = []
    for s in list_entity:
        for triple in triple_data:
            b = [s, '<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>', entity_type] == triple
            if np.all(b):
                entity.append(s)
                break
    predicted_heads = predicted_heads.loc[predicted_heads.head_label.isin(entity)]
    predicted_heads = reset_index(predicted_heads)
    return predicted_heads, entity


def reset_index(predicted_heads):
    predicted_heads.reset_index(inplace=True)
    predicted_heads.drop(columns=['index'], inplace=True)
    return predicted_heads


def plot_score_value(score_values, title):
    plt.plot(score_values)
    plt.xlabel("Entities")
    plt.ylabel("Score")
    plt.title(title)
    plt.show()
    plt.close()


def predict_tail(model, sub, prop, tf_testing):  # triples_factory=results.training
    predicted_tails_df = predict.get_tail_prediction_df(model, sub, prop, triples_factory=tf_testing)
    predicted_tails_df = filter_prediction_tail(predicted_tails_df, 'Protein:')
    predicted_tails_df = reset_index(predicted_tails_df)
    return predicted_tails_df


def filter_prediction_tail(predicted_tails_df, constraint):
    predicted_tails_df = predicted_tails_df[predicted_tails_df.tail_label.str.contains(constraint)]
    return predicted_tails_df


def get_precision(predicted_tails, cut_off):
    tp_fp = predicted_tails.iloc[:cut_off]
    tp = tp_fp.loc[tp_fp.in_training == True].shape[0]
    prec = tp / tp_fp.shape[0]
    return prec, tp


def get_recall(predicted_tails, tp):
    tp_fn = predicted_tails.loc[predicted_tails.in_training == True].shape[0]
    rec = tp / tp_fn
    return rec


def get_f_measure(precision, recall):
    f_measure = 2 * (precision * recall) / (precision + recall)
    return f_measure


def get_learned_embeddings(model):
    entity_representation_modules: List['pykeen.nn.RepresentationModule'] = model.entity_representations
    relation_representation_modules: List['pykeen.nn.RepresentationModule'] = model.relation_representations

    entity_embeddings: pykeen.nn.Embedding = entity_representation_modules[0]
    relation_embeddings: pykeen.nn.Embedding = relation_representation_modules[0]

    entity_embedding_tensor: torch.FloatTensor = entity_embeddings()
    relation_embedding_tensor: torch.FloatTensor = relation_embeddings()
    return entity_embedding_tensor, relation_embedding_tensor


def create_dataframe_predicted_entities(entity_embedding_tensor, entity, training):
    df = pd.DataFrame(entity_embedding_tensor.cpu().detach().numpy())
    df['target'] = list(training.entity_to_id)
    new_df = df.loc[df.target.isin(list(entity))]
    return new_df.iloc[:, :-1], new_df
    # return df


def elbow_KMeans(matrix, k_min, k_max):
    model = KMeans()
    visualizer = KElbowVisualizer(model, k=(k_min, k_max))
    visualizer.fit(matrix)
    num_cls = visualizer.elbow_value_
    visualizer.show(outpath='Plots/elbow_KG.pdf', bbox_inches='tight')
    return num_cls


def plot_cluster(num_cls, new_df, n):
    X = new_df.copy()
    kmeans = KMeans(n_clusters=num_cls, random_state=0)
    new_df['cluster'] = kmeans.fit_predict(new_df)
    # define and map colors
    col = list(colors.cnames.values())
    col = col[:num_cls]
    index = list(range(num_cls))
    color_dictionary = dict(zip(index, col))
    new_df['c'] = new_df.cluster.map(color_dictionary)
    #####PLOT#####
    from matplotlib.lines import Line2D
    fig, ax = plt.subplots(1, figsize=(8, 8))
    # plot data
    pca = PCA(n_components=2).fit(X)
    pca_c = pca.transform(X)
    plt.scatter(pca_c[:, 0], pca_c[:, 1], c=new_df.c, alpha=0.6, s=50)

    # create a list of legend elemntes
    ## markers / records
    legend_elements = [Line2D([0], [0], marker='o', color='w', label='Cluster {}'.format(i + 1),
                              markerfacecolor=mcolor, markersize=10) for i, mcolor in enumerate(col)]
    # plot legend
    plt.legend(handles=legend_elements, loc='upper right', fontsize=16)
    # title and labels
    plt.title('Clusters of Entities predicted', loc='left', fontsize=22)
    plt.savefig(fname='Plots/KMeans_KG_' + str(n) + ".pdf", format='pdf', bbox_inches='tight')
    plt.show()


def plot_two_classes(new_df, c1, c2, c1_label, c2_label):
    new_df['cls'] = ''
    new_df.loc[new_df.target.isin(c1), 'cls'] = c1_label
    new_df.loc[new_df.target.isin(c2), 'cls'] = c2_label
    X = new_df.iloc[:, :-2].copy()

    # define and map colors
    col = list(colors.cnames.values())
    # col = [col[9], col[3]]
    col = [mcolors.CSS4_COLORS['brown'], mcolors.CSS4_COLORS['lightcoral']]
    index = [c1_label, c2_label]
    color_dictionary = dict(zip(index, col))
    new_df['c'] = new_df.cls.map(color_dictionary)
    #####PLOT#####
    from matplotlib.lines import Line2D
    fig, ax = plt.subplots(1, figsize=(8, 8))
    # plot data
    pca = PCA(n_components=2).fit(X)
    pca_c = pca.transform(X)
    plt.scatter(pca_c[:, 0], pca_c[:, 1], c=new_df.c, s=50)  # alpha=0.6,

    # create a list of legend elemntes
    ## markers / records
    legend_elements = [Line2D([0], [0], marker='o', color='w', label=key,
                              markerfacecolor=mcolor, markersize=10) for key, mcolor in color_dictionary.items()]
    # plot legend
    plt.legend(handles=legend_elements, loc='upper right', fontsize=16)
    # title and labels
    # if n == 1:
    #     plt.title('Treatments in ' + '${\cal{T\_KG}}_{basic}$', loc='left', fontsize=22)
    # elif n == 2:
    #     plt.title('Treatments in ' + '$\cal{T\_KG}$', loc='left', fontsize=22)
    # else:
    #     plt.title('Treatments in ' + '${\cal{T\_KG}}_{random}$', loc='left', fontsize=22)
    #plt.savefig(fname='Plots/PCA_KG_' + str(n) + ".png", format='png', bbox_inches='tight', dpi=300, transparent=True)
    plt.savefig(fname='Plots/PCA' + c1_label+'_'+c2_label + '.pdf', format='pdf', bbox_inches='tight')
    plt.show()


def plot_KGE(new_df, entity_type):
    new_df['cls'] = ''
    new_df.reset_index(inplace=True)
    new_df.drop(columns=['index'], inplace=True)
    for i in range(new_df.shape[0]):
        t = new_df.iloc[i]['target']
        # new_df.iloc[i]['cls'] = entity_type[t]
        new_df.at[i, 'cls'] = entity_type[t]
    X = new_df.iloc[:, :-2].copy()

    # define and map colors
    index = list(new_df.cls.unique())
    col = list(colors.cnames.values())[:len(index)]
    color_dictionary = dict(zip(index, col))
    new_df['c'] = new_df.cls.map(color_dictionary)
    #####PLOT#####
    from matplotlib.lines import Line2D
    fig, ax = plt.subplots(1, figsize=(8, 8))
    # plot data
    pca = PCA(n_components=2).fit(X)
    pca_c = pca.transform(X)
    plt.scatter(pca_c[:, 0], pca_c[:, 1], c=new_df.c, s=50)  # alpha=0.6,

    # create a list of legend elemntes
    ## markers / records
    legend_elements = [Line2D([0], [0], marker='o', color='w', label=key,
                              markerfacecolor=mcolor, markersize=10) for key, mcolor in color_dictionary.items()]
    # plot legend
    plt.legend(handles=legend_elements, loc='upper right', fontsize=16)
    # title and labels
    # if n == 1:
    #     plt.title('Treatments in ' + '${\cal{T\_KG}}_{basic}$', loc='left', fontsize=22)
    # elif n == 2:
    #     plt.title('Treatments in ' + '$\cal{T\_KG}$', loc='left', fontsize=22)
    # else:
    #     plt.title('Treatments in ' + '${\cal{T\_KG}}_{random}$', loc='left', fontsize=22)
    #plt.savefig(fname='Plots/PCA_KG_' + str(n) + ".png", format='png', bbox_inches='tight', dpi=300, transparent=True)
    plt.savefig(fname='Plots/PCA.pdf', format='pdf', bbox_inches='tight')
    plt.show()
    

def load_graph(file_name):
    g1 = Graph()
    g1.parse(file_name, format="ttl")
    return g1


def sparql_results_to_df(results: SPARQLResult) -> pd.DataFrame:
    """
    Export results from an rdflib SPARQL query into a `pandas.DataFrame`,
    using Python types. See https://github.com/RDFLib/rdflib/issues/1179.
    """
    return pd.DataFrame(
        data=([None if x is None else x.toPython() for x in row] for row in results),
        columns=[str(x) for x in results.vars],
    )


def get_triple(graph, predicted_heads, entity_type):
    list_entity = list(predicted_heads.head_label)
    list_entity = ', '.join(list_entity)
    query = """    
    select distinct ?s
    where {
        ?s <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> """ + entity_type + """
        FILTER (?s in (""" + str(list_entity) + """))
        }
        """
    results = graph.query(query)
    df_cls = sparql_results_to_df(results)
    df_cls['s'] = '<' + df_cls['s'].astype(str) + '>'
    entity = list(df_cls.s)

    predicted_heads = predicted_heads.loc[predicted_heads.head_label.isin(entity)]
    predicted_heads = reset_index(predicted_heads)
    return predicted_heads, entity
