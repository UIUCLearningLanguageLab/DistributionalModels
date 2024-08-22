import torch
from pathlib import Path
from ..corpora.xAyBz import XAYBZ
from ..models.srn import SRN
from ..models.lstm import LSTM
from ..models.mlp import MLP
from ..models.transformer import Transformer
from ..scripts.create_model import create_model
import yaml
import re


def construct_python_tuple(loader, node):
    return tuple(loader.construct_sequence(node))


yaml.SafeLoader.add_constructor("tag:yaml.org,2002:python/tuple", construct_python_tuple)


def load_model(param_path, param_name, model_index, epoch_list):
    for param in param_path.iterdir():
        if param.name == param_name:
            corpus = None
            for child in sorted(param.iterdir(), reverse=True):
                if 'param2val' in child.name:
                    with open(child, 'r') as file:
                        param2val = yaml.safe_load(file)
                    corpus = XAYBZ(sentence_sequence_rule=param2val['sentence_sequence_rule'],
                                   random_seed=param2val['random_seed'],
                                   ab_category_size=param2val['ab_category_size'],
                                   num_ab_categories=param2val['num_ab_categories'],
                                   num_omitted_ab_pairs=param2val['num_omitted_ab_pairs'])
                    missing_training_words = corpus.create_vocab()
                elif child.is_dir():
                    model_type = param2val['model_type']
                    model_path = child / 'saves'
                    models = [model for model in model_path.iterdir() if child.is_dir()]
                    sorted_models = sorted(models, key=lambda x: x.stat().st_mtime)
                    model_chosen = sorted_models[model_index]
                    model_list = []
                    for epoch in epoch_list:
                        # model = create_model(model_index, corpus.vocab_list, param2val)
                        epoch_file = [f for f in model_chosen.iterdir() if f.name.split('.')[0] == epoch][0]
                        if model_type == 'srn':
                            model = SRN.load_model(epoch_file)
                        elif model_type == 'lstm':
                            model = LSTM.load_model(epoch_file)
                        elif model_type == 'mlp':
                            model = MLP.load_model(epoch_file)
                        else:
                            model = Transformer.load_model(epoch_file)
                        model_list.append(model)

    return param2val, corpus, model_list


def load_models(param_path, param_name, num_epochs=None, epoch_selected=None, num_models=None, model_selected=None):
    for param in param_path.iterdir():
        if param.name == param_name:
            corpus = None
            for child in sorted(param.iterdir(), reverse=True):
                if 'param2val' in child.name:
                    with open(child, 'r') as file:
                        param2val = yaml.safe_load(file)
                    corpus = XAYBZ(sentence_sequence_rule=param2val['sentence_sequence_rule'],
                                   random_seed=param2val['random_seed'],
                                   ab_category_size=param2val['ab_category_size'],
                                   num_ab_categories=param2val['num_ab_categories'],
                                   num_omitted_ab_pairs=param2val['num_omitted_ab_pairs'])
                    missing_training_words = corpus.create_vocab()
                elif child.is_dir():
                    model_type = param2val['model_type']
                    model_path = child / 'saves'
                    models = [model for model in model_path.iterdir() if model.is_dir() and model.name != 'extra_eval']
                    sorted_models = sorted(models, key=lambda x: x.stat().st_mtime)
                    if num_models is not None:
                        sorted_models = [sorted_models[index] for index in range(num_models)]
                    elif model_selected is not None:
                        sorted_models = [sorted_models[index] for index in model_selected]
                    models_dict = {}
                    for i, m in enumerate(sorted_models):
                        model_list = []
                        epoch_list = sorted(m.iterdir(), key=lambda f: int(re.search(r'(\d+)', f.name).group(1)))
                        if num_epochs is not None:
                            epoch_list = [epoch_list[index] for index in range(num_epochs)]
                        elif epoch_selected is not None:
                            epoch_list = [epoch_list[index] for index in epoch_selected]
                        for epoch_file in epoch_list:
                            if model_type == 'srn':
                                model = SRN.load_model(epoch_file)
                            elif model_type == 'lstm':
                                model = LSTM.load_model(epoch_file)
                            elif model_type == 'mlp':
                                model = MLP.load_model(epoch_file)
                            else:
                                model = Transformer.load_model(epoch_file)
                            model_list.append(model)
                        models_dict[i] = model_list
    return param2val, corpus, models_dict
