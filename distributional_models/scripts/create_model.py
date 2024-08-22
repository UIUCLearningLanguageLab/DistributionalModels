import torch

from ..models.srn import SRN
from ..models.lstm import LSTM
from ..models.mlp import MLP
from ..models.transformer import Transformer


def create_model(model_index, vocab_list, train_params):
    if train_params['random_seed'] is None:
        torch.manual_seed(model_index)
    else:
        torch.manual_seed(train_params['random_seed'] + model_index)

    if train_params['model_type'] == 'lstm':
        model = LSTM(vocab_list,
                     train_params['rnn_embedding_size'],
                     train_params['rnn_hidden_size'],
                     train_params['weight_init_hidden'],
                     train_params['dropout_rate'],
                     train_params['activation_function'])
        if train_params['weight_init_hidden'] != 0:
            model.init_weights(weight_init_hidden=train_params['weight_init_hidden'],
                               weight_init_linear=train_params['weight_init_linear'])

    elif train_params['model_type'] == 'srn':
        model = SRN(vocab_list,
                    train_params['rnn_embedding_size'],
                    train_params['rnn_hidden_size'],
                    train_params['weight_init_hidden'],
                    train_params['dropout_rate'],
                    train_params['activation_function'])
        if train_params['weight_init_hidden'] != 0:
            model.init_weights(weight_init_hidden=train_params['weight_init_hidden'],
                               weight_init_linear=train_params['weight_init_linear'])

    elif train_params['model_type'] == 'w2v':
        model = MLP(vocab_list,
                    train_params['w2v_embedding_size'],
                    train_params['w2v_hidden_size'],
                    train_params['weight_init_hidden'],
                    train_params['dropout_rate'],
                    train_params['activation_function'])

    elif train_params['model_type'] == 'transformer':
        model = Transformer(vocab_list,
                            train_params['sequence_length'],
                            train_params['transformer_embedding_size'],
                            train_params['transformer_num_heads'],
                            train_params['transformer_attention_size'],
                            train_params['transformer_hidden_size'],
                            train_params['weight_init_hidden'],
                            train_params['activation_function'],
                            train_params['device'])
    else:
        raise ValueError(f"Unrecognized model type {train_params['model_type']}")
    print(model.layer_dict)
    model.params = train_params
    model.set_device(train_params['device'])
    return model
