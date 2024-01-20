from ..models.srn import SRN
from ..models.lstm import LSTM
from ..models.mlp import MLP
from ..models.transformer import Transformer


def create_model(vocab_list, train_params):

    if train_params['model_type'] == 'lstm':
        model = LSTM(vocab_list,
                     train_params['rnn_embedding_size'],
                     train_params['rnn_hidden_size'],
                     train_params['weight_init'],
                     train_params['dropout_rate'])

    elif train_params['model_type'] == 'srn':
        model = SRN(vocab_list,
                    train_params['rnn_embedding_size'],
                    train_params['rnn_hidden_size'],
                    train_params['weight_init'],
                    train_params['dropout_rate'])

    elif train_params['model_type'] == 'w2v':
        model = MLP(vocab_list,
                    train_params['w2v_embedding_size'],
                    train_params['w2v_hidden_size'],
                    train_params['weight_init'],
                    train_params['dropout_rate'])

    elif train_params['model_type'] == 'transformer':
        model = Transformer(vocab_list,
                            train_params['sequence_length'],
                            train_params['transformer_embedding_size'],
                            train_params['transformer_num_heads'],
                            train_params['transformer_attention_size'],
                            train_params['transformer_hidden_size'],
                            train_params['weight_init'],
                            train_params['device'])
    else:
        raise ValueError(f"Unrecognized model type {train_params['model_type']}")
    print(model.layer_dict)
    model.set_device(train_params['device'])
    return model
