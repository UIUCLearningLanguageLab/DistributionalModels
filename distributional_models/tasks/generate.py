import copy
import torch
import numpy as np


def generate_sequence(model, tokens=("look", "at"), sequence_length=10, temperature=0.8):
    input_token_list = list(tokens)
    final_token_list = copy.deepcopy(input_token_list)

    # Convert tokens to indices
    # input_index_list = [model.vocab_index_dict.get(token, -1) for token in input_token_list]
    # if -1 in input_index_list:
    #     return "Prime word not in vocab"

    # Check for model's sequence length attribute
    max_seq_length = getattr(model, 'sequence_length', sequence_length)

    if model.model_type in ['lstm', 'srn']:
        model.init_network(batch_size=1)
    else:
        model.init_network()
    model.eval()

    for _ in range(sequence_length - len(input_token_list)):
        # Prepare the input tensor
        # input_tensor = torch.tensor([input_index_list[-max_seq_length:]]).to(model.device)
        input_token = input_token_list[-max_seq_length:]
        output_list, _ = model.test_sequence(input_token)
        output = output_list[0]
        # output = model(input_tensor)
        if model.model_type in ['transformer', 'mlp']:
            last_output = output[0, -1, :]
        else:
            last_output = output
        # # Generate next token
        last_output = torch.tensor(last_output)
        output_distribution = last_output.detach().view(-1).div(temperature).exp()

        if torch.isnan(output_distribution).any() or torch.isinf(output_distribution).any():
            print("NaN or Inf values in output_distribution")
            break

        try:
            top_i = torch.multinomial(output_distribution, 1)[0].item()
        except RuntimeError as e:
            print("Error during sampling:", e)
            break

        # # Append the predicted index and corresponding token
        input_token_list.append(model.vocab_list[top_i])

        final_token_list.append(model.vocab_list[top_i])
        #
        # # Ensure the sequence does not exceed the maximum length
        # if len(input_index_list) > max_seq_length:
        #     input_index_list.pop(0)

    output_string = ' '.join(final_token_list)
    return output_string
