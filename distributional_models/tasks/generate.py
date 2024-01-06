import copy
import torch


def generate_sequence(model, prime_token_list=("look", "at"), sequence_length=10, temperature=0.8):
    prime_token_list = list(prime_token_list)

    prime_id_list = []
    all_tokens_in_vocab = True
    for token in prime_token_list:
        if token in model.vocab_index_dict:
            prime_id_list.append(model.vocab_index_dict[token])
        else:
            all_tokens_in_vocab = False

    if all_tokens_in_vocab and len(prime_id_list) > 0:

        predicted_list = copy.deepcopy(prime_token_list)
        model.init_layer_states()
        model.eval()

        for i in range(len(prime_id_list)):
            output = model.forward(prime_id_list[i])

            # output_distribution = output.detach.view(-1).div(temperature).exp()
            # top_i = torch.multinomial(output_distribution, 1)[0]
            # predicted_char = self.vocab_list[top_i]
            # predicted_list.append(predicted_char)

        # for i in range(sequence_length):
        #
        #
        #
        #
		# for i in range(len(prime_id_tensor_list) - 1):
		# 	x = prime_id_tensor_list[i].unsqueeze(0)
		# 	output, hidden_state_list = self.forward(x.to(self.device), hidden_state_list)
		# x = prime_id_tensor_list[-1].unsqueeze(0)
		# for i in range(self.params.PREDICT_LENGTH):
		# 	output, hidden_state_list = self.forward(x.to(self.device), hidden_state_list)
		# 	output_distribution = output.data.view(-1).div(self.params.TEMPERATURE).exp()
		# 	top_i = torch.multinomial(output_distribution, 1)[0]
		# 	predicted_char = self.vocab_list[top_i]
		# 	predicted_list.append(predicted_char)
		# 	x = torch.tensor([top_i])
		# if unit_type == "words":
		# 	predicted_string = " ".join(predicted_list)
		# else:
		# 	predicted_string = "".join(predicted_list)
		# return predicted_string