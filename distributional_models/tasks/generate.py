import copy
import torch


def generate_sequence(model, corpus, tokens=("look", "at"), sequence_length=10, temperature=0.8):

    input_token_list = list(tokens)
    final_token_list = copy.deepcopy(input_token_list)

    input_index_list = []
    for token in input_token_list:
        if token in corpus.vocab_index_dict:
            input_index_list.append(corpus.vocab_index_dict[token])
        else:
            return f"Prime word {token} not in vocab"

    model.init_network(batch_size=1)
    model.eval()

    output = None
    for index in input_index_list:
        input_tensor = torch.tensor([index]).unsqueeze(0).to(model.device)
        output = model(input_tensor)

    if output is not None:
        output_distribution = output.detach().view(-1).div(temperature).exp()
        top_i = torch.multinomial(output_distribution, 1)[0]

        final_token_list.append(corpus.vocab_list[top_i])

        for i in range(sequence_length-1):
            input_tensor = top_i.unsqueeze(0).unsqueeze(0).to(model.device)
            output = model(input_tensor)
            output_distribution = output.detach().view(-1).div(temperature).exp()
            if torch.isnan(output_distribution).any() or torch.isinf(output_distribution).any():
                print("NaN or Inf values in output_distribution")
            try:
                top_i = torch.multinomial(output_distribution, 1)[0]
            except:
                print(output_distribution)
            final_token_list.append(corpus.vocab_list[top_i])

    output_string = ' '.join(final_token_list)

    return output_string



            # output_distribution = output.detach.view(-1).div(temperature).exp()
            #
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