import pandas as pd
import torch.nn.functional as F


def output_probabilities(model, input_sequence_list, sequence_category_list, average_category=False):

    # [A11, y1, B12, ., A12, y1, B11, .]
    # [(present_A, []), (), (), (), (), (), (), ()]

    # [for whole vocab, you assign each output/vocab to a condition cateogry (e.g. omitted B1]

    df = pd.DataFrame(columns=["input_token_category", "input_token", "output_token_category", "output_token", "p"])

    for sequence in input_sequence_list:
        for token in sequence:
            model.eval()
            x = vocab_index_dict[token]
            output = model.forward(x)
            softmax_tensor = F.softmax(output, dim=0)

            new_row_data = {"input_token_category": 0,
                            "input_token": 0,
                            "output_token_category": 0,
                            "output_token": 0,
                            "p": 0}

            # Create a DataFrame (or Series) for the new row
            new_row_df = pd.DataFrame([new_row_data])

            # Concatenate the new row with the existing DataFrame
            df = pd.concat([df, new_row_df], ignore_index=True)

    pass