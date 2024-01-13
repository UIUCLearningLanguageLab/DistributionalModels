import copy
import re
import random
import pandas as pd
from .corpus import Corpus


class Childes(Corpus):

    def __init__(self):
        super().__init__()

        self.input_path = None
        self.language = None
        self.age_range_tuple = None
        self.sex_list = None
        self.collection_name = None
        self.add_punctuation = None
        self.exclude_target_child = None
        self.order = None

    def get_documents_from_childes_db_file(self, input_path=None, language=None, age_range_tuple=None, sex_list=None,
                                           collection_name=None, add_punctuation=True, exclude_target_child=True,
                                           order="age_ordered", num_documents=None):

        self.input_path = input_path
        self.language = language
        self.age_range_tuple = age_range_tuple
        self.sex_list = sex_list
        self.collection_name = collection_name
        self.add_punctuation = add_punctuation
        self.exclude_target_child = exclude_target_child
        self.order = order

        print("Loading Documents from ChildesDB file Into df")
        full_utterance_df = pd.read_csv(self.input_path)
        subset_utterance_df = self.subset_documents(full_utterance_df)

        # Create a dictionary to map transcript_id to target_child_age
        transcript_age_map = {row['transcript_id']: row[
            'target_child_age'] for index, row in subset_utterance_df.iterrows()}

        # Convert the dictionary to a list of tuples (transcript_id, target_child_age)
        unique_transcript_age_list = list(transcript_age_map.items())

        # sort the list of tuples by age (the second element)
        sorted_list = self.sort_documents_by_age(unique_transcript_age_list)

        counter = 0
        for transcript_id, age in sorted_list:
            # Filter the DataFrame for rows with the current transcript_id
            filtered_df = subset_utterance_df[subset_utterance_df['transcript_id'] == transcript_id]

            # Extract the desired columns and convert to a list of tuples
            tuples_list = filtered_df[['utterance_order', 'gloss', 'type']].apply(tuple, axis=1).tolist()

            document_info_dict = {}
            document_name = str(round(age, 3)).replace('.', '_') + "_" + str(transcript_id)
            document_info_dict['name'] = document_name
            document_info_dict['age'] = age

            document_sequence_list = []
            for i in range(len(tuples_list)):
                cleaned_text = self.clean_text(tuples_list[i][1])
                utterance_token_list = self.create_utterance_token_list(cleaned_text, tuples_list[i][2])
                document_sequence_list.append(utterance_token_list)

            self.add_document(document_sequence_list, tokenized=True, document_info_dict=document_info_dict)
            counter += 1
            if counter % 100 == 0:
                print(f"    Loaded {counter} documents")


    @staticmethod
    def clean_text(input_text):

        input_text = input_text.lower()

        # Replace + or - with _
        pattern = r'(?<=\S)[+-](?=\S)'
        replaced_text = re.sub(pattern, '_', input_text)

        return replaced_text

    @staticmethod
    def tokenize(text_string):
        token_list = text_string.split()
        return token_list

    @staticmethod
    def get_punctuation(utterance_type):
        punctuation_dict = {'declarative': ".",
                            "question": "?",
                            "trail off": ";",
                            "imperative": "!",
                            "imperative_emphatic": "!",
                            "interruption": ":",
                            "self interruption": ";",
                            "quotation next line": ";",
                            "interruption question": "?",
                            "missing CA terminator": ".",
                            "broken for coding": ".",
                            "trail off question": "?",
                            "quotation precedes": ".",
                            "self interruption question": "?",
                            "question exclamation": "?"}
        return punctuation_dict[utterance_type]

    def create_utterance_token_list(self, gloss, utterance_type):
        token_list = self.tokenize(gloss)
        if self.add_punctuation:
            token_list.append(self.get_punctuation(utterance_type))
        return token_list

    def subset_documents(self, utterance_df):

        utterance_df = utterance_df.dropna(subset=['gloss'])

        if self.language is not None:
            utterance_df = utterance_df[utterance_df['language'] == self.language]
        if self.collection_name is not None:
            utterance_df = utterance_df[utterance_df['collection'] == self.collection_name]
        if self.sex_list is not None:
            utterance_df = utterance_df[utterance_df['target_child_sex'].isin(self.sex_list)]
        
        # TODO TEST THIS
        if self.age_range_tuple is not None:
            utterance_df = utterance_df.dropna(subset=['target_child_age'])
            utterance_df = utterance_df[(utterance_df[
                                             'target_child_age'] >= self.age_range_tuple[0]) & (utterance_df[
                                              'target_child_age'] <= self.age_range_tuple[1])]
        
        if self.exclude_target_child:
            utterance_df = utterance_df[utterance_df["speaker_role"] != "Target_Child"]

        utterance_df = utterance_df.sort_values(by=["target_child_age", "transcript_id", "utterance_order"],
                                                ascending=[True, True, True])

        return utterance_df

    def sort_documents_by_age(self, tuple_list):
        if self.order == "age_ordered":
            sorted_tuple_list = sorted(tuple_list, key=lambda x: x[1])
        elif self.order == 'reverse_age_ordered':
            sorted_tuple_list = sorted(tuple_list, key=lambda x: x[1], reverse=True)
        else:
            sorted_tuple_list = copy.deepcopy(tuple_list)
            random.shuffle(sorted_tuple_list)

        return sorted_tuple_list

    def batch_docs_by_age(self, num_batches=10, order="age_ordered"):
        if order == 'reversed':
            self.document_list.reverse()
        elif order == 'shuffled':
            random.shuffle(self.document_list)
        elif order == 'age_ordered':
            pass
        else:
            raise ValueError(f"Unrecognized corpus document order {order}")

        k, m = divmod(len(self.document_list), num_batches)
        self.document_list = [
            self.document_list[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(num_batches)]
