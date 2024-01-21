import random
import copy
from . import corpus
import itertools


class XAYBZ(corpus.Corpus):

    def __init__(self,
                 num_ab_categories=2,
                 ab_category_size=3,

                 x_category_size=0,
                 y_category_size=3,
                 z_category_size=0,
                 min_x_per_sentence=0,
                 max_x_per_sentence=0,
                 min_y_per_sentence=1,
                 max_y_per_sentence=1,
                 min_z_per_sentence=0,
                 max_z_per_sentence=0,

                 document_organization_rule='all_pairs',
                 document_repetitions=1,
                 document_sequence_rule='massed',

                 sentence_repetitions_per_document=0,
                 sentence_sequence_rule='random',

                 word_order_rule='fixed',
                 include_punctuation=True,

                 random_seed=None
                 ):
        super().__init__()

        '''
        Create documents composed sentences, which are composed of words.
        Words belong to syntactic categories A,B,x,y,z denoting their sentence position
        Sentences are constructed using the following rules:
            S = (XP) + Ai + (YP) + Bi + (ZP) .
            XP = (XP) + x
            YP = (YP) + y
            ZP = (ZP) + z
            and where i refers to specific and linked A and B sub-categories
        Results in sentences of the form:
            [x1, x2, ..., xn, A, y1, y2, ..., yn, B, z1, z2, ..., z3 .]

        All A and B sub-categories are sets of identical size
        A and B words are named with the following conventions:
            First character is always A or B, denoting sentence position
            Second character is a number denoting the sub-category
            Third character is a number denoting which the word's index within the sub-category
            Example: A32
                - occupies the first syntactic spot in the pair (because of the A),
                - belongs to the third sub-category,
                - and is the second word in the third category

        A legal sentence, can pair any words from matching A and B sub-categories EXCEPT for those that
        have the same index within that category.
        In other words:
            A11-B12 and A33-B35 are legal pairs,
            A11-B11 is NOT a legal pair, because the words share the same index within their subcategories
            A11-B22 is NOT a legal pair, because the words belong to different sub-categories



            Example:
                num_AB_categories = 3
                AB_category_size = 4
                x_category_size = 4
                y_category_size = 8
                z_category_size = 4

                AB_category_list = [1,2,3]
                AB_category_dict = {1: [[A11, A12, A13, A14], [B11, B12, B13, B14]],
                                    2: [[A21, A22, A23, A24], [B21, B22, B23, B24]],
                                    3: [[A31, A32, A33, A34], [B31, B32, B33, B34]]}

                included_AB_pair_list = [[A11, B12], [A11, B13], [A11, B14],
                                         [A12, B11], [A12, B13], [A12, B14],
                                         [A13, B11], [A13, B12], [A13, B14],
                                         [A14, B11], [A14, B12], [A14, B13],
                                         [A21, B22], [A21, B23], [A21, B24],
                                         [A22, B21], [A22, B23], [A22, B24],
                                         [A23, B21], [A23, B22], [A23, B24],
                                         [A24, B21], [A24, B22], [A24, B23],
                                         [A31, B32], [A31, B33], [A31, B34],
                                         [A32, B31], [A32, B33], [A32, B34],
                                         [A33, B31], [A33, B32], [A33, B34],
                                         [A34, B31], [A34, B32], [A34, B33]]

                ommitted_AB_pair_list = [[A11, B11], [A12, B12], [A13, B13], [A14, B14],
                                         [A21, B21], [A22, B22], [A23, B23], [A24, B24],
                                         [A31, B31], [A32, B32], [A33, B33], [A34, B34]]

                x_list = [x1, x2, x3, x4]
                y_list = [y1, y2, y3, y4, y5, y6, y7, y8]
                z_list = [z1, z2, z3, z4]

        There are a deterministic number of sentences with unique target pairs, determined jointly by
            category size and num categories and equal to:
            num_unique_target_sentences = target_category_size*(target_category_size-1)*num_target_categories

        These sentences are distributed into documents according to document_organization_rule, with allowed values:
            - "all_pairs": each document contains all possible sentence pairs, resulting in a single base document
            - "one_pair_each_category": each document contains a single sentence from each category, resulting in the 
            number
                of base documents equaling: num_sentences / num_target_categories
            - "single_sentence": each document contains a single sentence, resulting in num_unique_target_sentences base 
            documents
            - "single_category": each document contains all sentences from a single category,
                resulting in num_target_categories base documents

        Within each document, its sentences are repeated sentence_repetitions_per_document number of times, each time 
        with different
            random interveners, according to the rules described below.
        The sentences are organized by one of three rules: massed (which puts all sentences next to each other which 
        share the same A word),
            interleaved (which rotates the sentences by A word), and shuffled (which randomizes the order with regard 
            to A word)

        Each base document is repeated document_repetitions number of times, each time with different random 
        interveners, according
            to the rules described below.

        The documents are ordered according to the document_sequence_rule, which must be either 'massed' or 
        'interleaved'
            - massed: the documents that share the common property (word, category, etc.) are adjacent in the document 
            sequence order
                e.g. [A1,A1,A2,A2,A3,A3]
            - interleaved: the documents that share the common property are distributed in the document sequence order
                e.g. [A1,A2,A3,A1,A2,A3]

        Word order within sentences follows one of three rules:
            - random: completely randomized
            - targets_fixed: targets

        Sentences can either end with puncutation "." or have that omitted
        '''
        self.corpus_name = None

        self.generated_document_list = None
        self.generated_document_group_list = None
        self.generated_num_documents = None

        self.document_organization_rule = document_organization_rule
        self.document_repetitions = document_repetitions
        self.document_sequence_rule = document_sequence_rule

        self.num_ab_categories = num_ab_categories
        self.ab_category_size = ab_category_size
        self.ab_category_dict = None

        self.x_category_size = x_category_size
        self.min_x_per_sentence = min_x_per_sentence
        self.max_x_per_sentence = max_x_per_sentence
        self.x_list = None

        self.y_category_size = y_category_size
        self.min_y_per_sentence = min_y_per_sentence
        self.max_y_per_sentence = max_y_per_sentence
        self.y_list = None

        self.z_category_size = z_category_size
        self.min_z_per_sentence = min_z_per_sentence
        self.max_z_per_sentence = max_z_per_sentence
        self.z_list = None

        self.sentence_repetitions_per_document = sentence_repetitions_per_document
        self.sentence_sequence_rule = sentence_sequence_rule

        self.word_order_rule = word_order_rule
        self.include_punctuation = include_punctuation

        self.random_seed = random_seed
        random.seed(random_seed)

        self.generated_type_list = None
        self.ab_category_dict = None
        self.generated_vocab_index_dict = None
        self.generated_index_vocab_dict = None
        self.generated_vocabulary_size = None

        self.included_ab_pair_list = None
        self.omitted_ab_pair_list = None

        self.word_category_dict = None
        self.target_category_index_dict = None

        self.check_parameters()
        self.create_corpus_name()
        self.create_vocabulary()
        self.create_word_pair_list()

        self.create_documents()

    def create_corpus_name(self):
        self.corpus_name = "Corpus"
        self.corpus_name += "_" + str(self.num_ab_categories)
        self.corpus_name += "_" + str(self.ab_category_size)
        self.corpus_name += "_" + str(self.x_category_size)
        self.corpus_name += "_" + str(self.min_x_per_sentence)
        self.corpus_name += "_" + str(self.max_x_per_sentence)
        self.corpus_name += "_" + str(self.y_category_size)
        self.corpus_name += "_" + str(self.min_y_per_sentence)
        self.corpus_name += "_" + str(self.max_y_per_sentence)
        self.corpus_name += "_" + str(self.z_category_size)
        self.corpus_name += "_" + str(self.min_z_per_sentence)
        self.corpus_name += "_" + str(self.max_z_per_sentence)
        self.corpus_name += "_" + str(self.document_organization_rule)
        self.corpus_name += "_" + str(self.document_repetitions)
        self.corpus_name += "_" + str(self.document_sequence_rule)
        self.corpus_name += "_" + str(self.sentence_repetitions_per_document)
        self.corpus_name += "_" + str(self.sentence_sequence_rule)

        self.corpus_name += "_" + str(self.random_seed)

        ("\nCreating corpus {}".format(self.corpus_name))

    def __repr__(self):
        output_string = "\n{}\n".format(self.corpus_name)
        output_string += "    Vocab Size: {}\n".format(self.generated_vocabulary_size)
        output_string += "    Punctuation: [.]\n"
        output_string += "	  x Category: [{}]\n".format(",".join(self.x_list))
        output_string += "	  y Category: [{}]\n".format(",".join(self.y_list))
        output_string += "	  z Category: [{}]\n".format(",".join(self.z_list))

        for category, members in self.ab_category_dict.items():
            c1 = members[0]
            c2 = members[1]
            output_string += "	  {}: [{}]\n".format("A" + category, ",".join(c1))
            output_string += "	  {}: [{}]\n".format("B" + category, ",".join(c2))

        output_string += "    Documents:\n"
        for i in range(self.num_documents):
            output_string += "        Document:{} Group:{} Len:{}\n".format(i, self.generated_document_group_list[i],
                                                                            len(self.generated_document_list[i]))
            for j in range(len(self.generated_document_list[i])):
                output_string += "            [{}]\n".format(",".join(self.generated_document_list[i][j]))

        output_string += "\n"
        return output_string

    def check_parameters(self):
        if self.num_ab_categories < 1:
            raise Exception("ERROR: num_AB_categories must be between >=1".format())
        if self.ab_category_size < 2:
            raise Exception("ERROR: AB_category_size must be >= 2")
        if self.x_category_size < 0:
            raise Exception("ERROR: x_category_size must be >= 1")
        if self.min_x_per_sentence < 0:
            raise Exception("ERROR: min_x_per_sentence must be >= 0")
        if self.min_x_per_sentence > self.max_x_per_sentence:
            raise Exception("ERROR: min_x_per_sentence must be <= max_x_per_sentence")
        if self.max_x_per_sentence < 0:
            raise Exception("ERROR: max_x_per_sentence must be >= 0")
        if self.y_category_size < 0:
            raise Exception("ERROR: y_category_size must be >= 1")
        if self.min_y_per_sentence < 0:
            raise Exception("ERROR: min_y_per_sentence must be >= 0")
        if self.min_y_per_sentence > self.max_y_per_sentence:
            raise Exception("ERROR: min_y_per_sentence must be <= max_y_per_sentence")
        if self.max_y_per_sentence < 0:
            raise Exception("ERROR: max_y_per_sentence must be >= 0")
        if self.z_category_size < 0:
            raise Exception("ERROR: z_category_size must be >= 1")
        if self.min_z_per_sentence < 0:
            raise Exception("ERROR: min_z_per_sentence must be >= 0")
        if self.min_z_per_sentence > self.max_z_per_sentence:
            raise Exception("ERROR: min_z_per_sentence must be <= max_z_per_sentence")
        if self.max_z_per_sentence < 0:
            raise Exception("ERROR: max_z_per_sentence must be >= 0")
        # if self.sentence_repetitions_per_document < 1:
        # 	raise Exception("ERROR: sentence_repetitions_per_document must be >= 1")
        if self.document_repetitions < 1:
            raise Exception("ERROR: document_repetitions must be >= 1")
        if self.document_organization_rule not in ["all_pairs", "one_pair_each_category", "single_sentence",
                                                   "single_category"]:
            raise Exception("ERROR: Unrecognized document organization rule {}".format(self.document_organization_rule))
        if self.document_sequence_rule not in ["massed", "interleaved", 'random']:
            raise Exception("ERROR: Unrecognized document_sequence_rule {}".format(self.document_sequence_rule))

    @staticmethod
    def create_category_members(size, label):
        member_list = []
        if size > 0:
            for i in range(size):
                member = label + str(i + 1)
                member_list.append(member)
        return member_list

    def add_words_to_vocab(self, word_list):
        if len(word_list) > 0:
            for word in word_list:
                self.generated_vocab_index_dict[word] = self.generated_vocabulary_size
                self.generated_index_vocab_dict[self.generated_vocabulary_size] = word
                self.generated_type_list.append(word)
                self.generated_vocabulary_size += 1

    def create_vocabulary(self):
        self.generated_type_list = []
        self.ab_category_dict = {}
        self.generated_vocab_index_dict = {}
        self.generated_index_vocab_dict = {}
        self.generated_vocabulary_size = 0

        # self.unknown_token = '<unk>'
        self.add_words_to_vocab(['.'])
        for i in range(self.num_ab_categories):
            current_category = str(i + 1)

            category_label = "A" + current_category + "_"
            set1 = self.create_category_members(self.ab_category_size, category_label)
            self.add_words_to_vocab(set1)

            category_label = "B" + current_category + "_"
            set2 = self.create_category_members(self.ab_category_size, category_label)
            self.add_words_to_vocab(set2)

            self.ab_category_dict[current_category] = [set1, set2]

        self.x_list = self.create_category_members(self.x_category_size, "x")
        self.add_words_to_vocab(self.x_list)

        self.y_list = self.create_category_members(self.y_category_size, "y")
        self.add_words_to_vocab(self.y_list)

        self.z_list = self.create_category_members(self.z_category_size, "z")
        self.add_words_to_vocab(self.z_list)

    def create_word_pair_list(self):
        self.included_ab_pair_list = []
        self.omitted_ab_pair_list = []

        for category, data in self.ab_category_dict.items():
            set1 = data[0]
            set2 = data[1]
            for i in range(self.ab_category_size):
                for j in range(self.ab_category_size):
                    if i == j:
                        self.omitted_ab_pair_list.append((set1[i], set2[j]))
                    else:
                        self.included_ab_pair_list.append((set1[i], set2[j]))

    def get_pair_document_group(self, pair, index):
        if self.document_organization_rule == 'all_pairs':
            group = "all"
        elif self.document_organization_rule == 'one_pair_each_category':
            group = str(round(index % (len(self.included_ab_pair_list) / self.num_ab_categories)))
        elif self.document_organization_rule == 'single_sentence':
            group = "_".join(pair)
        elif self.document_organization_rule == 'single_category':
            group = pair[0][1]
        else:
            raise Exception("ERROR: unrecognized document_organization_rule {}".format(self.document_organization_rule))
        return group

    def create_documents(self):
        document_group_dict = {}
        included_ab_pair_list = copy.deepcopy(self.included_ab_pair_list)
        group_list = []
        # group = None
        for i in range(len(included_ab_pair_list)):
            pair = included_ab_pair_list[i]
            group = self.get_pair_document_group(pair, i)
            if group not in document_group_dict:
                document_group_dict[group] = []
                group_list.append(group)
            document_group_dict[group].append(pair)

        # group_size = len(document_group_dict[group])
        num_groups = len(document_group_dict)

        full_document_list = []
        full_document_group_list = []

        if self.document_sequence_rule == "massed":
            for i in range(num_groups):
                group = group_list[i]
                for j in range(self.document_repetitions):
                    document = copy.deepcopy(document_group_dict[group])
                    full_document_list.append(document)
                    full_document_group_list.append(group)

        elif self.document_sequence_rule == "interleaved":
            for i in range(self.document_repetitions):
                for j in range(num_groups):
                    group = group_list[j]
                    document = copy.deepcopy(document_group_dict[group])
                    full_document_list.append(document)
                    full_document_group_list.append(group)

        elif self.document_sequence_rule == "random":
            # TODO impliment this
            for i in range(num_groups):
                group = group_list[i]
                for j in range(self.document_repetitions):
                    document = copy.deepcopy(document_group_dict[group])
                    full_document_list.append(document)
                    full_document_group_list.append(group)
            random.shuffle(full_document_list)

        else:
            raise Exception("ERROR: Unrecognized document_sequence_rule {}".format(self.document_sequence_rule))

        self.generated_document_list = []
        self.generated_document_group_list = []

        # if sentence reps per doc == 0, then we are going to do the number of reps required to do all combos of ys
        # else do a fixed number of reps and choose a random y
        if self.sentence_repetitions_per_document == 0:
            y_lists = []
            for i in range(self.min_y_per_sentence, self.max_y_per_sentence + 1):
                y_lists += list(itertools.product(self.y_list, repeat=i))
        else:
            y_lists = None

        for i in range(len(full_document_list)):
            new_document = []

            current_document_template = copy.deepcopy(full_document_list[i])
            num_sentences = len(current_document_template)

            if self.sentence_repetitions_per_document > 0:
                if self.sentence_sequence_rule == "massed" or self.sentence_sequence_rule == "random":
                    for j in range(num_sentences):
                        current_ab_pair = current_document_template[j]
                        for k in range(self.sentence_repetitions_per_document):
                            sentence = self.create_sentence(current_ab_pair)
                            new_document.append(sentence)

                    if self.sentence_sequence_rule == "random":
                        random.shuffle(new_document)

                elif self.sentence_sequence_rule == "interleaved":
                    for j in range(self.sentence_repetitions_per_document):
                        for k in range(num_sentences):
                            current_ab_pair = current_document_template[k]
                            sentence = self.create_sentence(current_ab_pair)
                            new_document.append(sentence)
                else:
                    raise Exception("ERROR: unrecognized sentence_sequence_rule={}".format(self.sentence_sequence_rule))

                self.generated_document_list.append(new_document)
                self.generated_document_group_list.append(full_document_group_list[i])

            if self.sentence_repetitions_per_document == 0:

                if self.sentence_sequence_rule == "massed" or self.sentence_sequence_rule == "random":
                    for j in range(num_sentences):
                        current_ab_pair = current_document_template[j]
                        for k in range(len(y_lists)):
                            current_y_list = y_lists[k]
                            sentence = self.create_sentence(current_ab_pair, current_y_list)
                            new_document.append(sentence)

                    if self.sentence_sequence_rule == "random":
                        random.shuffle(new_document)

                elif self.sentence_sequence_rule == "interleaved":
                    for j in range(len(y_lists)):
                        current_y_list = y_lists[j]
                        for k in range(num_sentences):
                            current_ab_pair = current_document_template[k]
                            sentence = self.create_sentence(current_ab_pair, current_y_list)
                            new_document.append(sentence)

                self.generated_document_list.append(new_document)
                self.generated_document_group_list.append(full_document_group_list[i])

        self.generated_num_documents = len(self.generated_document_list)

        for document in self.generated_document_list:
            self.add_document(document, tokenized=True, document_info_dict=None)

    def create_sentence(self, ab_pair, current_y_list=None):
        sentence = []
        num_x = random.randint(self.min_x_per_sentence, self.max_x_per_sentence)
        num_y = random.randint(self.min_y_per_sentence, self.max_y_per_sentence)
        num_z = random.randint(self.min_z_per_sentence, self.max_z_per_sentence)
        for i in range(num_x):
            sentence.append(random.choice(self.x_list))

        sentence.append(ab_pair[0])

        if current_y_list is None:
            for i in range(num_y):
                sentence.append(random.choice(self.y_list))
        else:
            for y in current_y_list:
                sentence.append(y)

        sentence.append(ab_pair[1])

        for i in range(num_z):
            sentence.append(random.choice(self.z_list))

        if self.word_order_rule == "random":
            random.shuffle(sentence)

        if self.include_punctuation:
            sentence.append(".")
        return sentence

    def assign_category_index_to_token(self, document_list):

        # list of documents, which are lists of sequences, which are lists of tokens
        # [[[A11, y1, B12, .], [A11, y2, B12, .]]]
        # [[[[], y1, B12, .], [A11, y2, B12, .]]]

        if self.vocab_index_dict is None:
            raise Exception("ERROR: Vocab index dict does not exist")

        self.target_category_index_dict = {'Period': 0,
                                           'Present A': 1,
                                           'Omitted A': 2,
                                           'Legal A': 3,
                                           'Illegal A': 4,
                                           'Present B': 5,
                                           'Omitted B': 6,
                                           'Legal B': 7,
                                           'Illegal B': 8,
                                           'x': 9,
                                           'y': 10,
                                           'z': 11,
                                           'Other': 12}

        target_label_lists = copy.deepcopy(document_list)

        for i, document in enumerate(document_list):
            for j, sequence in enumerate(document):
                A_word = []
                B_word = []
                for token in sequence:
                    if token[0] == 'A':
                        A_word = token.split('_')
                    elif token[0] == 'B':
                        B_word = token.split('_')
                for k, token in enumerate(sequence):
                    token_category_list = []
                    for word, index in self.vocab_index_dict.items():
                        current_word = word.split('_')
                        if current_word[0] == '.':
                            token_category_list.append('Period')
                        elif current_word[0][0] == 'A':
                            if current_word == A_word:
                                token_category_list.append('Present A')
                            elif current_word[0][1:] == B_word[0][1:]:
                                if current_word[1] == B_word[1]:
                                    token_category_list.append('Omitted A')
                                else:
                                    token_category_list.append('Legal A')
                            else:
                                token_category_list.append('Illegal A')
                        elif current_word[0][0] == 'B':
                            if current_word == B_word:
                                token_category_list.append('Present B')
                            else:
                                if current_word[0][1:] == A_word[0][1:]:
                                    if current_word[1] == A_word[1]:
                                        token_category_list.append('Omitted B')
                                    else:
                                        token_category_list.append('Legal B')
                                else:
                                    token_category_list.append('Illegal B')
                        elif current_word[0][0]  == 'x':
                            token_category_list.append('x')
                        elif current_word[0][0] == 'y':
                            token_category_list.append('y')
                        elif current_word[0][0]  == 'z':
                            token_category_list.append('z')
                        else:
                            token_category_list.append('Other')

                        target_label_lists[i][j][k] = token_category_list
        return target_label_lists

    @staticmethod
    def create_word_category_dict(vocab_index_dict):
        word_category_dict = {}
        for word, index in vocab_index_dict.items():
            if word[0] == "A" or word[0] == 'B':
                category = word.split('_')[0]
            elif word == '.':
                category = '.'
            elif word[0] == 'x':
                category = word[0]
            elif word[0] == 'y':
                category = word[0]
            elif word[0] == 'z':
                category = word[0]
            else:
                category = 'OTHER'
            word_category_dict[word] = category
        return word_category_dict


