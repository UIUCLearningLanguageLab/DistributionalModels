from distributional_models.tasks.cohyponyms2 import Cohyponyms
from distributional_models.tasks.classifier import Classifier
from distributional_models.tasks.generate import generate_sequence
from distributional_models.tasks.sequence_predictions import SequencePredictions


def evaluate_model(label, model, corpus, train_params, training_took, loss_mean, the_categories=None, test_sequence_list=None):

    evaluation_dict = {}

    output_string = f"{label:8}  loss:{loss_mean:<7.4f}"
    took_string = f"  Took:{training_took:0.2f}"

    # TODO this doesnt implement evaluation using hidden states
    weight_matrix = model.get_weights(train_params['evaluation_layer'])

    if the_categories is not None:
        the_categories.set_instance_feature_matrix(weight_matrix, corpus.vocab_index_dict)

        if 'run_cohyponym_task' in train_params:
            if train_params['run_cohyponym_task']:
                the_cohyponym_task = Cohyponyms(the_categories,
                                                num_thresholds=train_params['cohyponym_num_thresholds'],
                                                similarity_metric=train_params['cohyponym_similarity_metric'],
                                                only_best_threshold=train_params['cohyponym_only_best_thresholds'])
                evaluation_dict['cohyponyms'] = the_cohyponym_task
                output_string += f" BA:{the_cohyponym_task.balanced_accuracy_mean:0.3f}-R:{the_cohyponym_task.correlation:0.3f}"
                took_string += f"-{the_cohyponym_task.took:0.2f}"

        if 'run_classifier_task' in train_params:
            if train_params['run_classifier_task']:
                the_classifier = Classifier(the_categories, train_params)
                evaluation_dict['classifier'] = the_classifier
                output_string += f"  Classify0:{the_classifier.train_means[0]:0.3f}-{the_classifier.test_means[0]:0.3f}"
                output_string += f"  Classify1:{the_classifier.train_means[1]:0.3f}-{the_classifier.test_means[1]:0.3f}"
                output_string += f"  ClassifyN:{the_classifier.train_means[-1]:0.3f}-{the_classifier.test_means[-1]:0.3f}"
                took_string += f"-{the_classifier.took:0.2f}"

    if 'predict_sequences' in train_params:
        if train_params['predict_sequences']:
            sequence_target_label_list = corpus.assign_category_index_to_token(test_sequence_list)
            print("corpus target_category_index dict", corpus.target_category_index_dict)
            the_sequence_predictions = SequencePredictions(model,
                                                           train_params,
                                                           test_sequence_list,
                                                           sequence_target_label_list=sequence_target_label_list,
                                                           token_category_dict=corpus.word_category_dict,
                                                           target_category_index_dict=corpus.target_category_index_dict)
            evaluation_dict['sequence_predictions'] = the_sequence_predictions
            output_string += f" SeqPred:{the_sequence_predictions.sequence_prediction_accuracy_mean:0.3f}"
            took_string += f"-{the_sequence_predictions.took:0.2f}"

    if train_params['generate_sequence']:
        generated_sequence = generate_sequence(model,
                                               corpus,
                                               train_params['prime_token_list'],
                                               train_params['generate_sequence_length'],
                                               train_params['generate_temperature'])
        output_string += f'   "{generated_sequence}"'

    evaluation_dict['output_string'] = output_string + took_string

    return evaluation_dict
