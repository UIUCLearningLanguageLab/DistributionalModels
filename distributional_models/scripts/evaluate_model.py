from distributional_models.tasks.cohyponyms2 import Cohyponyms
from distributional_models.tasks.classifier import Classifier
from distributional_models.tasks.generate import generate_sequence


def evaluate_model(label, model, the_categories, corpus, train_params, training_took, loss_mean):

    output_string = f"{label:8}  loss:{loss_mean:<7.4f}"
    # TODO this doesnt implement evaluation using hidden states
    weight_matrix = model.get_weights(train_params['evaluation_layer'])

    the_categories.set_instance_feature_matrix(weight_matrix, corpus.vocab_index_dict)

    if train_params['run_cohyponym_task']:
        the_cohyponym_task = Cohyponyms(the_categories,
                                        num_thresholds=train_params['cohyponym_num_thresholds'],
                                        similarity_metric=train_params['cohyponym_similarity_metric'],
                                        only_best_threshold=train_params['cohyponym_only_best_thresholds'])

        output_string += f" BA:{the_cohyponym_task.balanced_accuracy_mean:0.3f}-R:{the_cohyponym_task.correlation:0.3f}"
        ba_took = the_cohyponym_task.took
    else:
        ba_took = 0

    if train_params['run_classifier_task']:
        the_classifier = Classifier(the_categories, train_params)

        output_string += f"  Classify0:{the_classifier.train_means[0]:0.3f}-{the_classifier.test_means[0]:0.3f}"
        output_string += f"  Classify1:{the_classifier.train_means[1]:0.3f}-{the_classifier.test_means[1]:0.3f}"
        output_string += f"  ClassifyN:{the_classifier.train_means[-1]:0.3f}-{the_classifier.test_means[-1]:0.3f}"
        classifier_took = the_classifier.took
    else:
        classifier_took = 0

    if train_params['generate_sequence']:
        generated_sequence = generate_sequence(model,
                                               corpus,
                                               train_params['prime_token_list'],
                                               train_params['generate_sequence_length'],
                                               train_params['generate_temperature'])
        output_string += f'   "{generated_sequence}"'

    output_string += f"  Took:{training_took:0.2f}-{ba_took:0.2f}-{classifier_took:0.2f}"
    return output_string
