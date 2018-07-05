from regression_analysis.linear import split_data

from numpy import *
from math import *


def generate_labelled_data(length_of_data=None, number_of_independent_vars=None):
    return c_[random.rand(length_of_data, number_of_independent_vars), random.randint(2, size=length_of_data)]


def sigmoid(x):
    return 1/(1+exp(-x))


def logistic_gradients(weights=None, dict_data_=None, intercept=None):
    weighted_gradient = zeros(len(weights))
    intercept_gradient = 0

    for k in range(len(dict_data_['train_x'])):
        weighted_gradient = weighted_gradient + (sigmoid(dot(weights, dict_data_['train_x'][k, :])+intercept) -
                                                 dict_data_['train_y'][k])*dict_data_['train_x'][k, :]
        intercept_gradient = intercept_gradient + (sigmoid(dot(weights, dict_data_['train_x'][k, :])+intercept) -
                                                   dict_data_['train_y'][k])

    return {'weighted_gradient': weighted_gradient.astype('float')/len(dict_data_['train_y']),
            'biased_gradient': float(intercept_gradient)/len(dict_data_['train_y'])}


def logistic_gradient_descent(weights=None, dict_data_=None, intercept=None, learning_rate=None, weight_tolerance=None,
                              intercept_tolerance=None):
    assert 0 < learning_rate < 1, 'Learning rate value not valid!'

    while True:
        dict_ = logistic_gradients(weights=weights, dict_data_=dict_data_, intercept=intercept)
        weights_updated = weights - learning_rate * dict_['weighted_gradient']
        intercept_updated = intercept - learning_rate * dict_['biased_gradient']
        weight_diff_ = abs(weights_updated - weights)
        intercept_diff_ = abs(intercept_updated - intercept)

        if len(weight_diff_[weight_diff_ > weight_tolerance]) or abs(intercept_diff_ > intercept_tolerance):
            weights = weights_updated
            intercept = intercept_updated
        else:
            break

    return {'weights': weights_updated, 'intercept': intercept_updated}


def estimate_labels(dict_params=None, dict_splited=None, key_=None):
    predict_ = array([dot(dict_params['weights'], dict_splited[key_][k, :])+dict_params['intercept'] for k in
                      range(len(dict_splited[key_]))])
    return array([1 if sigmoid(pred_) >= 0.5 else 0 for pred_ in predict_])


def training_labels(dict_params=None, dict_splited=None, key_=None):
    dict_ = dict()
    dict_['training_actual_y'] = dict_splited['train_y']
    dict_['training_predicted_y'] = estimate_labels(dict_params=dict_params, dict_splited=dict_splited, key_=key_)

    return dict_


def testing_labels(dict_params=None, dict_splited=None, key_=None):
    dict_ = dict()
    dict_['testing_actual_y'] = dict_splited['test_y']
    dict_['testing_predicted_y'] = estimate_labels(dict_params=dict_params, dict_splited=dict_splited, key_=key_)

    return dict_


def calculate_parameters(dict_=None, actual_key=None, predicted_key=None):
    dict_param_ = dict()
    true_ = len([dict_[actual_key][k] for k in range(len(dict_[actual_key])) if dict_[actual_key][k] ==
                 dict_[predicted_key][k]])
    true_positive = len([dict_[actual_key][k] for k in range(len(dict_[actual_key])) if dict_[actual_key][k] == 1 and
                         dict_[predicted_key][k] == 1])
    false_positive = len([dict_[actual_key][k] for k in range(len(dict_[actual_key])) if dict_[actual_key][k] == 0 and
                         dict_[predicted_key][k] == 1])
    false_negative = len([dict_[actual_key][k] for k in range(len(dict_[actual_key])) if dict_[actual_key][k] == 1 and
                         dict_[predicted_key][k] == 0])

    dict_param_['accuracy'] = float(true_)/float(len(dict_[actual_key]))
    dict_param_['precision'] = float(true_positive)/float(true_positive+false_positive)
    dict_param_['recall'] = float(true_positive)/float(true_positive+false_negative)
    dict_param_['f1_score'] = 2*((dict_param_['precision']*dict_param_['recall'])/(dict_param_['precision'] +
                                                                                  dict_param_['recall']))

    return dict_param_


def training_metrics(dict_=None, actual_key=None, predicted_key=None):
    return calculate_parameters(dict_=dict_, actual_key=actual_key, predicted_key=predicted_key)


def testing_metrics(dict_=None, actual_key=None, predicted_key=None):
    return calculate_parameters(dict_=dict_, actual_key=actual_key, predicted_key=predicted_key)


def logistic_regression_main(length_of_data=None, number_of_independent_vars=None, split_ratio=None, weights=None,
                             intercept=None, learning_rate=None, weight_tolerance=None, intercept_tolerance=None):
    data = generate_labelled_data(length_of_data=length_of_data, number_of_independent_vars=number_of_independent_vars)
    dict_splited = split_data(data_=data, split_ratio=split_ratio)
    params_dict = logistic_gradient_descent(weights=weights, dict_data_=dict_splited, intercept=intercept,
                                            learning_rate=learning_rate, weight_tolerance=weight_tolerance,
                                            intercept_tolerance=intercept_tolerance)

    dict_training = training_labels(dict_params=params_dict, dict_splited=dict_splited, key_='train_x')
    dict_testing = testing_labels(dict_params=params_dict, dict_splited=dict_splited, key_='test_x')
    training_metrics_dict = training_metrics(dict_=dict_training, actual_key='training_actual_y',
                                             predicted_key='training_predicted_y')
    testing_metrics_dict = testing_metrics(dict_=dict_testing, actual_key='testing_actual_y',
                                           predicted_key='testing_predicted_y')

    return training_metrics_dict, testing_metrics_dict
