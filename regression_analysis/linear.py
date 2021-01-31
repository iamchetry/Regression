from numpy import *


def define_data(length_of_data=None, number_of_independent_vars=None):
    return random.rand(length_of_data, number_of_independent_vars)


def define_response(data_=None, actual_weights=None, actual_intercept=None):
    return c_[data_, matmul(data_, actual_weights)+actual_intercept]


def split_data(data_=None, split_ratio=None):
    assert 0 < split_ratio < 1, 'Split Ratio value not allowed!'

    train_x = data_[:int(round(len(data_)*split_ratio, 0)), :-1]
    train_y = data_[:int(round(len(data_)*split_ratio, 0)), -1]
    test_x = data_[int(round(len(data_)*split_ratio, 0)):, :-1]
    test_y = data_[int(round(len(data_)*split_ratio, 0)):, -1]

    return {'train_x': train_x, 'train_y': train_y, 'test_x': test_x, 'test_y': test_y}


def obtain_gradient(weights=None, dict_data_=None, intercept=None):
    residual_vec_ = array([dict_data_['train_y'][k]-(dot(weights, dict_data_['train_x'][k, :])+intercept)
                           for k in range(len(dict_data_['train_y']))])
    weighted_gradient = -2*matmul(dict_data_['train_x'].T, residual_vec_)
    intercept_gradient = -2*sum(residual_vec_)

    return {'weighted_gradient': weighted_gradient.astype('float')/len(dict_data_['train_y']),
            'biased_gradient': float(intercept_gradient)/len(dict_data_['train_y'])}


def gradient_descent(weights=None, dict_data_=None, intercept=None, learning_rate=None, weight_tolerance=None,
                     intercept_tolerance=None):
    assert 0 < learning_rate < 1, 'Learning rate value not valid!'

    while True:
        dict_ = obtain_gradient(weights=weights, dict_data_=dict_data_, intercept=intercept)
        weights_updated = weights - learning_rate*dict_['weighted_gradient']
        intercept_updated = intercept - learning_rate*dict_['biased_gradient']
        weight_diff_ = abs(weights_updated - weights)
        intercept_diff_ = abs(intercept_updated - intercept)

        if len(weight_diff_[weight_diff_ > weight_tolerance]) > 0 or intercept_diff_ > intercept_tolerance:
            weights = weights_updated
            intercept = intercept_updated
        else:
            break

    return {'weights': weights_updated, 'intercept': intercept_updated}


def training_prediction(dict_splited=None, dict_params=None):
    dict_ = dict()
    dict_['training_actual_y'] = dict_splited['train_y']
    dict_['training_predicted_y'] = array([dot(dict_params['weights'], dict_splited['train_x'][k, :]) +
                                          dict_params['intercept'] for k in range(len(dict_splited['train_x']))])

    return dict_


def testing_prediction(dict_splited=None, dict_params=None):
    dict_ = dict()
    dict_['testing_actual_y'] = dict_splited['test_y']
    dict_['testing_predicted_y'] = array([dot(dict_params['weights'], dict_splited['test_x'][k, :]) +
                                          dict_params['intercept'] for k in range(len(dict_splited['test_x']))])

    return dict_


def obtain_loss(dict_training=None, dict_testing=None):
    dict_ = dict()
    dict_['training_error'] = dot(dict_training['training_actual_y']-dict_training['training_predicted_y'],
                                  dict_training['training_actual_y'] - dict_training['training_predicted_y'])/len(
        dict_training['training_actual_y'])
    dict_['testing_error'] = dot(dict_testing['testing_actual_y']-dict_testing['testing_predicted_y'],
                                 dict_testing['testing_actual_y'] - dict_testing['testing_predicted_y'])/len(
        dict_testing['testing_actual_y'])

    return dict_


def linear_regression_main(length_of_data=None, number_of_independent_vars=None, split_ratio=None, weights=None,
                           intercept=None, learning_rate=None, weight_tolerance=None, intercept_tolerance=None,
                           actual_weights=None, actual_intercept=None):
    data = define_data(length_of_data=length_of_data, number_of_independent_vars=number_of_independent_vars)
    data = define_response(data_=data, actual_weights=actual_weights, actual_intercept=actual_intercept)
    dict_splited = split_data(data_=data, split_ratio=split_ratio)
    params_dict = gradient_descent(weights=weights, dict_data_=dict_splited, intercept=intercept,
                                   learning_rate=learning_rate, weight_tolerance=weight_tolerance,
                                   intercept_tolerance=intercept_tolerance)

    training_dict = training_prediction(dict_splited=dict_splited, dict_params=params_dict)
    testing_dict = testing_prediction(dict_splited=dict_splited, dict_params=params_dict)
    return obtain_loss(dict_training=training_dict, dict_testing=testing_dict)
