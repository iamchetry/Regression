from regression_analysis.linear import linear_regression_main
from regression_analysis.logistic import logistic_regression_main


print linear_regression_main(length_of_data=100000, number_of_independent_vars=3, split_ratio=0.75, weights=[1, 1, 1],
                             intercept=0, learning_rate=0.001, weight_tolerance=10**-4, intercept_tolerance=10**-5,
                             actual_weights=[0.83, -0.91, 1.4], actual_intercept=0.2345)


print logistic_regression_main(length_of_data=100000, number_of_independent_vars=3, split_ratio=0.75, weights=[1, 1, 1],
                               intercept=0, learning_rate=0.001, weight_tolerance=10**-4, intercept_tolerance=10**-5,
                               actual_weights=[1.2, 0.79, -0.95], actual_intercept=0.123)
