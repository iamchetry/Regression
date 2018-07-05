from regression_analysis.linear import linear_regression_main
from regression_analysis.logistic import logistic_regression_main


print linear_regression_main(length_of_data=50000, number_of_independent_vars=3, split_ratio=0.75, weights=[1, 1, 1],
                             intercept=0, learning_rate=0.001, weight_tolerance=10**-3, intercept_tolerance=10**-3)


print logistic_regression_main(length_of_data=50000, number_of_independent_vars=3, split_ratio=0.75, weights=[1, 1, 1],
                               intercept=0, learning_rate=0.001, weight_tolerance=10**-3, intercept_tolerance=10**-3)
