import numpy as np

def held_out_validation_set(X, Y, ratio, make_model,  error_fun):
    cutoff = int(np.size(X, 0)*(1-ratio))
    train_X = np.copy(X[:cutoff])
    train_Y = np.copy(Y[:cutoff])
    model = make_model(train_X, train_Y)
    val_X =  np.copy(X[cutoff:])
    val_Y =  np.copy(Y[cutoff:])
    return (error_fun(model, train_X, train_Y), error_fun(model, val_X, val_Y))

def k_fold_cross_validation(X, Y, k, make_model, model_params, error_fun):
    num_points = np.size(X, 0)
    slices = np.linspace(0, num_points, num = k+1, dtype = int)
    train_err = 0
    validation_error = 0
    for i in range(k):
        train_X = np.concatenate((X[:slices[i]], X[slices[i+1]:]))
        train_Y = np.concatenate((Y[:slices[i]], Y[slices[i+1]:]))
        model = make_model(train_X, train_Y, model_params)
        train_err += error_fun(model, train_X, train_Y)
        val_X = X[slices[i]:slices[i+1]]
        val_Y = Y[slices[i]:slices[i+1]]
        validation_error += error_fun(model, val_X, val_Y)
    return (train_err/k, validation_error/k)

#A = np.array([[1,2], [3,4], [5,6], [7,8]])
# B = np.array([[9], [10], [12], [13]])
# k_fold_cross_validation(A, B, 3 , print, print)
# print(A)