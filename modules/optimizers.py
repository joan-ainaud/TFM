import numpy as np

def optimizer_gds(parameters, loss, gradient, Nepochs=250, lr=1e-2, epochs_print = 50, hyper={}, return_gradients=True):
    loss_list = np.empty((Nepochs,))
    if return_gradients: grad_list = np.empty((Nepochs,len(parameters)))
    for epoch in range(Nepochs):
        grad_val = gradient(parameters)
        parameters = parameters - lr * grad_val
        
        loss_val = loss(parameters)
        if not(epochs_print is None):
            if epoch % epochs_print == 0 or epoch == Nepochs - 1:
                print(f"Epoch {epoch}, Loss: {loss_val}")

        loss_list[epoch] = loss_val
        if return_gradients: grad_list[epoch] = grad_val
    
    if return_gradients: return parameters, loss_list, grad_list
    return parameters, loss_list