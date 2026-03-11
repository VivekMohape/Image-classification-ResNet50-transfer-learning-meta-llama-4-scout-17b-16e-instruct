import matplotlib.pyplot as plt

def plot_training(train_loss,val_loss):

    plt.plot(train_loss,label="Train")
    plt.plot(val_loss,label="Validation")

    plt.legend()
    plt.show()
