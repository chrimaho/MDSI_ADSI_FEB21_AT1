# Define function to plot confidence matrix and ROC curve for model evaluation
def plot_cm_roc(reg, X_val, y_val):
    
    """
    Args:
        reg: Fitted classifier.
        X_val: Numpy array of input values in validation data.
        y_val: Numpy array of actual target values in validation data.
    """
    
    # Print confusion matrix (standard and normalized formats)
    fig = plt.figure(figsize=(8,8))
    ax1 = fig.add_subplot(121) 
    ax1.title.set_text("Confusion matrix-Training (normalized)")
    plot_confusion_matrix(reg, X_train, y_train, 
                          cmap=plt.cm.Oranges,
                          colorbar=False,
                          normalize='true',
                          ax=ax1)

    ax2 = fig.add_subplot(122)
    ax2.title.set_text("Confusion matrix-Validation (normalized)")
    plot_confusion_matrix(reg, X_val, y_val,
                          cmap=plt.cm.Blues,
                          colorbar=False,
                          normalize='true',
                          ax=ax2)

    plt.subplots_adjust(wspace=0.5)
    plt.show() 
    
    # Plot ROC curve
    plot_roc_curve(reg, X_val, y_val)
    plt.plot([0, 1], [0, 1],'r--')
    plt.title("Receiver Operating Characteristic")
    plt.show()
    
    return None 