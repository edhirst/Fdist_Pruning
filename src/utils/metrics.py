def calculate_accuracy(predictions, labels):
    correct = (predictions.argmax(dim=1) == labels).float().sum()
    accuracy = correct / labels.size(0)
    return accuracy.item()

def calculate_loss(loss_function, predictions, labels):
    return loss_function(predictions, labels)

def log_metrics(epoch, accuracy, loss):
    print(f'Epoch: {epoch}, Accuracy: {accuracy:.4f}, Loss: {loss:.4f}')