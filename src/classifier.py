## Classification Model
class Classifier(torch.nn.Module):
    def __init__(self, n_feature, n_output):
        super(Classifier, self).__init__()
        self.layer = torch.nn.Linear(n_feature, n_output)   # single layer

    def forward(self, x):
        x = F.relu(self.layer(x))      # activation function for hidden layer
        return x

## For each model in the list train and test classification algorithm
def classifier_performance( list_models , train_loader , test_loader ):
    accuracy_list = []
    num_models = 1
    for model in list_models:

        net = Classifier(n_feature=8*num_models, n_output=10)     # define the network
        optimizer = torch.optim.SGD(net.parameters(),  lr=0.001, momentum=0.9 )
        loss_func = torch.nn.CrossEntropyLoss()  # the target label is NOT an one-hotted

        # Train classifier
        for epoch in range(20):  # loop over the dataset multiple times

            running_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                # get the inputs
                inputs, labels = data

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(model.forward(inputs)[1]) #encode by mu
                loss = loss_func(outputs, labels)
                loss.backward()
                optimizer.step()

        # Test Classifier
        correct = 0
        total = 0
        with torch.no_grad():
            for data in test_loader:
                images, labels = data
                outputs = net(model.forward(images)[1]) #encode by mu
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total

        #Store accuracy
        accuracy_list.append(accuracy)
        num_models += 1

    return np.asarray( accuracy_list )
