import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm_notebook as tqdm


#new 20191205 which accepts a list of valid clips (those without a target in the background of a non-target clip)
def get_weighted_sampler(dataset, subset, replacement: bool=True):
    df = dataset._df.iloc[dataset._valid_indices].reset_index()
    value_counts_dict = df.target.value_counts().to_dict()
    value_counts = df.target.map(value_counts_dict)
    sampler_weights = 1. / value_counts
    sampler_weights = sampler_weights.to_numpy()
    sampler_weights = sampler_weights[subset.indices]
    return torch.utils.data.sampler.WeightedRandomSampler(weights=sampler_weights, num_samples=sampler_weights.size, replacement=replacement)


def get_training_set_generator(dataset, training_set, batch_size, stratified, replacement: bool=True):
    if stratified:
        sampler = get_weighted_sampler(dataset, training_set, replacement)
        return torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=False, sampler=sampler, drop_last=True)
    else:
        return torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=True, drop_last=True)

def get_validation_set_generator(validation_set, batch_size):
    return torch.utils.data.DataLoader(validation_set, batch_size=batch_size, shuffle=False, drop_last=False)

def loss_batch(model, loss_func, local_batch, local_labels, offset, opt=None):
    logits = model(local_batch)
    
    #Chris Scott: I've tried training where I max-pool the logits into 5 bins and only take the centre 
    #bin as the prediction for the clip and it seems to works fine. With 5 bins and 2 second clips this means 
    #the final prediction is the max score in a region of 0.4 seconds centred on the clip label. This is wide 
    #enough to cover the duration of short flight calls (or the central body of longer calls) but narrow enough 
    #to filter out nearby calls. Keeping the clip length at 2 seconds might seem redundant with this setup but 
    #because the background noise is estimated from each clip it still makes sense to keep a reasonable clip length. 
    adapool_output_size = 5 # pooling 2 second clip gives 5x 0.4 second bins
    offset = adapool_output_size // 2 # fixed offset replaces previous offset that changed during training
    adapool = torch.nn.AdaptiveMaxPool1d(adapool_output_size)
    logits = model(local_batch)
    logits_max = adapool(logits)[:,:,offset:-offset]
    logits_max, _ = torch.max(logits_max, -1, keepdim=False)

    loss = loss_func(logits_max, local_labels) #20200109 update to only calculate single loss in validation

    if opt is not None:
        logits_kappa = model(local_batch) #20200109 update to calculate second loss in training only
        logit_pairing_loss = (logits-logits_kappa).pow(2).mean()
        loss = loss + logit_pairing_loss
    
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        opt.zero_grad()

    return loss.item()

def one_hot(local_labels, n_classes=2):
    m = torch.eye(n_classes)
    targets = m[local_labels.long()][:,1:] # drop first column (local_labels=0)
    return targets


def fit(epochs, model, optimizer, training_generator, validation_generator, device, checkpoint_dict):

    # define loss function
    # BCEWithLogitsLoss expects unnormalised values and combines a sigmoid layer and a BCELoss (Binary Cross Entropy) in one single class 
    # see: https://pytorch.org/docs/stable/nn.html#bcewithlogitsloss
    loss_func = torch.nn.BCEWithLogitsLoss() 

    offset = 6
    previous_epochs = checkpoint_dict['epoch']
    print('training model...')
    for epoch in tqdm(range(epochs)):
        # training loop
        epoch_loss = []
        nums = 0
        model.train()
        for local_batch, local_labels in training_generator:
            local_labels = one_hot(local_labels, checkpoint_dict['n_classes'])
            local_batch, local_labels = local_batch.to(device), local_labels.to(device)
            loss = loss_batch(model, loss_func, local_batch, local_labels, offset, opt=optimizer)
            epoch_loss.append(loss * len(local_batch))
            nums += len(local_batch)
        training_loss = np.sum(epoch_loss) / nums
        checkpoint_dict['training_loss'].append(training_loss)
        
        # validation loop
        epoch_loss = []
        nums = 0
        model.eval()
        with torch.no_grad():
            for local_batch, local_labels in validation_generator:
                local_labels = one_hot(local_labels, checkpoint_dict['n_classes'])
                local_batch, local_labels = local_batch.to(device), local_labels.to(device)
                loss = loss_batch(model, loss_func, local_batch, local_labels, 1, opt=None)
                epoch_loss.append(loss * len(local_batch))
                nums += len(local_batch)
            validation_loss = np.sum(epoch_loss) / nums
            checkpoint_dict['validation_loss'].append(validation_loss)
        
        # save the model if the validation error is lower than previously recorded
        if validation_loss < checkpoint_dict['loss']:
            checkpoint_dict['epoch'] = previous_epochs + epoch + 1
            checkpoint_dict['loss'] = validation_loss
            checkpoint_dict['model_state_dict'] = model.state_dict()
            checkpoint_dict['optimizer_state_dict'] = optimizer.state_dict()
            torch.save(checkpoint_dict, checkpoint_dict['filename'])

        print('Epoch {}, training loss: {}, validation loss: {}'.format(epoch, training_loss, validation_loss))

        if (epoch%2==0):
            offset = max(offset-1, 1)

def plot_precision_recall(model, validation_generator, device, channel):
    from sklearn.metrics import precision_recall_curve
    from sklearn.metrics import roc_auc_score
    
    y_test, y_score = [], []
    with torch.no_grad():
        for local_batch, local_labels in validation_generator:
            logits = model(local_batch.to(device))
            #print(logits.shape)
            scores = logits.sigmoid().cpu().detach().numpy()[:,channel,:]
            #print(scores.shape)
            max_scores = np.max(scores, -1, keepdims=False)
            #y_test.append(local_labels.numpy())
            y_test.append(np.where(local_labels.numpy()==(channel+1), 1, 0))
            y_score.append(max_scores)
    y_test = np.hstack(y_test)
    y_score = np.hstack(y_score)
    
    auc = roc_auc_score(y_test, y_score)
    print('AUC = ' + str(auc))
    
    precision, recall, thresholds = precision_recall_curve(y_test, y_score)
    thresholds = np.append(thresholds, 1.)
    
    plt.plot(thresholds, precision, label='precision')
    plt.plot(thresholds, recall, label='recall')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0,1.0])
    plt.xlabel('Score Threshold')
    plt.legend(loc="lower left")
    plt.title('Precision and Recall. AUC=' + str(round(auc,3)) + ' (channel=' + str(channel) + ')')
    plt.show()

def plot_validation_predictions(model, validation_set, device):
    for i in range(len(validation_set)):
        x,y = validation_set[i]
        print(y)
        model.visualise(torch.from_numpy(x[None,:]).to(device))

def get_and_print_device(use_cuda: bool=True):
    if use_cuda:
        # print pytorch version and use cuda if available
        print('PyTorch version: ', torch.__version__)
        use_cuda = torch.cuda.is_available()
        print('CUDA available: ', use_cuda)
        device = torch.device("cuda:0" if use_cuda else "cpu")
        if device.type == 'cuda':
            print(torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
    return device

def plot_losses(training_loss, validation_loss):
    # plot training and validation losses returned from fit method
    plt.plot(training_loss, label='training')
    plt.plot(validation_loss, label='validation')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.show()

def save_losses(training_loss, validation_loss, csvname: str='losses'):
    #save the training and validation losses returned from fit method
    losses = pd.DataFrame()
    losses['training'] = training_loss
    losses['validation'] = validation_loss
    losses.to_csv(csvname, sep=',', header = True, index = True)

#updated 20191209 to print filename and offset of clips
def plot_top_losses(model, validation_set, device, checkpoint_dict, k=10):
    loss_func = torch.nn.BCEWithLogitsLoss()
   
    model.eval()
    with torch.no_grad():
        N = len(validation_set)
        losses = np.zeros(N)
        for i in range(N):
            x, y = validation_set[i]
            local_batch = torch.from_numpy(x[None,:])
            local_labels = one_hot(torch.from_numpy(np.asarray(y, dtype=np.int32)).view(1), checkpoint_dict['n_classes'])
            local_batch, local_labels = local_batch.to(device), local_labels.to(device)
            loss = loss_batch(model, loss_func, local_batch, local_labels, 1, opt=None)
            losses[i] = loss
        k = min(k, N)
        indices = np.argpartition(losses, -k)[-k:]
        for index in indices:
            dataset_index = validation_set.dataset._valid_indices[validation_set.indices[index]]
            filename = validation_set.dataset._df.iloc[dataset_index]['filename']
            offset = validation_set.dataset._df.iloc[dataset_index]['offset']
            x, y = validation_set[index]
            print('filename: {}, offset: {}, target: {}'.format(filename, offset, y))
            model.visualise(torch.from_numpy(x[None,:]).to(device))
            
print('loaded train and fit module')