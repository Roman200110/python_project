import torch

def save_checkpoint(state, filename="checkpoint.pth.tar"):
    """
    Saves the model checkpoint.

    Parameters
    ----------
    state : dict
        state dictionary containing epoch, model state, and optimizer state
    filename : str
        path to save the checkpoint file
    """
    torch.save(state, filename)

def load_checkpoint(checkpoint, model, optimizer):
    """
    Loads the model checkpoint.

    Parameters
    ----------
    checkpoint : str
        path to the checkpoint file
    model : torch.nn.Module
        the model to load the state into
    optimizer : torch.optim.Optimizer
        the optimizer to load the state into
    """
    checkpoint = torch.load(checkpoint, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    return epoch
