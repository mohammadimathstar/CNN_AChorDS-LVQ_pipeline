import torch
from torch import nn, Tensor
import argparse
from typing import Callable
from sklearn.metrics import confusion_matrix, accuracy_score  # f1_score
import yaml


def metrics(y_true: Tensor, y_pred: Tensor, nclasses):

    assert y_true.shape == y_pred.shape, f'their shape is labels: {y_true.shape}, pred:{y_pred.shape}'

    acc = accuracy_score(y_true.cpu().numpy(), y_pred.cpu().numpy())
    c = confusion_matrix(
        y_true.cpu().numpy(), y_pred.cpu().numpy(),
        labels=range(nclasses),
        # normalize='true'
    )
    return 100 * acc, c


def compute_classification_metrics(y_true: Tensor, y_pred: Tensor, nclasses: int) -> tuple:
    """
    Compute classification accuracy and confusion matrix.

    Parameters:
    -----------
    y_true : Tensor
        True class labels, a tensor of shape (num_samples,).
    y_pred : Tensor
        Predicted class labels, a tensor of shape (num_samples,).
    nclasses : int
        Number of classes.

    Returns:
    --------
    tuple
        A tuple containing:
        - accuracy (float): Classification accuracy in percentage.
        - confusion_matrix (ndarray): Confusion matrix of shape (nclasses, nclasses).

    Raises:
    -------
    AssertionError
        If the shapes of `y_true` and `y_pred` do not match.
    """
    assert y_true.shape == y_pred.shape, f'Shape mismatch: labels: {y_true.shape}, predictions: {y_pred.shape}'

    # Convert tensors to numpy arrays
    y_true_np = y_true.cpu().numpy()
    y_pred_np = y_pred.cpu().numpy()

    # Compute accuracy
    accuracy = accuracy_score(y_true_np, y_pred_np)

    # Compute confusion matrix
    conf_matrix = confusion_matrix(y_true_np, y_pred_np, labels=range(nclasses))

    return 100 * accuracy, conf_matrix



def winner_prototype_indices(
        ydata: Tensor,
        yprotos_mat: Tensor,
        distances: Tensor
) -> Tensor:
    """
    Find the closest prototypes to a batch of features.

    Parameters:
    -----------
    ydata : Tensor
        Labels of input images, shape (batch_size,).
    yprotos_mat : Tensor
        Labels of prototypes, shape (nclass, number_of_prototypes).
        This can be used for prototypes with the same or different labels (W^+ and W^-).
    distances : Tensor
        Distances between images and prototypes, shape (batch_size, number_of_prototypes).

    Returns:
    --------
    Tensor
        A tensor containing the indices of the winner prototypes for each image in the batch.
    """
    assert distances.ndim == 2, (f"Distances should be a matrix of shape (batch_size, number_of_prototypes), "
                                 f"but got {distances.shape}")

    # Generate a mask for the prototypes corresponding to each image's label
    mask = yprotos_mat[ydata]
    # Y = yprotos_mat[ydata.tolist()]

    # Apply the mask to distances
    distances_sparse = distances * mask
    # distances_sparse = distances * Y

    # Find the index of the closest prototype for each image
    winner_indices = torch.stack(
        [
            torch.argwhere(w).T[0,
            torch.argmin(
                w[torch.argwhere(w).T],
            )
            ] for w in torch.unbind(distances_sparse)
        ], dim=0
    ).T

    return winner_indices


def winner_prototype_distances(
        ydata: Tensor,
        yprotos_matrix: Tensor,
        yprotos_comp_matrix: Tensor,
        distances: Tensor
) -> tuple:
    """
    Find the distances between winner prototypes and data.

    Parameters:
    -----------
    ydata : Tensor
        Labels of data, shape (nbatch,).
    yprotos_matrix : Tensor
        Matrix containing non-zero elements in the c-th row for prototypes with label 'c',
        shape (nclass, nprotos).
    yprotos_comp_matrix : Tensor
        Matrix containing non-zero elements in the c-th row for prototypes not with label 'c',
        shape (nclass, nprotos).
    distances : Tensor
        Distances between data and prototypes, shape (nbatch, nprotos).

    Returns:
    --------
    tuple
        A tuple containing:
        - Dplus (Tensor): Distance matrix of winner prototypes, shape (nbatch, nprotos).
        - Dminus (Tensor): Distance matrix of non-winner prototypes, shape (nbatch, nprotos).
        - iplus (Tensor): Indices of winner prototypes for each image in the batch, shape (nbatch,).
        - iminus (Tensor): Indices of non-winner prototypes for each image in the batch, shape (nbatch,).
    """
    nbatch, nprotos = distances.shape

    # Find the indices of winner and non-winner prototypes
    iplus = winner_prototype_indices(ydata, yprotos_matrix, distances)
    iminus = winner_prototype_indices(ydata, yprotos_comp_matrix, distances)

    # Extract distances for winner and non-winner prototypes
    Dplus = torch.zeros_like(distances)
    Dminus = torch.zeros_like(distances)
    Dplus[torch.arange(nbatch), iplus] = distances[torch.arange(nbatch), iplus]
    Dminus[torch.arange(nbatch), iminus] = distances[torch.arange(nbatch), iminus]

    return Dplus, Dminus, iplus, iminus


def MU_fun(Dplus: torch.Tensor, Dminus: torch.Tensor) -> torch.Tensor:
    """
    Compute the Mu values for each batch.

    Parameters:
    -----------
    Dplus : torch.Tensor
        Matrix containing distances from positive prototypes, shape (nbatch, nprotos).
    Dminus : torch.Tensor
        Matrix containing distances from negative prototypes, shape (nbatch, nprotos).

    Returns:
    --------
    torch.Tensor
        Array of size (nbatch,) containing Mu values.
    """
    # Compute the numerator and denominator separately
    numerator = (Dplus - Dminus).sum(dim=1)
    denominator = (Dplus + Dminus).sum(dim=1)

    # Avoid division by zero and return Mu values
    return torch.where(denominator != 0, numerator / denominator, torch.zeros_like(numerator))


def IdentityLoss():
    """
    Identity Loss function for prototype-based classification.

    Returns:
    --------
    function
        A function that computes the loss and winner prototype indices given the data, prototype matrices, and distance matrix.
    """
    def f(ydata, yprotos_matrix, yprotos_comp_matrix, distance_matrix: torch.Tensor):
        """
        Compute the Identity Loss and winner prototype indices.

        Parameters:
        -----------
        ydata : Tensor
            Labels of input images.
        yprotos_matrix : Tensor
            Matrix of labels for prototypes.
        yprotos_comp_matrix : Tensor
            Matrix of complementary labels for prototypes.
        distance_matrix : torch.Tensor
            Distance matrix between images and prototypes.

        Returns:
        --------
        tuple
            A tuple containing:
            - loss (Tensor): Identity Loss.
            - iplus (Tensor): Indices of winner prototypes for positive class.
            - iminus (Tensor): Indices of winner prototypes for negative class.
        """
        # Calculate distances to positive and negative prototypes
        Dplus, Dminus, iplus, iminus = winner_prototype_distances(ydata, yprotos_matrix, yprotos_comp_matrix, distance_matrix)

        # Compute Identity Loss using MU_fun and sum across batch
        loss = nn.Identity()(MU_fun(Dplus, Dminus)).sum()

        return loss, iplus, iminus

    return f


def SigmoidLoss(sigma: int = 100):
    """
    Create a Sigmoid Loss function with a specified sigma parameter.

    Parameters:
    -----------
    sigma : int, optional
        Sigma parameter for the Sigmoid function. Default is 100.

    Returns:
    --------
    function
        A Sigmoid Loss function that takes ydata, yprotos_matrix, yprotos_comp_matrix, and distance_matrix as input.
    """

    def f(ydata, yprotos_matrix, yprotos_comp_matrix, distance_matrix: torch.Tensor):
        """
        Compute the Sigmoid Loss.

        Parameters:
        -----------
        ydata : torch.Tensor
            Labels of input images.
        yprotos_matrix : torch.Tensor
            One-hot encoded labels of prototypes.
        yprotos_comp_matrix : torch.Tensor
            One-hot encoded complementary labels of prototypes.
        distance_matrix : torch.Tensor
            Matrix containing distances between images and prototypes.

        Returns:
        --------
        tuple
            A tuple containing:
            - Sigmoid Loss (torch.Tensor): Resulting loss.
            - iplus (torch.Tensor): Indices of winning prototypes from the positive class.
            - iminus (torch.Tensor): Indices of winning prototypes from the negative class.
        """
        Dplus, Dminus, iplus, iminus = winner_prototype_distances(ydata, yprotos_matrix, yprotos_comp_matrix,
                                                                  distance_matrix)

        # Calculate Sigmoid Loss
        loss = nn.Sigmoid()(sigma * MU_fun(Dplus, Dminus)).sum()

        return loss, iplus, iminus

    return f


def ReLULoss():
    """
    Create a function to compute the ReLU-based loss.

    Returns:
    --------
    function
        A function that computes the ReLU-based loss given appropriate inputs.
    """
    def compute_loss(ydata, yprotos_matrix, yprotos_comp_matrix, distance_matrix):
        """
        Compute the ReLU-based loss.

        Parameters:
        -----------
        ydata : Tensor
            Labels of input images.
        yprotos_matrix : Tensor
            One-hot encoded labels of prototypes.
        yprotos_comp_matrix : Tensor
            Complementary one-hot encoded labels of prototypes.
        distance_matrix : Tensor
            Matrix of distances between images and prototypes.

        Returns:
        --------
        tuple
            A tuple containing:
            - loss (Tensor): ReLU-based loss.
            - indices of winner prototypes for positive distances (Tensor).
            - indices of winner prototypes for negative distances (Tensor).
        """
        Dplus, Dminus, iplus, iminus = winner_prototype_distances(ydata, yprotos_matrix, yprotos_comp_matrix, distance_matrix)
        loss = nn.ReLU()(MU_fun(Dplus, Dminus)).sum()
        return loss, iplus, iminus

    return compute_loss


def LeakyReLULoss(negative_slope: float = 0.01):
    """
    Create a LeakyReLU loss function with a given negative slope.

    Parameters:
    -----------
    negative_slope : float, optional
        The negative slope of the LeakyReLU activation function. Default is 0.01.

    Returns:
    --------
    function
        A function that computes the LeakyReLU loss for a given set of inputs.

    Notes:
    ------
    This function returns a closure, which is a function that retains the environment in which it was created.
    This allows the returned function to access the `negative_slope` parameter even after `LeakyReLULoss` returns.
    """

    def f(ydata, yprotos_matrix, yprotos_comp_matrix, distance_matrix: torch.Tensor):
        """
        Compute the LeakyReLU loss for a batch of inputs.

        Parameters:
        -----------
        ydata : Tensor
            Labels of input images.
        yprotos_matrix : Tensor
            Labels of prototypes.
        yprotos_comp_matrix : Tensor
            Complementary labels of prototypes.
        distance_matrix : Tensor
            Distance matrix between images and prototypes.

        Returns:
        --------
        tuple
            A tuple containing:
            - loss (Tensor): The LeakyReLU loss.
            - iplus (Tensor): Indices of winning prototypes with positive labels.
            - iminus (Tensor): Indices of winning prototypes with negative labels.
        """
        Dplus, Dminus, iplus, iminus = winner_prototype_distances(ydata, yprotos_matrix, yprotos_comp_matrix,
                                                                  distance_matrix)

        # Apply LeakyReLU with the given negative slope
        loss = nn.LeakyReLU(negative_slope)(MU_fun(Dplus, Dminus)).sum()

        return loss, iplus, iminus

    return f


def ELULoss(alpha: float = 1):
    """
    Creates a closure for computing the ELU loss.

    Parameters:
    -----------
    alpha : float, optional
        Alpha parameter for ELU activation. Default is 1.

    Returns:
    --------
    function
        A function for computing the ELU loss.

    Notes:
    ------
    The returned function takes the following parameters:
    - ydata : Tensor
        Labels of input images.
    - yprotos_matrix : Tensor
        One-hot encoded labels of prototypes (W^+).
    - yprotos_comp_matrix : Tensor
        One-hot encoded labels of complementary prototypes (W^-).
    - distance_matrix : Tensor
        Distances between images and prototypes.

    Returns a tuple containing the loss value, indices of winner prototypes from W^+, and indices of winner prototypes from W^-.
    """

    def f(ydata, yprotos_matrix, yprotos_comp_matrix, distance_matrix: torch.Tensor):
        """
        Compute ELU loss.

        Parameters:
        -----------
        ydata : Tensor
            Labels of input images.
        yprotos_matrix : Tensor
            One-hot encoded labels of prototypes (W^+).
        yprotos_comp_matrix : Tensor
            One-hot encoded labels of complementary prototypes (W^-).
        distance_matrix : Tensor
            Distances between images and prototypes.

        Returns:
        --------
        tuple
            A tuple containing:
            - loss_value : Tensor
                ELU loss value.
            - winner_indices_plus : Tensor
                Indices of winner prototypes from W^+.
            - winner_indices_minus : Tensor
                Indices of winner prototypes from W^-.
        """
        Dplus, Dminus, iplus, iminus = winner_prototype_distances(ydata, yprotos_matrix, yprotos_comp_matrix,
                                                                  distance_matrix)

        # Compute the ELU loss
        loss_value = nn.ELU(alpha)(MU_fun(Dplus, Dminus)).sum()

        return loss_value, iplus, iminus

    return f


def RReLULoss(lower=0.125, upper=0.3333333333333333):
    """
    Create a function to compute the RReLULoss.

    Parameters:
    -----------
    lower : float, optional
        Lower bound for the RReLU activation function. Default is 0.125.
    upper : float, optional
        Upper bound for the RReLU activation function. Default is 0.3333333333333333.

    Returns:
    --------
    function
        A function that computes the RReLULoss given inputs.

    Notes:
    ------
    The RReLULoss is computed using the winner prototype distances and a modified RReLU activation function.
    """

    def f(ydata, yprotos_matrix, yprotos_comp_matrix, distance_matrix: torch.Tensor):
        """
        Compute the RReLULoss.

        Parameters:
        -----------
        ydata : torch.Tensor
            True class labels of input data.
        yprotos_matrix : torch.Tensor
            One-hot encoded labels of prototypes.
        yprotos_comp_matrix : torch.Tensor
            One-hot encoded complementary labels of prototypes.
        distance_matrix : torch.Tensor
            Matrix of distances between input data and prototypes.

        Returns:
        --------
        tuple
            A tuple containing:
            - loss (torch.Tensor): Computed RReLULoss.
            - iplus (torch.Tensor): Indices of the winning prototypes with the same labels as input data.
            - iminus (torch.Tensor): Indices of the winning prototypes with complementary labels to input data.
        """
        # Compute winner prototype distances
        Dplus, Dminus, iplus, iminus = winner_prototype_distances(ydata, yprotos_matrix, yprotos_comp_matrix,
                                                                  distance_matrix)

        # Compute modified RReLU activation function on MU
        mu = MU_fun(Dplus, Dminus)
        rrelu = nn.RReLU(lower, upper)
        loss = rrelu(mu).sum()

        return loss, iplus, iminus

    return f


def GELULoss():
    """
    GELU Loss function for prototype-based classification.

    Returns:
    --------
    function
        A function that computes the loss and winner prototype indices given the data, prototype matrices, and distance matrix.
    """
    def f(ydata, yprotos_matrix, yprotos_comp_matrix, distance_matrix: torch.Tensor):
        """
        Compute the GELU Loss and winner prototype indices.

        Parameters:
        -----------
        ydata : Tensor
            Labels of input images.
        yprotos_matrix : Tensor
            Matrix of labels for prototypes.
        yprotos_comp_matrix : Tensor
            Matrix of complementary labels for prototypes.
        distance_matrix : torch.Tensor
            Distance matrix between images and prototypes.

        Returns:
        --------
        tuple
            A tuple containing:
            - loss (Tensor): GELU Loss.
            - iplus (Tensor): Indices of winner prototypes for positive class.
            - iminus (Tensor): Indices of winner prototypes for negative class.
        """
        # Calculate distances to positive and negative prototypes
        Dplus, Dminus, iplus, iminus = winner_prototype_distances(ydata, yprotos_matrix, yprotos_comp_matrix, distance_matrix)

        # Compute Identity Loss using MU_fun and sum across batch
        loss = nn.GELU()(MU_fun(Dplus, Dminus)).sum()

        return loss, iplus, iminus

    return f


def get_loss_function(config_file_path) -> Callable:
    """
    Get the appropriate loss function based on the command-line arguments.

    Parameters:
    -----------
    config_file_path : str
        Path to the YAML configuration file.

    Returns:
    --------
    Callable
        The selected loss function.

    Notes:
    ------
    Supported loss functions:
    - 'sigmoid': SigmoidLoss with optional 'sigma'.
    - 'relu': ReLULoss.
    - 'leaky_relu': LeakyReLULoss with optional 'sigma'.
    - 'elu': ELULoss with optional 'sigma'.
    - 'rrelu': RReLULoss.
    - Any other value: IdentityLoss.
    """

    with open(config_file_path, "r") as file:
        config = yaml.safe_load(file)

    if config['train']['hyperparams']['act_fun'] == 'sigmoid':
        sigma = config['train']['hyperparams']['sigma'] or 100
        return SigmoidLoss(sigma)
    elif config['train']['hyperparams']['act_fun'] == 'relu':
        return ReLULoss()
    elif config['train']['hyperparams']['act_fun'] == 'leaky_relu':
        sigma = config['hyperparams']['sigma'] or 0.1
        return LeakyReLULoss(sigma)
    elif config['train']['hyperparams']['act_fun'] == 'elu':
        sigma = config['hyperparams']['sigma'] or 1
        return ELULoss(alpha=sigma)
    elif config['train']['hyperparams']['act_fun'] == 'rrelu':
        return RReLULoss()
    elif config['train']['hyperparams']['act_fun'] == 'gelu':
        return GELULoss()
    else:
        return IdentityLoss()
