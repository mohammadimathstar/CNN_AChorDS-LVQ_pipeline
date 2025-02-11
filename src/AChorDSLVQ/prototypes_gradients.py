import torch
from torch.autograd import Function
from src.utils.grassmann import compute_distances_on_grassmann_mdf


def rotate_data(xs: torch.Tensor, rotation_matrix: torch.Tensor, winner_ids: torch.Tensor, return_rotation_matrix: bool = False):
    """
    Rotate the input data based on the winner prototypes.

    Args:
        xs (torch.Tensor): Input data of shape (batch_size, dim_of_data, dim_of_subspace).
        rotation_matrix (torch.Tensor): Rotation matrices of shape (batch_size, num_of_prototypes, dim_of_subspace, dim_of_subspace).
        winner_ids (torch.Tensor): Indices of winner prototypes for each data point, of shape (batch_size, 2).
        return_rotation_matrix (bool, optional): Whether to return the rotation matrices. Defaults to False.

    Returns:
        torch.Tensor or tuple: Rotated data if return_rotation_matrix is False, otherwise rotated data along with rotation matrices.
    """
    assert xs.ndim == 3, f"Input data should be of shape (batch_size, dim_of_data, dim_of_subspace), but it is of shape {xs.shape}"
    assert winner_ids.shape[1] == 2, f"There should only be two winner prototypes (W^+) for each data point, but found {winner_ids.shape[1]} winners."

    nbatch = xs.shape[0]
    Qwinners = rotation_matrix[torch.arange(nbatch).unsqueeze(-1), winner_ids]  # Shape: (batch_size, 2, dim_of_subspace, dim_of_subspace)

    Qwinners1, Qwinners2 = Qwinners[:, 0], Qwinners[:, 1]  # Shape: (batch_size, dim_of_subspace, dim_of_subspace)
    rotated_xs1, rotated_xs2 = torch.bmm(xs, Qwinners1), torch.bmm(xs, Qwinners2)  # Shape: (batch_size, dim_of_data, dim_of_subspace)

    if return_rotation_matrix:
        return rotated_xs1, rotated_xs2, Qwinners1, Qwinners2
    return rotated_xs1, rotated_xs2


def rotate_prototypes(xprotos: torch.Tensor, rotation_matrix: torch.Tensor, winner_ids: torch.Tensor):
    """
    Rotate the winner prototypes based on the provided rotation matrices.

    Args:
        xprotos (torch.Tensor): Tensor of prototypes with shape (nprotos, dim_of_data, dim_of_subspace).
        rotation_matrix (torch.Tensor): Tensor of rotation matrices with shape (batch_size, nprotos, dim_of_subspace, dim_of_subspace).
        winner_ids (torch.Tensor): Tensor of winner prototype indices with shape (batch_size, 2).

    Returns:
        tuple: Rotated prototypes (rotated_proto1, rotated_proto2).
    """
    # Ensure the input prototypes tensor has the correct dimensions
    assert xprotos.ndim == 3, f"Expected xprotos to be of shape (nprotos, dim_of_data, dim_of_subspace), but got shape {xprotos.shape}"
    # Ensure there are exactly two winner prototypes for each data point
    assert winner_ids.shape[
               1] == 2, f"Expected winner_ids to have shape (batch_size, 2), but got shape {winner_ids.shape}"

    # Get batch size and number of prototypes
    nbatch, nprotos = rotation_matrix.shape[:2]

    # Extract the rotation matrices for the winner prototypes
    Qwinners = rotation_matrix[
        torch.arange(nbatch).unsqueeze(-1), winner_ids]  # shape: (batch_size, 2, dim_of_subspace, dim_of_subspace)
    Qwinners1, Qwinners2 = Qwinners[:, 0], Qwinners[:, 1]

    # Ensure the extracted winner rotation matrices have the correct batch size
    assert Qwinners1.shape[0] == nbatch, f"Expected Qwinners1 to have batch size {nbatch}, but got {Qwinners1.shape[0]}"

    # Extract the winner prototypes based on winner_ids
    xprotos_winners = xprotos[winner_ids]  # shape: (batch_size, 2, dim_of_data, dim_of_subspace)
    xprotos1, xprotos2 = xprotos_winners[:, 0], xprotos_winners[:, 1]

    # Rotate the winner prototypes using batch matrix multiplication
    rotated_proto1 = torch.bmm(xprotos1, Qwinners1.to(xprotos1.dtype))
    rotated_proto2 = torch.bmm(xprotos2, Qwinners2.to(xprotos1.dtype))

    return rotated_proto1, rotated_proto2



class DistanceLayer(Function):
    @staticmethod
    def forward(ctx, xs_subspace, xprotos, relevances):
        """
        Forward pass of the ChordalPrototypeLayer.

        Args:
            ctx: Context object to save tensors for backward computation.
            xs_subspace (torch.Tensor): Input subspaces.
            xprotos (torch.Tensor): Prototypes.
            relevances (torch.Tensor): Relevance parameters.

        Returns:
            tuple: Output distance and Qw.
        """

        # Compute distances between data and prototypes
        output = compute_distances_on_grassmann_mdf(
            xs_subspace,
            xprotos,
            relevances,
        )

        ctx.save_for_backward(
            xs_subspace, xprotos, relevances,
            output['distance'], output['Q'], output['Qw'], output['canonicalcorrelation'])
        return output['distance'], output['Qw']

    @staticmethod
    def backward(ctx, grad_output, grad_qw):
        """
        Backward pass of the ChordalPrototypeLayer.

        Args:
            ctx: Context object containing saved tensors from forward pass.
            grad_output (torch.Tensor): Gradient of the loss with respect to the output.
            grad_qw (torch.Tensor): Gradient of Qw.

        Returns:
            tuple: Gradient with respect to inputs.
        """
        
        nbatch = grad_output.shape[0]

        xs_subspace, xprotos, relevances, distances, Q, Qw, cc = ctx.saved_tensors
        diag_rel = torch.tile(
            relevances[0],
            (xprotos.shape[-2], 1)
        )

        # there are some example that their gradient of loss (for sigmoid loss fun) is zero
        # here we set their indices to negative (we do not use them for training)
        device = grad_output.get_device()


        # Handle cases where the gradient of loss is zero (for sigmoid cost function)
        winner_ids = torch.stack([
            torch.nonzero(gd).T[0] if len(torch.nonzero(gd).T[0]) == 2 else torch.tensor([-1, -2], device=device) for gd
            in
            torch.unbind(grad_output)
        ], dim=0)

        if len(torch.argwhere(winner_ids<0)) > 0:
            s = torch.argwhere((winner_ids < 0)[:, 0]).T[0]
            Q[s] = 0
            Qw[s] = 0

        # **********************************************
        # ********** gradient of prototypes ************
        # **********************************************

        # Rotate data points (based on winner prototypes)
        rotated_xs1, rotated_xs2, Qwinners1, Qwinners2 = rotate_data(xs_subspace,
                                                                     Q,
                                                                     winner_ids,
                                                                     return_rotation_matrix=True)
        dist_grad1 = grad_output[torch.arange(nbatch), winner_ids[torch.arange(nbatch), 0]]
        dist_grad2 = grad_output[torch.arange(nbatch), winner_ids[torch.arange(nbatch), 1]]

        # gradient of prototypes
        grad_protos1 = - rotated_xs1 * diag_rel.unsqueeze(0) * dist_grad1.unsqueeze(-1).unsqueeze(-1)
        grad_protos2 = - rotated_xs2 * diag_rel.unsqueeze(0) * dist_grad2.unsqueeze(-1).unsqueeze(-1)

        # **********************************************
        # ************ gradient of inputs **************
        # **********************************************

        # Rotate prototypes (based on winner prototypes)
        rotated_xprotos1, rotated_xprotos2 = rotate_prototypes(xprotos, Qw, winner_ids)

        # gradient of principal direction of data (U)
        grad_U1 = - rotated_xprotos1 * diag_rel.unsqueeze(0) * dist_grad1.unsqueeze(-1).unsqueeze(-1)
        grad_U2 = - rotated_xprotos2 * diag_rel.unsqueeze(0) * dist_grad2.unsqueeze(-1).unsqueeze(-1)

        # gradient of data: CHECKKKKKKKKKKKKKKKKKK
        grad_data_subspace = (
            torch.bmm(
                grad_U1.to(Qwinners1.dtype), torch.transpose(Qwinners1, 2, 1)
            ) + torch.bmm(
                grad_U2.to(Qwinners1.dtype), torch.transpose(Qwinners2, 2, 1)
            )
        )
        # grad_data_subspace = (
        #         torch.bmm(grad_U1.to(Qwinners1.dtype), Qwinners1) + torch.bmm(grad_U2.to(Qwinners1.dtype), Qwinners2)
        # )
        assert grad_data_subspace.shape[0] == nbatch, (f"grad of data should be of shape ({nbatch}, D, d) but it is"
                                                       f" {grad_data_subspace.shape[0]}.")

        # **********************************************
        # ********** gradient of relevances ************
        # **********************************************
        CanCorrwinners1 = cc[torch.arange(nbatch), winner_ids[torch.arange(nbatch), 0]]
        CanCorrwinners2 = cc[torch.arange(nbatch), winner_ids[torch.arange(nbatch), 1]]
        grad_rel = - (
                CanCorrwinners1 * dist_grad1.unsqueeze(-1) +
                CanCorrwinners2 * dist_grad2.unsqueeze(-1)
        )

        grad_xs = grad_protos = grad_relevances = None

        if ctx.needs_input_grad[0]:
            grad_xs = grad_data_subspace
        if ctx.needs_input_grad[1]:
            grad_protos = torch.zeros_like(xprotos)
            grad_protos[winner_ids[torch.arange(nbatch), 0]] = grad_protos1.to(grad_protos.dtype)
            grad_protos[winner_ids[torch.arange(nbatch), 1]] = grad_protos2.to(grad_protos.dtype)
            # print(grad_protos.shape)
        if ctx.needs_input_grad[2]:
            grad_relevances = grad_rel

        return grad_xs, grad_protos, grad_relevances
