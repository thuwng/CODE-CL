import torch
__all__ = ["compute_conceptor", "or_operation", "aperture_adaptation", "and_operation", "not_operation",
           "incremental_conceptor_extension", "measure_conceptor_capacity", "similarity", "conceptor_to_correlation"]

def aperture_adaptation(conceptor, gamma):
    """
    Compute the aperture adaptation fo a conceptor (C). For instance, applying the operation to a conceptor (C) with
    adapdation alpha, will result in a new conceptor (C*) with adaptation gamma*alpha
    :param conceptor:
    :param gamma:
    :return:
    """
    identity = torch.eye(conceptor.size(0)).to(conceptor.device)
    C = torch.matmul(conceptor, torch.linalg.inv(conceptor + (gamma**-2)*(identity - conceptor)))
    U, S, Ut = torch.svd(C)
    return torch.matmul(torch.matmul(U, torch.diag(torch.clamp(S, min=1e-8, max=0.9999999))), U.T)


def compute_conceptor(data, aperture=4):
    identity = torch.eye(data.size(0)).to(data.device)
    R = torch.matmul(data, data.T) / data.size(1)
    conceptor = torch.matmul(R, torch.linalg.inv(R + (aperture ** -2) * identity))
    return conceptor


def conceptor_to_correlation(conceptor, aperture=4):
    identity = torch.eye(conceptor.size(0)).to(conceptor.device)
    return (aperture**-2)*torch.matmul(torch.linalg.pinv(identity-conceptor), conceptor)

def or_operation(conceptor1, conceptor2, aperture=4):
    return not_operation(and_operation(not_operation(conceptor1), not_operation(conceptor2)))


def not_operation(conceptor):
    identity = torch.eye(conceptor.size(0)).to(conceptor.device)
    return identity - conceptor


def and_operation(C, B, aperture=None):
    dim = C.size(0)
    tol = 1e-6
    UC, SC, _ = torch.svd(C)
    UB, SB, _ = torch.svd(B)
    numRankC = int(torch.sum(1.0 * (SC > tol)))
    numRankB = int(torch.sum(1.0 * (SB > tol)))
    UC0 = UC[:, numRankC:]
    UB0 = UB[:, numRankB:]
    W, Sigma, Wt = torch.svd(torch.matmul(UC0, UC0.T) + torch.matmul(UB0, UB0.T))
    numRankSigma = int(torch.sum(1.0 * (Sigma > tol)))
    Wgk = W[:, numRankSigma:]
    CandB = torch.matmul(torch.matmul(Wgk, torch.linalg.inv(torch.matmul(
        torch.matmul(Wgk.T, (torch.linalg.pinv(C, atol=tol) + torch.linalg.pinv(B, atol=tol) - torch.eye(dim).to(C.device))), Wgk))),
                         Wgk.T)
    U, S, Ut = torch.svd(CandB)
    CandB = torch.matmul(torch.matmul(U, torch.diag(torch.clamp(S, min=1e-8, max=0.9999999))), U.T)
    return CandB


def incremental_conceptor_extension(conceptor1, conceptor2, m, n, aperture=4):
    term = aperture_adaptation(conceptor1, (m**0.5)*(aperture**-1))
    term = or_operation(term, conceptor2)
    return aperture_adaptation(term, (m+n)**0.5*aperture)


def measure_conceptor_capacity(conceptor):
    _, S, _ = torch.svd(conceptor)
    return S.mean()


def similarity(conceptor1, conceptor2):
    U1, S1, _ = torch.svd(conceptor1)
    U2, S2, _ = torch.svd(conceptor2)
    num = torch.norm(torch.matmul(torch.diag(S1**0.5), torch.matmul(U1.T, torch.matmul(U2, torch.diag(S2**0.5)))))
    den = torch.norm(S1)*torch.norm(S2)
    return (num**2)/den
