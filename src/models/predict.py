import logging

import torch

logger = logging.getLogger(__name__)

def predict_single_sample(model, tokens_tensor, device="cpu") -> float:
    """
    Runs a single sequence of tokens through the given model and returns the raw anomaly score.
    """
    model.eval()
    model.to(device)
    tokens_tensor = tokens_tensor.to(device)

    with torch.no_grad():
        score = model(tokens_tensor).item()

    return score

def batch_predict(model, dataloader, device="cpu") -> list:
    """
    Runs batch predictions for a given PyTorch model.
    """
    model.eval()
    model.to(device)
    all_scores = []

    with torch.no_grad():
        for x, _ in dataloader:
            x = x.to(device)
            out = model(x)
            all_scores.extend(out.cpu().numpy().tolist())

    return all_scores
