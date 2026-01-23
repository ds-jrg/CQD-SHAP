import torch
from kbc.learn import kbc_model_load
import pandas as pd

class LinkPrediction:
    def __init__(self, model_path):
        self.kbc, _, _ = kbc_model_load(model_path)
        self.model = self.kbc.model
        self.model.eval()
        self.device = next(self.model.parameters()).device
        # print("Successfully loaded model and set to eval mode (device: {})".format(self.device))

    def predict(self, h_id, r_id, return_df=True, k=-1, score_normalize=False):
        h_emb = self.model.embeddings[0](torch.tensor([h_id], device=self.device))
        r_emb = self.model.embeddings[1](torch.tensor([r_id], device=self.device))
        scores = self.model.forward_emb(h_emb, r_emb)
        
        if score_normalize:
            scores = torch.sigmoid(scores)
            
        if return_df:
            df = pd.DataFrame(scores.cpu().detach().numpy()[0], columns=["score"])
            df = df.sort_values(by="score", ascending=False)
            if k > 0:
                df = df.head(k)
            return df
        else:
            if k > 0:
                scores = scores.topk(k)
        return scores
    
    def predict_batch(self, h_ids, r_ids, k=-1, score_normalize=False):
        if not torch.is_tensor(h_ids):
            h_ids = torch.tensor(h_ids, device=self.device, dtype=torch.long)

        # If r_ids is a scalar, repeat it for each head
        if isinstance(r_ids, int):
            r_ids = torch.tensor([r_ids] * len(h_ids), device=self.device, dtype=torch.long)
        elif not torch.is_tensor(r_ids):
            r_ids = torch.tensor(r_ids, device=self.device, dtype=torch.long)

        h_emb = self.model.embeddings[0](h_ids)
        r_emb = self.model.embeddings[1](r_ids)

        # Debug shapes
        #print("h_emb:", h_emb.shape, "r_emb:", r_emb.shape)

        scores = self.model.forward_emb(h_emb, r_emb)  # [batch_size, num_entities]
        
        if score_normalize:
            scores = torch.sigmoid(scores)

        if k > 0:
            scores, indices = torch.topk(scores, k, dim=1)
            return scores, indices
        return scores