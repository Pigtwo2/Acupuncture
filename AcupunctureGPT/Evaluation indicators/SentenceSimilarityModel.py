
import torch
import torch.nn as nn
import numpy as np
import random
from transformers import BertTokenizer, BertModel
import torch.nn.functional as F

import matplotlib.pyplot as plt
import seaborn as sns

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class SentenceEmbeddingEnhancer(nn.Module):
    def __init__(self, bert_width, transformer_width):
        super(SentenceEmbeddingEnhancer, self).__init__()
        self.conv1d_bert = nn.Conv1d(in_channels=bert_width, out_channels=bert_width, kernel_size=1)
        self.conv1d_transformer = nn.Conv1d(in_channels=transformer_width, out_channels=transformer_width,
                                            kernel_size=1)
        self.linear = nn.Linear(bert_width, bert_width)

    def forward(self, embeddings_A, SA_A):

        embeddings_A = embeddings_A.transpose(1, 2)
        SA_A = SA_A.transpose(1, 2)

        embeddings_A_conv = self.conv1d_bert(embeddings_A)
        max_pool_out = F.max_pool1d(embeddings_A_conv, kernel_size=embeddings_A_conv.shape[2]).squeeze(2)
        avg_pool_out = F.avg_pool1d(embeddings_A_conv, kernel_size=embeddings_A_conv.shape[2]).squeeze(2)

        mask_input = max_pool_out + avg_pool_out
        Em_Mask1 = torch.sigmoid(self.linear(mask_input))

  
        embeddings_A1 = embeddings_A.transpose(1, 2) * Em_Mask1.unsqueeze(1)

        SA_A_conv = self.conv1d_transformer(SA_A)
        global_avg_pool_out = F.adaptive_avg_pool1d(SA_A_conv, 1).squeeze(2)
        SA_Mask1 = torch.sigmoid(self.linear(global_avg_pool_out))

  
        SA_A1 = SA_A.transpose(1, 2) * SA_Mask1.unsqueeze(1)

        masks = torch.stack([Em_Mask1, SA_Mask1], dim=1)
        masks = F.softmax(masks, dim=1)
        Em_Mask2, SA_Mask2 = masks[:, 0, :], masks[:, 1, :]

 
        embeddings_A2 = embeddings_A1 * Em_Mask2.unsqueeze(1)
        SA_A2 = SA_A1 * SA_Mask2.unsqueeze(1)

        # 融合输出
        SA_A_output = embeddings_A2 + SA_A2
        # return SA_A_output


        return SA_A_output,SA_Mask1,SA_Mask2,SA_A1,SA_A2, Em_Mask1, Em_Mask2,mask_input,embeddings_A1,embeddings_A2  


    def visualize_with_heatmap(self, embeddings, title=''):
        plt.figure(figsize=(8, 5))
   
        if embeddings.dim() == 1:
            embeddings = embeddings.unsqueeze(0)  
        if embeddings.dim() == 3:
            embeddings = embeddings[0]  
        ax = sns.heatmap(embeddings.detach().cpu().numpy(), cmap='viridis',annot=True)  
        # plt.title(title)
        plt.xlabel('')
        plt.ylabel('')
        plt.show(block=True)


class SentenceSimilarityModel(nn.Module):
    def __init__(self, bert_model_name: str = 'bert-base-chinese', seed=42) -> None:
        super(SentenceSimilarityModel, self).__init__()
     
        self.bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.bert = BertModel.from_pretrained(bert_model_name)
        set_seed(seed)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.bert.config.hidden_size, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

   
        self.enhancer = SentenceEmbeddingEnhancer(self.bert.config.hidden_size, self.bert.config.hidden_size)

    def forward(self, sentA: str, sentB: str) -> torch.Tensor:
      
        input_ids_A = self.bert_tokenizer(sentA, return_tensors="pt", padding=True, truncation=True, max_length=512)['input_ids']
        input_ids_B = self.bert_tokenizer(sentB, return_tensors="pt", padding=True, truncation=True, max_length=512)['input_ids']
        embeddings_A = self.bert(input_ids_A)[0]
        embeddings_B = self.bert(input_ids_B)[0]
        SA_A = self.transformer_encoder(embeddings_A)
        SA_B = self.transformer_encoder(embeddings_B)

        Fusion=embeddings_A+SA_A


        # enhanced_SA_A = self.enhancer(embeddings_A, SA_A)
        # enhanced_SA_B = self.enhancer(embeddings_B, SA_B)

        # enhanced_SA_A, Em_Mask1_A, Em_Mask2_A = self.enhancer(embeddings_A, SA_A)
        # enhanced_SA_B, Em_Mask1_B, Em_Mask2_B = self.enhancer(embeddings_B, SA_B)
        #
        SA_A_output,SA_Mask1,SA_Mask2,SA_A1,SA_A2, Em_Mask1, Em_Mask2,mask_input,embeddings_A1,embeddings_A2 = self.enhancer(embeddings_A, SA_A)
        enhanced_SA_B = self.enhancer(embeddings_B, SA_B)

        avg_enhanced_SA_A = torch.mean(SA_A_output, dim=1)
        # avg_enhanced_SA_B = torch.mean(enhanced_SA_B, dim=1)

        enhanced_SA_B_tensor = enhanced_SA_B[0]  
        avg_enhanced_SA_B = torch.mean(enhanced_SA_B_tensor, dim=1)

        similarity = F.cosine_similarity(avg_enhanced_SA_A, avg_enhanced_SA_B, dim=1, eps=1e-8).unsqueeze(-1)

    
        self.enhancer.visualize_with_heatmap(embeddings_A[0], 'Heatmap of embeddings_A')
        self.enhancer.visualize_with_heatmap(SA_A[0], 'Heatmap of SA_A')
        self.enhancer.visualize_with_heatmap(Fusion[0], 'Heatmap of SA_A')

        self.enhancer.visualize_with_heatmap(mask_input, 'Heatmap of mask_input')
        self.enhancer.visualize_with_heatmap(Em_Mask1, 'Heatmap of Em_Mask1')
        self.enhancer.visualize_with_heatmap(Em_Mask2, 'Heatmap of Em_Mask2')
        self.enhancer.visualize_with_heatmap(embeddings_A1, 'Heatmap of embeddings_A1')
        self.enhancer.visualize_with_heatmap(embeddings_A2, 'Heatmap of embeddings_A2')

        self.enhancer.visualize_with_heatmap(SA_Mask1, 'Heatmap of SA_Mask1')
        self.enhancer.visualize_with_heatmap(SA_Mask2, 'Heatmap of SA_Mask2')
        self.enhancer.visualize_with_heatmap(SA_A1, 'Heatmap of SA_A1')
        self.enhancer.visualize_with_heatmap(SA_A2, 'Heatmap of SA_A2')
        self.enhancer.visualize_with_heatmap(SA_A_output, 'Heatmap of SA_A_output')

        return similarity

        # return similarity, (embeddings_A, SA_A, Em_Mask1_A, Em_Mask2_A), (embeddings_B, SA_B, Em_Mask1_B, Em_Mask2_B)



if __name__ == '__main__':
    model = SentenceSimilarityModel()

    Label=" "
    Sentence=" "

    similarity_score = model(Label, Sentence).item()
    print("Similarity Score:", similarity_score)


