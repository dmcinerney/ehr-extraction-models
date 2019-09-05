from models.clinical_bert.model import ClinicalBertExtractionModel
model = ClinicalBertExtractionModel()
import torch
input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
outputs = model(input_ids, input_mask)
print(outputs)
print(outputs[0].shape, outputs[1].shape)
