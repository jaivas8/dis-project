import torch
import pandas as pd
from model import get_train_test_loaders,evaluate
from LstmDecoder import LstmDecoder
from utils import calculate_precision_recall_f1
torch.manual_seed(0)


if __name__ == "__main__":
    # Load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LstmDecoder(device=device, input_dim=3, l1_out=[8,16,32,32], lstm_hidden_dim=[32,32,64], l2_out=[64,128,128], decoder_chan=8).to(device)
    model_loader = torch.load('final.pth')
    model.load_state_dict(model_loader)

    image_path = 'network/data/segmented_images'
    data_path = 'network/data'
    ribbon_type = 'one_ribbon'
    batch_size = 16
    train_test_split = 0.8
    results= []
    for length in range(2,1002,25):
        max_len = length
        train_loaders, test_loaders = get_train_test_loaders(image_path, ribbon_type, data_path, max_len, batch_size, train_test_split)
        avg_train_loss = 0
        avg_test_loss = 0
        avg_train_precision = 0
        avg_train_recall = 0
        avg_train_f1 = 0
        avg_test_precision = 0
        avg_test_recall = 0
        avg_test_f1 = 0


        for loader in train_loaders:
            train_loss , train_precisions, train_recalls, train_f1s = evaluate(model, device, loader)
            avg_train_loss += train_loss
            avg_train_precision += train_precisions
            avg_train_recall += train_recalls
            avg_train_f1 += train_f1s
        avg_train_loss /= len(train_loaders)
        avg_train_precision /= len(train_loaders)
        avg_train_recall /= len(train_loaders)
        avg_train_f1 /= len(train_loaders)

        for loader in test_loaders:
            test_loss, test_precisions, test_recalls, test_f1s = evaluate(model, device, loader)
            avg_test_loss += test_loss
            avg_test_precision += test_precisions
            avg_test_recall += test_recalls
            avg_test_f1 += test_f1s
        
        avg_test_loss /= len(test_loaders)
        avg_test_precision /= len(test_loaders)
        avg_test_recall /= len(test_loaders)
        avg_test_f1 /= len(test_loaders)

        result = {
            'length': length,
            'train_loss': avg_train_loss,
            'train_precision': avg_train_precision,
            'train_recall': avg_train_recall,
            'train_f1': avg_train_f1,
            'test_loss': avg_test_loss,
            'test_precision': avg_test_precision,
            'test_recall': avg_test_recall,
            'test_f1': avg_test_f1
        }
        results.append(result)
    df = pd.DataFrame(results)
    df.to_csv(f'results/model_evaluation/accuracy_final.csv')
       