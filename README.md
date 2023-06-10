# Transfer Learning with DistilBERT

This project implements transfer learning with the DistilBERT model for text classification. The goal is to fine-tune the pre-trained DistilBERT model on the SST-2 dataset and achieve improved accuracy compared to the claimed performance of the `distilbert-base-uncased-finetuned-sst-2-english` model.

## Dataset

The SST-2 (Stanford Sentiment Treebank) dataset is used for fine-tuning the DistilBERT model. It consists of movie reviews classified into positive and negative sentiments. The dataset is preprocessed and split into train, validation, and test sets.

## Model Architecture

The DistilBERT model is a compact version of BERT (Bidirectional Encoder Representations from Transformers) that retains most of the performance while being smaller and faster. The pre-trained `distilbert-base-uncased-finetuned-sst-2-english` model is used as the base model.

## Fine-tuning

The last two layers and the last six transformer blocks of the DistilBERT model are updated during the fine-tuning process. By selectively unfreezing and fine-tuning these layers, we aim to capture more task-specific information and improve the model's performance.

## Training and Evaluation

The model is trained on the train set and evaluated on the validation set. The training process involves optimizing the model's parameters using the Adam optimizer and minimizing the cross-entropy loss. The model's performance is evaluated in terms of accuracy on the validation set.

## Results

After fine-tuning the DistilBERT model and evaluating it on the validation set, we achieved an accuracy of 91.51%, surpassing the claimed accuracy of 91.3% by the author of `distilbert-base-uncased-finetuned-sst-2-english` on the dev set.

## Usage

To reproduce the results or use the fine-tuned model for inference, follow these steps:

1. Install the required dependencies.

2. Download the SST-2 dataset and preprocess it according to the required format.

3. Run the notebook `FT_TL_distilbert_sst-2.ipynb` to start the fine-tuning and transfer learning process. Adjust the hyperparameters as needed.

4. After training, the fine-tuned model can be used for inference on new data.

5. Optionally, you can experiment with different layers to update during fine-tuning or modify other aspects of the model architecture for further improvement.

## References

- Hugging Face Transformers: https://github.com/huggingface/transformers
- DistilBERT Finetuned SST-2 English: https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english
- "DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter" by Victor Sanh, Lysandre Debut, Julien Chaumond, and Thomas Wolf

Please refer to the documentation and original research papers for more information on the DistilBERT model and the SST-2 dataset.
