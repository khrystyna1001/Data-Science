from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

def main():
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model.eval()

    responses = [
        "The capital of France is Paris",
        "France capital is Paris the",
    ]

    def compute_ppl(text):
        encodings = tokenizer(text, return_tensors='pt')
        input_ids = encodings['input_ids']

        # Compute loss
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss
        
        ppl = torch.exp(loss).item()
        return ppl
    
    # Calculate PPL for each response
    for i, response in enumerate(responses, 1):
        ppl = compute_ppl(response)
        print(f"PPL Score: {ppl:.2f}\n")

if __name__ == "__main__":
    main()