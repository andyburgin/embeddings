from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
from utilities.args_parser import parse_args
from utilities.embeddings_print import print_direct_embeddings
import numpy as np
import torch

def main(tokenizer_name, model_name, prompt, min_score, min_prob):
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Load the model configuration and model
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Tokenize the input text
    tokenizer.pad_token_id = tokenizer.eos_token_id
    input_ids = tokenizer([prompt], return_tensors="pt")

    # make a forward pass
    outputs = model.generate(**input_ids, max_new_tokens=1 , return_dict_in_generate=True, output_scores=True)

    # output tokens and their scores
    print("Next Token Scores:")
    for index, score in zip(range(len(outputs.scores[0][0])),outputs.scores[0][0]):
        token = tokenizer.decode(index)
        if score > float(min_score):
            print(f"{index} {token} {score}")

    print("--------------------------------------------")

    # output tokens and their probability
    print("Next Token Probability:")
    predictions = torch.nn.functional.softmax(outputs.scores[0][0], dim=-1)
    for index, score in zip(range(len(predictions)),predictions):
        token = tokenizer.decode(index)
        if score > float(min_prob)/100:
            print(f"{index} {token} {score:.2%}")

if __name__ == "__main__":
    # parse the arguments
    args = parse_args()

    # execute
    main(args.tokenizer, args.model, args.prompt, args.score, args.probability)
