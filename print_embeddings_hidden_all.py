from transformers import AutoModel, AutoTokenizer
from utilities.args_parser import parse_args
from utilities.embeddings_print import print_direct_embeddings

def main(tokenizer_name, model_name, prompt):
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    # Load the model configuration and model
    model = AutoModel.from_pretrained(model_name)

    # Tokenize the input text
    tokens = tokenizer(prompt, return_tensors="pt")
    input_ids = tokens['input_ids']

    # make a forward pass
    outputs = model(input_ids, output_hidden_states=True)

    # Get embeddings from last hidden state layer
    embeddings = outputs.hidden_states

    for idx, e in enumerate(embeddings):
        print(f"Embedddings for hidden layer {idx}")
        # Use the utility function to print direct embeddings
        print_direct_embeddings(tokenizer, embeddings[idx], input_ids)

if __name__ == "__main__":
    # parse the arguments
    args = parse_args()

    # execute
    main(args.tokenizer, args.model, args.prompt)
