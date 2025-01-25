import torch
from transformers import AutoModel, AutoTokenizer, LlamaForCausalLM
from utilities.args_parser import parse_args
from utilities.embeddings_print import print_direct_embeddings

def main(tokenizer_name, model_name, prompt):
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    # Load the model configuration and model
    #model = AutoModel.from_pretrained(model_name)
    model = LlamaForCausalLM.from_pretrained(model_name)

    # Tokenize the input text
    tokens = tokenizer(prompt, return_tensors="pt")
    input_ids = tokens['input_ids']

    # Directly use the embeddings layer to get embeddings for the input_ids
    with torch.no_grad():
        #embeddings = model.get_output_embeddings()(input_ids)
        embeddings = model.get_output_embeddings()

    # Use the utility function to print direct embeddings
    #print_direct_embeddings(tokenizer, embeddings, input_ids)
    #print(embeddings)
    #print(type(model))

    # Retrieve the embeddings weight matrix
    dump = []
    embedding_weights = embeddings.weight.detach().cpu()
    outputFile = open("outputembeddings.txt", "w")

    for token_id in input_ids[0]:
        print(token_id.item())
        #print(embedding_weights[token_id].shape)
        #print(embedding_weights[token_id])
        csv_embedding = " ".join(map(str, embedding_weights[token_id.item()].numpy()))
        #print(csv_embedding)
        outputFile.write(csv_embedding)
        outputFile.write("\n")

    outputFile.close()

    #print the tokens
    print("[", end="")
    for i,t in zip(range(len(tokens.input_ids[0])), tokens.input_ids[0]):
        if i > 0:
            print(", ", end="")
        print(f"\"{tokenizer.decode([int(t)], skip_special_tokens=True)}\"", end="")
    print("]")

if __name__ == "__main__":
    # parse the arguments
    args = parse_args()

    # execute
    main(args.tokenizer, args.model, args.prompt)

