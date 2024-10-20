# Introduction
These scripts are really about exploring embeddings, specifically input embeddings.

## Print Layers
If we need to print out the layers of an LLM, just call (by default it'll choose Gemma 2B)

```bash
python print_layers.py
```

and for other models such as mistral

```bash
python print_layers.py --model "mistralai/Mistral-7B-v0.1"
```

or llama 7b

```bash
python print_layers.py --model "meta-llama/Llama-2-7b-hf"
```

larger models such as llama-2-70b chat

```bash
python print_layers.py --model "meta-llama/Llama-2-70b-chat-hf" 
```


## Print Tokens
If we need to print out the tokens of an LLM, just call (by default it'll choose Gemma 2B) and the phrase "Who is Ada Lovelace?"

```bash
python print_layers.py
```

and for other models such as mistral

```bash
python print_tokens.py --tokenizer "mistralai/Mistral-7B-v0.1" --prompt "Who is Kitty Purry?"
```

```bash
python print_output.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --tokenizer TinyLlama/TinyLlama-1.1B-Chat-v1.0 --prompt "I like to listen to music on my"  --score 8.2 --probability 0.01
```


## Print Embeddings
There are several scripts we can use to give us details about embeddings

```bash
python load_embeddings.py -tokenizer "meta-llama/Meta-Llama-3-8b-Instruct" --embeddings_file "./output/llama3_8b_embeddings_layer.pth" --prompt "Who is Kitty Purry?" --dimensions 4096 
```

```bash
python print_embeddings_hidden.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --tokenizer TinyLlama/TinyLlama-1.1B-Chat-v1.0 --prompt "who is ada lovelace" 

```bash
python print_embeddings_hidden_all.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --tokenizer TinyLlama/TinyLlama-1.1B-Chat-v1.0 --prompt "who is ada lovelace" 
```

```bash
python print_embeddings_input.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --tokenizer TinyLlama/TinyLlama-1.1B-Chat-v1.0 --prompt "who is ada lovelace" 
```

```bash
python print_embeddings_shape.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --tokenizer TinyLlama/TinyLlama-1.1B-Chat-v1.0 --prompt "who is ada lovelace" 
```

## Extract Embeddings
```bash
python extract_embeddings.py --tokenizer "meta-llama/Meta-Llama-3-8b-Instruct" --model "meta-llama/Meta-Llama-3-8b-Instruct" --embeddings_file "./output/llama3_8b_embeddings_layer.pth" --dimensions 4096
```

```bash
python extract_embeddings_hidden_all.py  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --tokenizer TinyLlama/TinyLlama-1.1B-Chat-v1.0 --prompt "who is ada lovelace" --output_prefix output/embedding
```

```bash
python extract_embeddings_input.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --tokenizer TinyLlama/TinyLlama-1.1B-Chat-v1.0 --prompt "who is ada lovelace" --output_prefix output/input
```

```bash
python visualize_cosine_similarity.py --tokenizer "meta-llama/Meta-Llama-3-8b-Instruct" --model "meta-llama/Meta-Llama-3-8b-Instruct" --embeddings_file "./output/llama3_8b_embeddings_layer.pth" --dimensions 4096 --prompt "Sit Sat Mat Bat Hat Cat Nap Kit Kat Dog Fish Tree Math London Paris Rio Berlin Sydney Moscow Red Blue Green Black White for while print loop"
```


## Rendering Embeddings and Layers

### Extract input and layer embeddings
A python notebook is included to render the extracted embeddings data. To do this you need to run the following scripts 

```bash
mkdir radio

# extract input embeddings as input.txt
python extract_embeddings_input.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --tokenizer TinyLlama/TinyLlama-1.1B-Chat-v1.0 --prompt "I like to listen to music on my" --output_prefix radio/input

# extract embeddings from the hidden layers
python extract_embeddings_hidden_all.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --tokenizer TinyLlama/TinyLlama-1.1B-Chat-v1.0 --prompt "I like to listen to music on my" --output_prefix radio/embeddings
```
That will produce a series of files with the embedding data in the `radio` folder.

### Run the notebook
Start the Jupyter notebook server
```bash
docker run -it --rm -p 8888:8888 -v /home/andy/Documents/embeddings:/home/jovyan/work quay.io/jupyter/minimal-notebook:latest
```
Look for the url with the access token in the output - e.g.
```
http://127.0.0.1:8888/lab?token=58d424acfb90a81bad951f85e2ca23c7624e6cd50b6f8a5f
```
Then open the notebook `render_layers.ipynb` and set the path an dinitial word list, images will be rendered to the folder defined in `png_prefix`.

Click thorugh each section of the notebook and watch the images render.