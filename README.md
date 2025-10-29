# PreFact: Knowledge Propagation Regulating Network Toward Preferred Facts for Knowledge-aware Recommendation
This is our implementation for the paper: PreFact: Knowledge Propagation Regulating Network Toward Preferred Facts for Knowledge-aware Recommendation

## Environment Settings

- torch==2.7.1
- python==3.13.4
- pykeen==1.11.1
- networkx==3.3
- pandas==2.3.0
- prettytable==3.16.0
- setproctitle==1.3.6
- torch-scatter==2.1.2

## Example to run the codes.

1. Config `dataset` and `model_name` in `KGE.py`, then run it to get pretrained KGE embeddings. This is done by python package `pykeen`.

2. Run the following command to train PreFact on Last-FM dateset with GPU 0:

    ```
    python main.py --dataset last-f --gpu_id 0
    ```

3. After training, you can check the log files in `./logs`

## Parameter Tuning

All the parameters are in `./utils/parser.py`
